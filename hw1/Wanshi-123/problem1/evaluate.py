# evaluate.py #AI-supported
import os
import math
import json
from typing import List, Dict, Tuple
from utils import match_anchors_to_targets
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from utils import compute_iou  



def _xyxy_to_cxcywh(boxes: torch.Tensor):
    x1, y1, x2, y2 = boxes.unbind(-1)
    w  = (x2 - x1).clamp(min=1e-6)
    h  = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)

def _cxcywh_to_xyxy(boxes: torch.Tensor):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _decode_offsets(anchors_xyxy: torch.Tensor, deltas: torch.Tensor):

    a = _xyxy_to_cxcywh(anchors_xyxy)
    tx, ty, tw, th = deltas.unbind(-1)
    cx = tx * a[:, 2] + a[:, 0]
    cy = ty * a[:, 3] + a[:, 1]
    w  = torch.exp(tw) * a[:, 2]
    h  = torch.exp(th) * a[:, 3]
    g  = torch.stack([cx, cy, w, h], dim=-1)
    return _cxcywh_to_xyxy(g)

def _torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float):

    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_thr)
    except Exception:

        keep = []
        idxs = scores.argsort(descending=True)
        while idxs.numel() > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.numel() == 1:
                break
            ious = compute_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
            remain = (ious <= iou_thr).nonzero(as_tuple=False).squeeze(1)
            idxs = idxs[remain + 1]
        return torch.tensor(keep, device=boxes.device)


@torch.no_grad()
def predict_on_batch(model,
                     images: torch.Tensor,
                     anchors: List[torch.Tensor],
                     score_thr: float = 0.3,
                     iou_thr: float = 0.5,
                     topk_per_class: int = 200) -> List[Dict]:
  

    device = next(model.parameters()).device
    model.eval()
    preds = model(images.to(device))  # list of 3: [B, A*(5+C), H, W]

    B = images.shape[0]
    C = (preds[0].shape[1] // anchors[0].shape[0]) 
   

    results = []
    for b in range(B):
        boxes_all, scores_all, labels_all, scales_all = [], [], [], []
        for scale_idx, (p_map, anc) in enumerate(zip(preds, anchors)):
            _, ch, H, W = p_map.shape
       
            A = anc.shape[0] // (H * W)
            ch_per_anchor = ch // A
            C = ch_per_anchor - 5

       
            p = p_map[b].view(A, ch_per_anchor, H, W).permute(2, 3, 0, 1).reshape(H * W * A, ch_per_anchor)
            loc = p[:, :4]
            obj_logit = p[:, 4]
            cls_logit = p[:, 5:]

            obj_prob = torch.sigmoid(obj_logit)                     # [N]
            cls_prob = F.softmax(cls_logit, dim=-1)                 # [N, C]
            scores = obj_prob.unsqueeze(1) * cls_prob               # [N, C]

       
            boxes = _decode_offsets(anc.to(device), loc)

            for c in range(C):
                sc = scores[:, c]
                keep = sc > score_thr
                if keep.any():
                    b_c = boxes[keep]
                    s_c = sc[keep]
                    idx = _torch_nms(b_c, s_c, iou_thr)
                    if topk_per_class is not None and idx.numel() > topk_per_class:
                        idx = idx[:topk_per_class]
                    boxes_all.append(b_c[idx])
                    scores_all.append(s_c[idx])
                    labels_all.append(torch.full((idx.numel(),), c, dtype=torch.long, device=device))
                    scales_all.append(torch.full((idx.numel(),), scale_idx, dtype=torch.long, device=device))

        if len(boxes_all) == 0:
            results.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device),
                'scales': torch.zeros((0,), dtype=torch.long, device=device),
            })
        else:
            results.append({
                'boxes': torch.cat(boxes_all, dim=0),
                'scores': torch.cat(scores_all, dim=0),
                'labels': torch.cat(labels_all, dim=0),
                'scales': torch.cat(scales_all, dim=0),
            })
    return results



def compute_ap(predictions: List[Dict],
               ground_truths: List[Dict],
               iou_threshold: float = 0.5) -> Tuple[float, torch.Tensor, torch.Tensor]:

    gt_by_img = {}
    for g in ground_truths:
        img_id = g['image_id']
        gt_by_img[img_id] = {
            'boxes': g['boxes'].clone(),
            'used': torch.zeros((g['boxes'].shape[0],), dtype=torch.bool)
        }
    num_gt = sum(g['boxes'].shape[0] for g in ground_truths)

    all_boxes, all_scores, all_imgids = [], [], []
    for p in predictions:
        if p['boxes'].numel() == 0:
            continue
        n = p['boxes'].shape[0]
        all_boxes.append(p['boxes'])
        all_scores.append(p['scores'])
        all_imgids.append(torch.full((n,), p['image_id'], dtype=torch.long, device=p['boxes'].device))
    if len(all_boxes) == 0:
   
        return 0.0, torch.tensor([0.0]), torch.tensor([0.0])

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    imgids = torch.cat(all_imgids, dim=0)

    order = scores.argsort(descending=True)
    boxes = boxes[order]
    scores = scores[order]
    imgids = imgids[order]

    tp = torch.zeros((boxes.shape[0],), dtype=torch.float32, device=boxes.device)
    fp = torch.zeros_like(tp)

    for i in range(boxes.shape[0]):
        img_id = int(imgids[i].item())
        if img_id not in gt_by_img or gt_by_img[img_id]['boxes'].numel() == 0:
            fp[i] = 1.0
            continue

        gt_boxes = gt_by_img[img_id]['boxes']
        used = gt_by_img[img_id]['used']
        ious = compute_iou(boxes[i].unsqueeze(0), gt_boxes)[0]  

        iou_max, j = (ious.max().item(), int(ious.argmax().item())) if ious.numel() > 0 else (0.0, -1)
        if iou_max >= iou_threshold and not used[j]:
            tp[i] = 1.0
            gt_by_img[img_id]['used'][j] = True 
        else:
            fp[i] = 1.0


    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)


    mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
   
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze(1)
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return ap, precision.detach().cpu(), recall.detach().cpu()



def visualize_detections(image, predictions: Dict, ground_truths: Dict, save_path: str):


    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')
    draw = ImageDraw.Draw(img)


    if ground_truths and 'boxes' in ground_truths and ground_truths['boxes'].numel() > 0:
        for x1, y1, x2, y2 in ground_truths['boxes'].tolist():
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)


    if predictions and 'boxes' in predictions and predictions['boxes'].numel() > 0:
        for (x1, y1, x2, y2), score, label in zip(
            predictions['boxes'].tolist(),
            predictions['scores'].tolist(),
            predictions['labels'].tolist()
        ):
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            txt = f"{int(label)}:{score:.2f}"
            tw = draw.textlength(txt); th = 12
            draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=(255, 0, 0))
            draw.text((x1 + 2, y1 - th - 1), txt, fill=(255, 255, 255))


    anchors = predictions.get('anchors', None)
    if anchors is not None and ground_truths and ground_truths['boxes'].numel() > 0:
   
        if isinstance(anchors, (list, tuple)):
            scale_ids, anchors_all = [], []
            for i, a in enumerate(anchors):
                anchors_all.append(a)
                scale_ids.append(torch.full((a.shape[0],), i, dtype=torch.long))
            anchors_all = torch.cat(anchors_all, 0).cpu()
            scale_ids = torch.cat(scale_ids, 0)
        else:
            anchors_all = anchors.cpu()
            scale_ids   = torch.zeros((anchors_all.shape[0],), dtype=torch.long)

        gt_boxes  = ground_truths['boxes'].float().cpu()
        gt_labels = ground_truths['labels'].long().cpu()

        pos_thr = float(predictions.get('pos_threshold', 0.5))
        neg_thr = float(predictions.get('neg_threshold', 0.3))

        matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
            anchors_all, gt_boxes, gt_labels, pos_threshold=pos_thr, neg_threshold=neg_thr
        )

   
        palette = [(66,133,244), (219,68,55), (244,180,0)]  
        for idx in pos_mask.nonzero(as_tuple=False).squeeze(1).tolist():
            x1,y1,x2,y2 = anchors_all[idx].tolist()
            c = palette[int(scale_ids[idx]) % len(palette)]
            draw.rectangle([x1, y1, x2, y2], outline=c, width=2)

     
        if predictions.get('draw_neg', False):
            import numpy as np
            neg_idx = neg_mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
            k = min(int(predictions.get('neg_sample', 150)), neg_idx.size)
            if k > 0:
                for idx in np.random.choice(neg_idx, size=k, replace=False).tolist():
                    x1,y1,x2,y2 = anchors_all[idx].tolist()
                    draw.rectangle([x1, y1, x2, y2], outline=(170,170,170), width=1)


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


@torch.no_grad()
def analyze_scale_performance(model, dataloader, anchors: List[torch.Tensor],
                              iou_thr: float = 0.5,
                              score_thr: float = 0.3) -> Dict:
    
    device = next(model.parameters()).device

    K = len(anchors)  
    scale_tp = [0] * K
    scale_pred = [0] * K
    size_tp = {k: {'small': 0, 'medium': 0, 'large': 0} for k in range(K)}

    def size_bucket(box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        w = (x2 - x1)
        h = (y2 - y1)
        s = max(w, h)
        if s <= 32:
            return 'small'
        elif s <= 96:
            return 'medium'
        else:
            return 'large'

    for images, targets in dataloader:
        images = images.to(device)
        preds = predict_on_batch(model, images, [a.to(device) for a in anchors],
                                 score_thr=score_thr, iou_thr=0.5)

        B = len(targets)
        for b in range(B):
            gt_boxes = targets[b]['boxes'].to(device).float()
            gt_labels = targets[b]['labels'].to(device).long()

            if preds[b]['boxes'].numel() == 0:
                continue

      
            boxes = preds[b]['boxes']
            scores = preds[b]['scores']
            labels = preds[b]['labels']
            scales = preds[b]['scales']

    
            for c in torch.unique(labels).tolist():
                mask_c = labels == c
                if mask_c.sum() == 0:
                    continue
                b_c = boxes[mask_c]
                s_c = scores[mask_c]
                sc_c = scales[mask_c]

                order = s_c.argsort(descending=True)
                b_c = b_c[order]
                s_c = s_c[order]
                sc_c = sc_c[order]

          
                gt_mask = (gt_labels == c)
                gt_c = gt_boxes[gt_mask]
                used = torch.zeros((gt_c.shape[0],), dtype=torch.bool, device=device)

                for i in range(b_c.shape[0]):
                    scale_pred[int(sc_c[i].item())] += 1 
                    if gt_c.numel() == 0:
                        continue
                    ious = compute_iou(b_c[i].unsqueeze(0), gt_c)[0]
                    j = int(torch.argmax(ious).item())
                    if ious[j] >= iou_thr and not used[j]:
                        used[j] = True
                        sc = int(sc_c[i].item())
                        scale_tp[sc] += 1
                        bucket = size_bucket(gt_c[j].tolist())
                        size_tp[sc][bucket] += 1

    report = {
        'per_scale': {
            k: {
                'tp': scale_tp[k],
                'pred': scale_pred[k],
                'precision': (scale_tp[k] / scale_pred[k]) if scale_pred[k] > 0 else 0.0,
                'size_tp': size_tp[k]
            } for k in range(K)
        }
    }

    print("\n[Scale analysis]")
    for k in range(K):
        r = report['per_scale'][k]
        print(f"  Scale {k}: TP={r['tp']}  Pred={r['pred']}  Precision={r['precision']:.3f}  "
              f"S/M/L: {r['size_tp']['small']}/{r['size_tp']['medium']}/{r['size_tp']['large']}")
    return report
