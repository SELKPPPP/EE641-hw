import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets

def _xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)

def _encode_offsets(anchors, gt_boxes, eps=1e-6):
    a = _xyxy_to_cxcywh(anchors)
    g = _xyxy_to_cxcywh(gt_boxes)
    tx = (g[:, 0] - a[:, 0]) / (a[:, 2].clamp(min=eps))
    ty = (g[:, 1] - a[:, 1]) / (a[:, 3].clamp(min=eps))
    tw = torch.log(g[:, 2].clamp(min=eps) / a[:, 2].clamp(min=eps))
    th = torch.log(g[:, 3].clamp(min=eps) / a[:, 3].clamp(min=eps))
    return torch.stack([tx, ty, tw, th], dim=-1)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets, anchors):
        """
        Compute multi-task loss.
        
        Args:
            predictions: List of tensors from each scale
            targets: List of dicts with 'boxes' and 'labels' for each image
            anchors: List of anchor tensors for each scale
            
        Returns:
            loss_dict: Dict containing:
                - loss_obj: Objectness loss
                - loss_cls: Classification loss  
                - loss_loc: Localization loss
                - loss_total: Weighted sum
        """
        # For each prediction scale:
        # 1. Match anchors to targets
        # 2. Compute objectness loss (BCE)
        # 3. Compute classification loss (CE) for positive anchors
        # 4. Compute localization loss (Smooth L1) for positive anchors
        # 5. Apply hard negative mining (3:1 ratio)
        B = predictions[0].shape[0]
        C = self.num_classes
        ch_per_anchor = 5 + C

        preds_flat = []
        for pred, anchors in zip(predictions, anchors):
            B_, ch, H, W = pred.shape
            A = anchors.shape[0] // (H * W)
            pred = pred.view(B, A, ch_per_anchor, H, W).permute(0, 3, 4, 1, 2)
            pred = pred.reshape(B, H*W*A, ch_per_anchor)
            preds_flat.append(pred)
        preds_all = torch.cat(preds_flat, dim=1)    # [B, A_total, 5+C]
        anchors_all = torch.cat(anchors, 0)   # [A_total, 4]

        pred_loc = preds_all[..., 0:4]
        pred_obj = preds_all[..., 4]
        pred_cls = preds_all[..., 5:]

        total_loss_obj, total_loss_cls, total_loss_loc = 0.0, 0.0, 0.0

        for b in range(B):
            t = targets[b]
            gt_boxes, gt_labels = t["boxes"], t["labels"]

            matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                anchors_all, gt_boxes, gt_labels
            )

            # obj loss
            obj_target = torch.zeros_like(pred_obj[b])
            obj_target[pos_mask] = 1.0
            loss_obj = F.binary_cross_entropy_with_logits(pred_obj[b], obj_target)

            # cls loss
            if pos_mask.any():
                cls_logits =pred_cls[b][pos_mask]
                cls_target = matched_labels[pos_mask] - 1
                loss_cls = F.cross_entropy(cls_logits, cls_target)
                # loc loss
                loc_target = _encode_offsets(anchors_all[pos_mask], matched_boxes[pos_mask])
                loss_loc = F.smooth_l1_loss(pred_loc[b][pos_mask], loc_target)
            else:
                loss_cls = torch.tensor(0.0)
                loss_loc = torch.tensor(0.0)

            total_loss_obj += loss_obj
            total_loss_cls += loss_cls
            total_loss_loc += loss_loc

        loss_obj = total_loss_obj / B
        loss_cls = total_loss_cls / B
        loss_loc = total_loss_loc / B
        loss_total = loss_obj + loss_cls + 2.0 * loss_loc 

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_loc": loss_loc,
            "loss_total": loss_total
        }
    
    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples.
        
        Args:
            loss: Loss values for all anchors
            pos_mask: Boolean mask for positive anchors
            neg_mask: Boolean mask for negative anchors
            ratio: Negative to positive ratio
            
        Returns:
            selected_neg_mask: Boolean mask for selected negatives
        """
        num_pos = int(pos_mask.sum().item())
        num_neg_can = int(neg_mask.sum().item())

        num_neg_keep =min(num_neg_can, max(1, ratio * max(1, num_pos)))

        if num_neg_keep == 0:
            return torch.zeros_like(neg_mask)


        neg_losses = loss.clone()
        neg_losses[~neg_mask] = -1e9   

        _, idx =torch.topk(neg_losses, num_neg_keep)
        selected = torch.zeros_like(neg_mask)
        selected[idx] = True

        return selected