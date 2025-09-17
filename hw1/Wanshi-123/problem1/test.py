from evaluate import predict_on_batch, compute_ap
from torch.utils.data import DataLoader
from dataset import ShapeDetectionDataset
from utils import generate_anchors
from model import MultiScaleDetector
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


val_dataset = ShapeDetectionDataset("datasets/detection/val", "datasets/detection/val_annotations.json")
val_loader  = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b]))
model = MultiScaleDetector(num_classes=3).to(device)
model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
model.eval()
feature_map_sizes = [(56,56), (28,28), (14,14)]
anchor_scales = [[16,24,32], [48,64,96], [96,128,192]]
anchors = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)

all_pred_cls0, all_gt_cls0 = [], []
img_id = 0
for images, targets in val_loader:
    images = images.to(device)
    batch_preds = predict_on_batch(model, images, anchors, score_thr=0.3, iou_thr=0.5)
    B = images.shape[0]
    for b in range(B):
        mask0 = (batch_preds[b]['labels'] == 0)
        all_pred_cls0.append({
            'image_id': img_id,
            'boxes': batch_preds[b]['boxes'][mask0].detach().cpu(),
            'scores': batch_preds[b]['scores'][mask0].detach().cpu()
        })
    
        gt_mask0 = (targets[b]['labels'] == 0)
        all_gt_cls0.append({
            'image_id': img_id,
            'boxes': targets[b]['boxes'][gt_mask0]
        })
        img_id += 1

ap0, prec0, rec0 = compute_ap(all_pred_cls0, all_gt_cls0, iou_threshold=0.5)
print(f"AP(class=0 @ IoU=0.5) = {ap0:.4f}")

from evaluate import visualize_detections
from PIL import Image


image_path = "datasets/detection/val/001001.png"
visualize_detections(image_path, batch_preds[b], targets[b], save_path="results/vis/000123.png")


from evaluate import analyze_scale_performance
analyze_scale_performance(model, val_loader, anchors, iou_thr=0.5, score_thr=0.05)
