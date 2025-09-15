import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.
    
    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size
        
    Returns:
        anchors: List of tensors, each of shape [num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    # For each feature map:
    # 1. Create grid of anchor centers
    # 2. Generate anchors with specified scales and ratios
    # 3. Convert to absolute coordinates

    anchors_all = []

    for (H, W),scales in zip(feature_map_sizes, anchor_scales):
        stride = image_size / H 
        xs = (torch.arange(W) + 0.5) * stride
        ys = (torch.arange(H) + 0.5) * stride
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")

        cx = grid_x.reshape(-1)
        cy = grid_y.reshape(-1)

        level_anchors = []
        for s in scales:  # 1:1 anchors
            w = s
            h = s
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            boxes = torch.stack([x1,y1,x2,y2], dim=1)
            boxes[:,0::2] = boxes[:,0::2].clamp(0, image_size)
            boxes[:,1::2] = boxes[:,1::2].clamp(0, image_size)
            level_anchors.append(boxes)

        anchors_all.append(torch.cat(level_anchors, dim=0))

    return anchors_all

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))


    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])   
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  

    wh = (rb - lt).clamp(min=0)      # [N, M, 2]

    inter = wh[..., 0]* wh[..., 1]       # [N, M]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))  # [N]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)) # [M]

    union = (area1[:, None] + area2[None, :] - inter) # [N, M]

    iou = inter / union                                         # [N, M]
    return iou

def match_anchors_to_targets(anchors, target_boxes, target_labels, 
                            pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.
    Calculate Iou to choose anchors in three anchors
    
    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors
        
    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    A = anchors.shape[0]

    #Init
    matched_labels = torch.zeros(A, dtype=torch.long)
    matched_boxes  = torch.zeros(A, 4, dtype=anchors.dtype)

    if target_boxes.numel() == 0:
        pos_mask = torch.zeros(A, dtype=torch.bool)
        neg_mask = torch.ones(A,  dtype=torch.bool)
        return matched_labels, matched_boxes, pos_mask, neg_mask

    iou_mat = compute_iou(anchors, target_boxes)

    max_iou, gt_idx = iou_mat.max(dim=1)  

    pos_mask =(max_iou >= pos_threshold)
    neg_mask =(max_iou < neg_threshold)


    best_anchor_for_each_gt = iou_mat.argmax(dim=0)  
    pos_mask[best_anchor_for_each_gt] = True
    gt_idx[best_anchor_for_each_gt] = torch.arange(target_boxes.shape[0])

    matched_boxes[pos_mask]  = target_boxes[gt_idx[pos_mask]]
    matched_labels[pos_mask] = target_labels[gt_idx[pos_mask]] + 1  

    neg_mask = torch.logical_and(neg_mask, ~pos_mask)

    return matched_labels, matched_boxes, pos_mask, neg_mask