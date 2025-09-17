import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extract (x, y) coordinates from heatmaps.

    Args:
        heatmaps: Tensor of shape [batch, num_keypoints, H, W]

    Returns:
        coords: Tensor of shape [batch, num_keypoints, 2]
    """
    B, K, H, W = heatmaps.shape

    # flatten each [H, W] -> [H*W], take argmax -> [B, K]
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)      # [B, K], 0..H*W-1

    # Convert to (x, y) coordinates (heatmap grid)
    y = (idx // W).to(torch.float32)       # row index -> y
    x = (idx %  W).to(torch.float32)       # col index -> x

    coords = torch.stack([x, y], dim=-1)   # [B, K, 2], (x, y)
    return coords



def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    """
    Compute PCK at various thresholds.

    Args:
        predictions: Tensor of shape [N, num_keypoints, 2]
        ground_truths: Tensor of shape [N, num_keypoints, 2]
        thresholds: List of threshold values (as fraction of normalization)
        normalize_by: 'bbox' for bounding box diagonal, 'torso' for torso length

    Returns:
        pck_values: Dict mapping threshold to accuracy
    """
    # to tensors (float32)
    pred = predictions if isinstance(predictions, torch.Tensor) else torch.tensor(predictions, dtype=torch.float32)
    gt   = ground_truths if isinstance(ground_truths, torch.Tensor) else torch.tensor(ground_truths, dtype=torch.float32)
    pred = pred.to(torch.float32)
    gt   = gt.to(torch.float32)

    N, K, _ = gt.shape

    # Effective gt>= 0 
    valid = (gt[..., 0] >= 0) & (gt[..., 1] >= 0)  # [N, K]

    # distance [N, K]
    dists = torch.linalg.norm(pred - gt, dim=-1)

    
    if normalize_by == 'torso' and K >= 5:
    
        head = gt[:, 0, :]          
        feet_mid = (gt[:, 3, :] + gt[:, 4, :]) / 2.0
        torso_len = torch.linalg.norm(feet_mid - head, dim=-1)  #
        bad = ~torch.isfinite(torso_len) | (torso_len <= 1e-6)
        if bad.any():
            normalize_by = 'bbox'
        norm = torso_len
    if normalize_by == 'bbox':
        x_min, _ = torch.min(gt[..., 0].where(valid, torch.inf), dim=1)  # [N]
        y_min, _ = torch.min(gt[..., 1].where(valid, torch.inf), dim=1)
        x_max, _ = torch.max(gt[..., 0].where(valid, -torch.inf), dim=1)
        y_max, _ = torch.max(gt[..., 1].where(valid, -torch.inf), dim=1)
        w = (x_max - x_min).clamp_min(1e-6)
        h = (y_max - y_min).clamp_min(1e-6)
        norm = torch.sqrt(w * w + h * h) 

    norm_mat = norm.unsqueeze(1).expand_as(dists).clamp_min(1e-6)

    total_valid = valid.sum().item()
    if total_valid == 0:
        return {float(t): 0.0 for t in thresholds}

    pck_values = {}
    for t in thresholds:
        thr = float(t)
        hits = (dists <= thr * norm_mat) & valid  # [N, K]
        acc = hits.sum().item() / total_valid
        pck_values[thr] = acc

    return pck_values


def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    """
    Plot PCK curves comparing both methods.
    """
    th_hm = sorted(pck_heatmap.keys())
    th_rg = sorted(pck_regression.keys())
    x_hm  = th_hm
    y_hm  = [pck_heatmap[t] for t in x_hm]
    x_rg  = th_rg
    y_rg  = [pck_regression[t] for t in x_rg]

    plt.figure(figsize=(6, 4))
    plt.plot(x_hm, y_hm, marker='o', linewidth=2, label='Heatmap')
    plt.plot(x_rg, y_rg, marker='s', linewidth=2, label='Regression')
    plt.xlabel('Threshold (fraction of normalization)')
    plt.ylabel('PCK (accuracy)')
    plt.title('PCK Curves: Heatmap vs. Regression')
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    """
    Visualize predicted and ground truth keypoints on image.
    """

    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim == 3 and img.size(0) == 1:
            img = img.squeeze(0)  # [H,W]
        img = img.numpy()
    else:
        img = np.asarray(image)

    # norm [0,1]
    img_min, img_max = float(np.min(img)), float(np.max(img))
    if img_max > img_min:
        disp = (img - img_min) / (img_max - img_min)
    else:
        disp = img

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    pred = to_np(pred_keypoints)
    gt   = to_np(gt_keypoints)

    plt.figure(figsize=(4, 4))
    plt.imshow(disp, cmap='gray', vmin=0, vmax=1)
    # GT o
    if gt.size > 0:
        plt.scatter(gt[:, 0], gt[:, 1], s=30, marker='o', edgecolors='k', facecolors='none', label='GT')
    # Pred x
    if pred.size > 0:
        plt.scatter(pred[:, 0], pred[:, 1], s=40, marker='x', c='r', label='Pred')

    plt.axis('off')
    plt.legend(loc='lower right', frameon=True, fontsize=8)
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
