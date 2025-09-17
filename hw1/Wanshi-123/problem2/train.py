import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from evaluate import compute_pck, extract_keypoints_from_heatmaps, plot_pck_curves, visualize_predictions
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def train_heatmap_model(model, train_loader, val_loader, num_epochs=30):
    """
    Train the heatmap-based model.

    Uses MSE loss between predicted and target heatmaps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val = float("inf")
    logs = {"heatmap": []}

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        total = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)                 
            loss = criterion(preds, targets)  
            loss.backward()
            optimizer.step()

            bsz = imgs.size(0)
            running += loss.item() * bsz
            total += bsz
        train_loss = running / total

        # val
        model.eval()
        running = 0.0
        total = 0
        preds_all = []
        gts_all = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(imgs)
                loss = criterion(preds, targets)
                bsz = imgs.size(0)
                running += loss.item() * bsz
                total += bsz

                #pck
                pred_xy_hm = extract_keypoints_from_heatmaps(preds)   # [B,K,2]
                gt_xy_hm   = extract_keypoints_from_heatmaps(targets)   # [B,K,2]

                Hh, Wh = preds.shape[-2], preds.shape[-1]
                sx = (128 - 1) / max(1, (Wh - 1))
                sy = (128 - 1) / max(1, (Hh - 1))
                pred_xy = pred_xy_hm.float()
                gt_xy   = gt_xy_hm.float()
                pred_xy[..., 0] *= sx; pred_xy[..., 1] *= sy
                gt_xy[...,   0] *= sx; gt_xy[...,   1] *= sy

                preds_all.append(pred_xy.cpu())
                gts_all.append(gt_xy.cpu())

        val_loss = running / max(1, total)

        print(f"Heatmap_model | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        pck_list = []

        preds_all = torch.cat(preds_all, dim=0)  # [N,K,2]
        gts_all   = torch.cat(gts_all,   dim=0)  # [N,K,2]
        pck_this_epoch = compute_pck(preds_all, gts_all, thresholds=(0.05, 0.1, 0.15, 0.2),normalize_by="bbox")
        pck_list.append(pck_this_epoch)
        final_pck = pck_this_epoch  

        # log
        logs["heatmap"].append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "pck": {str(k): float(v) for k, v in pck_this_epoch.items()},
        })

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/heatmap_model.pth")

    return logs, final_pck


def train_regression_model(model, train_loader, val_loader, num_epochs=30):
    """
    Train the direct regression model.

    Uses MSE loss between predicted and target coordinates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val = float("inf")
    logs = {"regression": []}

    for epoch in range(num_epochs):
        # train
        model.train()
        running = 0.0
        total = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)                
            loss = criterion(preds, targets)   
            loss.backward()
            optimizer.step()

            bsz = imgs.size(0)
            running += loss.item() * bsz
            total += bsz
        train_loss = running / max(1, total)

        # val
        model.eval()
        running = 0.0
        total = 0
        preds_all = []
        gts_all = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(imgs)
                loss = criterion(preds, targets)
                bsz = imgs.size(0)
                running += loss.item() * bsz
                total += bsz

                #pck
                pred_xy = preds.view(bsz, 5, 2).float()
                gt_xy   = targets.view(bsz, 5, 2).float()

            
                scale_x = (128 - 1)  # W-1
                scale_y = (128 - 1)  # H-1
                pred_xy[..., 0] *= scale_x
                pred_xy[..., 1] *= scale_y
                gt_xy[..., 0] *= scale_x
                gt_xy[..., 1] *= scale_y

                preds_all.append(pred_xy.cpu())
                gts_all.append(gt_xy.cpu())
        val_loss = running / max(1, total)

        pck_list = []

        preds_all = torch.cat(preds_all, dim=0)  # [N,K,2]
        gts_all   = torch.cat(gts_all,   dim=0)  # [N,K,2]
        pck_this_epoch = compute_pck(preds_all, gts_all, thresholds=(0.05, 0.1, 0.15, 0.2), normalize_by="bbox")
        pck_list.append(pck_this_epoch)

    
        final_pck = pck_this_epoch  

        print(f"Regression model | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # log
        logs["regression"].append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "pck": {str(k): float(v) for k, v in pck_this_epoch.items()}
        })

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/regression_model.pth")

    return logs, final_pck


def main():
    TRAIN_IMAGE_DIR= "datasets/keypoints/train"
    VAL_IMAGE_DIR= "datasets/keypoints/val"
    TRAIN_ANN_FILE = "datasets/keypoints/train_annotations.json"
    VAL_ANN_FILE = "datasets/keypoints/val_annotations.json"

    heat_train_ds = KeypointDataset(TRAIN_IMAGE_DIR, TRAIN_ANN_FILE, output_type="heatmap",heatmap_size=64, sigma=2.0)
    heat_val_ds  = KeypointDataset(VAL_IMAGE_DIR, VAL_ANN_FILE, output_type="heatmap", heatmap_size=64, sigma=2.0)

    reg_train_ds = KeypointDataset(TRAIN_IMAGE_DIR, TRAIN_ANN_FILE, output_type="regression")
    reg_val_ds  = KeypointDataset(VAL_IMAGE_DIR, VAL_ANN_FILE, output_type="regression")

    bs = 32
    heat_train_loader = DataLoader(heat_train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    heat_val_loader  = DataLoader(heat_val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    reg_train_loader  = DataLoader(reg_train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    reg_val_loader = DataLoader(reg_val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

 
    heat_model = HeatmapNet(num_keypoints=5)
    reg_model = RegressionNet(num_keypoints=5)

    # train 
    log_heat, final_pck_heatmap = train_heatmap_model(heat_model, heat_train_loader, heat_val_loader, num_epochs=30)
    log_reg , final_pck_reg = train_regression_model(reg_model, reg_train_loader, reg_val_loader, num_epochs=30)

    plot_pck_curves(final_pck_heatmap, final_pck_reg, "results/visualizations/pck_curve.png")

    merged = {}
    merged.update(log_heat)
    merged.update(log_reg)
    with open("results/training_log.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()