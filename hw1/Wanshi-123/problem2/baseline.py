import os
import json
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet 
from train import train_heatmap_model
from evaluate import (
    extract_keypoints_from_heatmaps,
    compute_pck,
    visualize_predictions,
)


def ablation_study(dataset, model_class):
    """
    Conduct ablation studies on key hyperparameters.
    
    Experiments to run:
    1. Effect of heatmap resolution (32x32 vs 64x64 vs 128x128)
    2. Effect of Gaussian sigma (1.0, 2.0, 3.0, 4.0)
    3. Effect of skip connections (with vs without)
    """
    # Run experiments and save results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"heatmap_resolution": {}, "sigma": {}, "skip": {}}

    train_image_dir = dataset["train_image_dir"]
    val_image_dir = dataset["val_image_dir"]
    train_ann = dataset["train_ann"]
    val_ann = dataset["val_ann"]

    #for hm_size in (32, 64, 128):
       # train_ds = KeypointDataset(train_image_dir, train_ann, output_type="heatmap", heatmap_size=hm_size, sigma=2.0)
       # val_ds   = KeypointDataset(val_image_dir,   val_ann,   output_type="heatmap", heatmap_size=hm_size, sigma=2.0)

       # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4 ,pin_memory=True)
       # val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

       # model = model_class(num_keypoints=5, heatmap_size=hm_size).to(device)

        # 30 epochs
      #  _ = train_heatmap_model(model, train_loader, val_loader, num_epochs=30)



def analyze_failure_cases(model, test_loader):
    """
    Identify and visualize failure cases.
    
    Find examples where:
    1. Heatmap succeeds but regression fails
    2. Regression succeeds but heatmap fails
    3. Both methods fail
    """
    pass