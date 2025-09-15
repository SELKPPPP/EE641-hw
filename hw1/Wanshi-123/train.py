import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from utils import generate_anchors
from loss import DetectionLoss

def train_epoch(model, dataloader, criterion, optimizer, device, anchors):
    """Train for one epoch."""
    model.train()
    # Training loop
    total_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        for t in targets: 
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)

        optimizer.zero_grad()
        predictions = model(images)  # list of [B, A*(5+C), H, W]
        loss_dict = criterion(predictions, targets, anchors)
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, anchors):
    """Validate the model."""
    model.eval()
    # Validation loop
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)

            predictions = model(images)
            loss_dict = criterion(predictions, targets, anchors)
            total_loss += loss_dict["loss_total"].item()
    return total_loss / len(dataloader)

def main():
    # Configuration
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset, model, loss, optimizer
    # Training loop with logging
    # Save best model and training log
    train_dataset = ShapeDetectionDataset("datasets/detection/train", "datasets/detection/train_annotations.json")
    val_dataset   = ShapeDetectionDataset("datasets/detection/val", "datasets/detection/val_annotations.json")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = MultiScaleDetector(num_classes=3).to(device)

    # anchors
    feature_map_sizes = [(56,56), (28,28), (14,14)]
    anchor_scales = [[16,24,32], [48,64,96], [96,128,192]]
    anchors = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)

    # loss & optimizer
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    os.makedirs("results", exist_ok=True)
    log_file = "results/training_log.json"
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, anchors)
        val_loss = validate(model, val_loader, criterion, device, anchors)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model.pth")

        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
