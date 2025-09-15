
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        # Load and parse annotations
        # Store image paths and corresponding annotations

        #COCO style path
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # image_id to file_name
        #  0: "000000.png",
        #  1: "000001.png",
        #  2: "000002.png",
        self.id2filename = {img["id"]: img["file_name"] for img in coco["images"]}

        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"] #png
            bbox = ann["bbox"]  #[x1, y1, x2, y2]
            label = ann["category_id"] #circle = 0, square = 1, triangle = 2
    

            if img_id not in self.annotations:
                self.annotations[img_id] = {"boxes": [], "labels": []}


            self.annotations[img_id]["boxes"].append(bbox)
            self.annotations[img_id]["labels"].append(label)

        self.image_idx = list(self.id2filename.keys())
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_idx)
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, self.id2filename[img_id])

        # Read the image
        image = Image.open(img_path).convert("RGB")

        ann = self.annotations[img_id]
        boxes = torch.tensor(ann["boxes"], dtype=torch.int64)
        labels = torch.tensor(ann["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        # [3,224,224]
        if self.transform:
            image = self.transform(image)

        return image, target