import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0):
        """
        Initialize the keypoint dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)

    
        self.annotations = data["images"]
       
        self.num_samples = len(self.annotations)
        
        self.num_keypoints = len(self.annotations[0]["keypoints"])
      
    def __len__(self):
        return len(self.annotations)

    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.
        
        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap
            
        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        # For each keypoint:
        # 1. Create 2D gaussian centered at keypoint location
        # 2. Handle boundary cases
        img_h = getattr(self, 'img_h', 128)
        img_w = getattr(self, 'img_w', 128)

        keypoints = np.asarray(keypoints, dtype=np.float32)  # (K,2)
        K = keypoints.shape[0]
        heatmaps = np.zeros((K, height, width), dtype=np.float32) #Tensor of shape [num_keypoints, height, width]

        #grid generate
        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij"
        )

        #(W_heatmap​−1)​ / (W_image - 1)

        sx = (width - 1) / max(1.0, (img_w - 1))
        sy = (height - 1) / max(1.0, (img_h - 1))

        sigma = float(self.sigma)
    
        for k in range(K):
            x, y = keypoints[k]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x < 0 or y < 0 or x > img_w - 1 or y > img_h - 1:
                continue

            #Heatmap
            u = x * sx
            v = y * sy

            # 2D Gaussian：exp(-((x-u)^2 + (y-v)^2)/(2*sigma^2))
            g = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * (sigma ** 2)))
            heatmaps[k] = g 

        return torch.from_numpy(heatmaps).to(torch.float32)
    

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [5, 64, 64] (5 heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [10] (x,y for 5 keypoints, normalized to [0,1])
        """
        ann = self.annotations[idx]
        file_name = ann["file_name"]
        keypoints = np.array(ann["keypoints"], dtype=np.float32)  # (5,2)

        img_path = os.path.join(self.image_dir, file_name)
        img = Image.open(img_path).convert("L").resize((128, 128)) #grayscale
        img = np.array(img, dtype=np.float32) / 255.0
        image = torch.from_numpy(img).unsqueeze(0)  # [1,128,128]

        if self.output_type == "heatmap":
            targets = self.generate_heatmap(
                keypoints=keypoints,
                height=self.heatmap_size,
                width=self.heatmap_size,
            )  # [5,64,64]

        elif self.output_type == "regression":
            # Normalize to [0,1]
            norm_kpts = keypoints.copy()
            norm_kpts[:, 0]= norm_kpts[:, 0] / 128.0  #x / W
            norm_kpts[:, 1]= norm_kpts[:, 1] / 128.0  #y / H
            targets = torch.from_numpy(norm_kpts.reshape(-1)).float()  # [10]

        return image, targets