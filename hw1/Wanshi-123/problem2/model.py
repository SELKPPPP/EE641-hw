import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the heatmap regression network.

        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        # ----- Encoder -----
        # Input: [batch, 1, 128, 128]
        # Conv1: Conv(1→32) → BN → ReLU → MaxPool (128→64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        #Conv2: Conv(32→64) → BN → ReLU → MaxPool (64→32)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Conv3: Conv(64→128) → BN → ReLU → MaxPool (32→16)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Conv4: Conv(128→256) → BN → ReLU → MaxPool (16→8)
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # ----- Decoder -----
        # Deconv4: ConvTranspose(256→128) → BN → ReLU (8→16)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Deconv3: ConvTranspose(256→64) → BN → ReLU (16→32)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Deconv2: ConvTranspose(128→32) → BN → ReLU (32→64)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final: Conv(32→num_keypoints) (no activation)
        # Output: [batch, num_keypoints, 64, 64]
        self.final = nn.Conv2d(32, self.num_keypoints, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, 1, 128, 128]

        Returns:
            heatmaps: Tensor of shape [batch, num_keypoints, 64, 64]
        """
        # ----- Encoder -----
        e1 = self.enc1(x)       
        p1 = self.pool1(e1)        

        e2 = self.enc2(p1)        
        p2 = self.pool2(e2)      

        e3 = self.enc3(p2)        
        p3 = self.pool3(e3)   

        e4 = self.enc4(p3)         
        p4 = self.pool4(e4)        

        # ----- Decoder -----
        # Skip connections between encoder and decoder
        u4 = self.deconv4(p4)                 
        x4 = torch.cat([u4, p3], dim=1)        

        u3 = self.deconv3(x4)                
        x3 = torch.cat([u3, p2], dim=1)      

        u2 = self.deconv2(x3)              

        heatmaps = self.final(u2)           
        return heatmaps



class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the direct regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Use same encoder architecture as HeatmapNet
        # But add global pooling and fully connected layers
        # Output: [batch, num_keypoints * 2]


        # ----- Encoder -----
        # Input: [batch, 1, 128, 128]
        # Conv1: Conv(1→32) → BN → ReLU → MaxPool (128→64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        #Conv2: Conv(32→64) → BN → ReLU → MaxPool (64→32)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Conv3: Conv(64→128) → BN → ReLU → MaxPool (32→16)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Conv4: Conv(128→256) → BN → ReLU → MaxPool (16→8)
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B,256,1,1]

        # FC1: Linear(256→128) → ReLU → Dropout(0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        #FC2: Linear(128→64) → ReLU → Dropout(0.5)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        #FC3: Linear(64→num_keypoints*2) → Sigmoid
        self.fc3 = nn.Sequential(
            nn.Linear(64, self.num_keypoints * 2),
            nn.Sigmoid(),  # output in [0,1]
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            coords: Tensor of shape [batch, num_keypoints * 2]
                   Values in range [0, 1] (normalized coordinates)
        """
        e1 = self.enc1(x)    
        p1 = self.pool1(e1) 

        e2 = self.enc2(p1)   
        p2 = self.pool2(e2) 

        e3 = self.enc3(p2)   
        p3 = self.pool3(e3) 

        e4 = self.enc4(p3)   
        p4 = self.pool4(e4) 

        z = self.gap(p4).view(x.size(0), 256)  # [B,256]

        z = self.fc1(z)      # [B,128]
        z = self.fc2(z)      # [B,64]
        coords = self.fc3(z) # [B, 2*K] in [0,1]

        return coords