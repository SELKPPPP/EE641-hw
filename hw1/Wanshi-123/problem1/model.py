import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.m(x)


class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        """
        Initialize the multi-scale detector.
        
        Args:
            num_classes: Number of object classes (not including background)
            num_anchors: Number of anchors per spatial location
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction backbone
        # Extract features at 3 different scales
        
        # Detection heads for each scale
        # Each head outputs: [batch, num_anchors * (4 + 1 + num_classes), H, W]
        out_ch = num_anchors * (5 + num_classes)


        # Block1 (stem): 224 -> 112
        self.stem1 = ConvBNReLU(3, 32, k=3, s=1, p=1)
        self.stem2 = ConvBNReLU(32, 64, k=3, s=2, p=1)  

        # Block2: 112 -> 56  (Scale 1)
        self.block2 = ConvBNReLU(64, 128, k=3, s=2, p=1) 

        # Block3: 56 -> 28   (Scale 2 )
        self.block3 = ConvBNReLU(128, 256, k=3, s=2, p=1)

        # Block4: 28 -> 14   (Scale 3)
        self.block4 = ConvBNReLU(256, 512, k=3, s=2, p=1)

    
        # Scale 1 head 
        self.head1_conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) #3*3 Conv
        self.head1_pred = nn.Conv2d(128, out_ch, kernel_size=1, stride=1, padding=0) #1*1 Conv

        # Scale 2 head 
        self.head2_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.head2_pred = nn.Conv2d(256, out_ch, kernel_size=1, stride=1, padding=0)

        # Scale 3 head
        self.head3_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.head3_pred = nn.Conv2d(512, out_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 3, 224, 224]
            
        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
            where 5 = 4 bbox coords + 1 objectness score
        """

        """
        x: [B, 3, 224, 224]
        return: [p1, p2, p3]
          p1: [B, A*(5+C), 56, 56]
          p2: [B, A*(5+C), 28, 28]
          p3: [B, A*(5+C), 14, 14]
        """

        x = self.stem1(x)     # [B, 32, 224, 224]
        x = self.stem2(x)     # [B, 64, 112, 112]

        f1 = self.block2(x)   # [B, 128, 56, 56]   -> Scale 1
        f2 = self.block3(f1)  # [B, 256, 28, 28]   -> Scale 2
        f3 = self.block4(f2)  # [B, 512, 14, 14]   -> Scale 3

        # Heads
        h1 = torch.relu(self.head1_conv(f1))
        p1 = self.head1_pred(h1)

        h2 = torch.relu(self.head2_conv(f2))
        p2 = self.head2_pred(h2)

        h3 = torch.relu(self.head3_conv(f3))
        p3 = self.head3_pred(h3)

        return [p1, p2, p3]