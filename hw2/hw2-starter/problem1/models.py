"""
GAN models for font generation.
"""

import torch
import torch.nn as nn
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embedded = nn.Embedding(num_classes, num_features * 2)
        self.gamma_embed = nn.Linear(num_features * 2, num_features)
        self.beta_embed = nn.Linear(num_features * 2, num_features)

    def forward(self, x, y):
        if y.dim() == 2:
            y = y.argmax(dim=1)
        y = y.long()     
        out = self.bn(x)
        embed = self.embedded(y)
        gamma = self.gamma_embed(embed).unsqueeze(2).unsqueeze(3)
        beta = self.beta_embed(embed).unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100, conditional=False, num_classes=26):
        """
        Generator network that produces 28×28 letter images.
        
        Args:
            z_dim: Dimension of latent vector z
            conditional: If True, condition on letter class
            num_classes: Number of letter classes (26)
        """
        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        self.fc = nn.Linear(z_dim + (num_classes if conditional else 0), 128 * 7 * 7)
        self.num_classes = num_classes
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.cbn1 = ConditionalBatchNorm2d(64, num_classes) if conditional else nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        
        # Calculate input dimension
        input_dim = z_dim + (num_classes if conditional else 0)
        
        # Architecture proven to work well for this task:
        # Project and reshape: z → 7×7×128
        self.project = nn.Sequential(
            nn.Linear(input_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsample: 7×7×128 → 14×14×64 → 28×28×1
        self.main = nn.Sequential(
            # TODO: Implement upsampling layers
            # Use ConvTranspose2d with appropriate padding/stride
            # Include BatchNorm2d and ReLU (except final layer)
            # Final layer should use Tanh activation

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14x14 -> 28x28
            nn.Tanh()
        )
    
    def forward(self, z, class_label=None):
        """
        Generate images from latent code.
        
        Args:
            z: Latent vectors [batch_size, z_dim]
            class_label: One-hot encoded class labels [batch_size, num_classes]
        
        Returns:
            Generated images [batch_size, 1, 28, 28] in range [-1, 1]
        """
        # TODO: Implement forward pass
        # If conditional, concatenate z and class_label
        # Project to spatial dimensions
        # Apply upsampling network
        x = self.fc(torch.cat([z, class_label], dim=1) if self.conditional and class_label is not None else z)
        x = x.view(-1, 128, 7, 7)  # Reshape to [batch_size, 128, 7, 7]
        x = self.cbn1(self.deconv1(x), class_label) if self.conditional and class_label is not None else nn.ReLU(True)(self.deconv1(x))
        x = torch.relu(x)
        x = self.deconv2(x)
        x = torch.tanh(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, conditional=False, num_classes=26):
        """
        Discriminator network that classifies 28×28 images as real/fake.
        """
        super().__init__()
        self.conditional = conditional
        self.num_classes = num_classes
        
        # Proven architecture for 28×28 images:
        self.features = nn.Sequential(
            # TODO: Implement convolutional layers
            # 28×28×1 → 14×14×64 → 7×7×128 → 3×3×256
            # Use Conv2d with appropriate stride
            # LeakyReLU(0.2) and Dropout2d(0.25)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),


            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 14x14 -> 7x7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 7x7 -> 3x3
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        
        # Calculate feature dimension after convolutions
        feature_dim = 256 * 3 * 3  # Adjust based on your architecture

        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + (num_classes if conditional else 0), 1),
            nn.Sigmoid()
        )

       

    
    def forward(self, img, class_label=None):
        """
        Classify images as real (1) or fake (0).
        
        Returns:
            Probability of being real [batch_size, 1]
        """
        # TODO: Extract features, flatten, concatenate class if conditional
        x = self.features(img)
        x = x.view(x.size(0), -1)  # Flatten

        if self.conditional and class_label is not None:
            x = torch.cat([x, class_label], dim=1)
        
        validity = self.classifier(x) #sigmoid
        return validity
    
   