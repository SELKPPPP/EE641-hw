"""
GAN stabilization techniques to combat mode collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import defaultdict

def train_gan_with_fix(generator, discriminator, data_loader, 
                       num_epochs=100, fix_type='feature_matching'):
    """
    Train GAN with mode collapse mitigation techniques.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        fix_type: Stabilization method ('feature_matching', 'unrolled', 'minibatch')
        
    Returns:
        dict: Training history with metrics
    """
    
    if fix_type == 'feature_matching':
        # Feature matching: Match statistics of intermediate layers
        # instead of just final discriminator output
        
        def feature_matching_loss(real_images, fake_images, discriminator):
            """
            TODO: Implement feature matching loss
            
            Extract intermediate features from discriminator
            Match mean statistics: ||E[f(x)] - E[f(G(z))]||²
            Use discriminator.features (before final classifier)
            """
            # Extract features
            real_features = discriminator.features(real_images)
            fake_features = discriminator.features(fake_images)

            # Flatten features
            real_features = real_features.view(real_features.size(0), -1)
            fake_features = fake_features.view(fake_features.size(0), -1)

            # Compute mean feature vectors and MSE loss
            real_mean = real_features.mean(0)
            fake_mean = fake_features.mean(0)
            loss = F.mse_loss(real_mean, fake_mean)
            return loss
        

            
    elif fix_type == 'unrolled':
        # Unrolled GANs: Look ahead k discriminator updates
        
        def unrolled_discriminator(discriminator, real_data, fake_data, k=5):
            """
            TODO: Implement k-step unrolled discriminator
            
            Create temporary discriminator copy
            Update it k times
            Compute generator loss through updated discriminator
            """
            temp_discriminator = copy.deepcopy(discriminator)
            optimizer = torch.optim.Adam(temp_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            criterion = nn.BCELoss()
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1).to(real_data.device)
            fake_labels = torch.zeros(batch_size, 1).to(real_data.device)
            
            for _ in range(k):
                optimizer.zero_grad()
                outputs_real = temp_discriminator(real_data)
                loss_real = criterion(outputs_real, real_labels)
                
                outputs_fake = temp_discriminator(fake_data.detach())
                loss_fake = criterion(outputs_fake, fake_labels)
                
                d_loss = loss_real + loss_fake
                d_loss.backward()
                optimizer.step()
            
            return temp_discriminator
            
    elif fix_type == 'minibatch':
        # Minibatch discrimination: Let discriminator see batch statistics
        from types import MethodType
        class MinibatchDiscrimination(nn.Module):
            """
            TODO: Add minibatch discrimination layer to discriminator
            
            Compute L2 distance between samples in batch
            Concatenate statistics to discriminator features
            """
            def __init__(self, in_features, out_features, kernel_dims):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.kernel_dims = kernel_dims
                self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))
            
            def forward(self, x):
                batch_size = x.size(0)
                M = x.mm(self.T.view(self.in_features, -1))
                M = M.view(batch_size, self.out_features, self.kernel_dims)
                
                # Compute L2 distance between samples
                out = []
                for i in range(batch_size):
                    diff = M[i].unsqueeze(0) - M
                    sqr_diff = torch.square(diff).sum(2)
                    exp_neg_sqr = torch.exp(-sqr_diff)
                    out.append(exp_neg_sqr.sum(0))
                
                out = torch.stack(out)
                return torch.cat([x, out], dim=1)
            
        discriminator.mb = MinibatchDiscrimination(in_features=256 * 3 * 3, out_features=100, kernel_dims=5)
        feature_dim = 256 * 3 * 3 + 100  # Adjust based on your architecture and minibatch layer
        discriminator.classifier = nn.Sequential(
            nn.Linear(feature_dim + (26 if discriminator.conditional else 0), 1),
            nn.Sigmoid()
        )

        def forward_mb(self, img, class_label=None):
            x = self.features(img)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.mb(x)  # Apply minibatch discrimination

            if self.conditional and class_label is not None:
                x = torch.cat([x, class_label], dim=1)

            validity = self.classifier(x) #sigmoid
            return validity
        
        discriminator.forward = MethodType(forward_mb, discriminator)

    
    
    # Training loop with chosen fix
    # TODO: Implement modified training using selected technique
    

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = defaultdict(list)
    device = next(generator.parameters()).device

    for epoch in range(num_epochs):


        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            outputs_real = discriminator(real_images, labels)
            d_loss_real = criterion(outputs_real, real_labels)
            
            z = torch.randn(batch_size, 100).to(device)  # Assuming latent dim is 100
            fake_images = generator(z, labels)
            outputs_fake = discriminator(fake_images.detach(), labels)
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z, labels)
            outputs = discriminator(fake_images, labels)
            
            if fix_type == 'feature_matching':
                g_loss = feature_matching_loss(real_images, fake_images, discriminator)
            elif fix_type == 'unrolled':
                temp_discriminator = unrolled_discriminator(discriminator, real_images, fake_images, k=5)
                outputs_unrolled = temp_discriminator(fake_images, labels)
                g_loss = criterion(outputs_unrolled, real_labels)
            elif fix_type == 'minibatch':
                g_loss = criterion(outputs, real_labels)  # Standard loss; minibatch layer is in discriminator
            
            g_loss.backward()
            g_optimizer.step()
            
             # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        print(f"Epoch [{epoch+1}/{num_epochs}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    return history    
