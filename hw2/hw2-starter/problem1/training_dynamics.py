"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from metrics import  _simple_letter_classifier, mode_coverage_score
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid, save_image

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda'):
    """
    Standard GAN training implementation.
    
    Uses vanilla GAN objective which typically exhibits mode collapse.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation
        
    Returns:
        dict: Training history and metrics
    """
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            # TODO: Implement discriminator training step
            # 1. Zero gradients
            # 2. Forward pass on real images
            # 3. Compute real loss
            # 4. Generate fake images from random z
            # 5. Forward pass on fake images (detached)
            # 6. Compute fake loss
            # 7. Backward and optimize
            is_cond = getattr(generator, "conditional", False)
            d_optimizer.zero_grad()
            if is_cond:
                y_onehot = nn.functional.one_hot(labels, num_classes=26).float().to(device)
                out_real = discriminator(real_images, y_onehot)
            else:
                out_real = discriminator(real_images)
            
            d_loss_real = criterion(out_real, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, 100).to(device)  # Assuming latent dim is 100
            if is_cond:
                y_onehot = nn.functional.one_hot(labels, num_classes=26).float().to(device)
                fake_images = generator(z, y_onehot)
                out_fake = discriminator(fake_images.detach(), y_onehot)
            else:
                fake_images = generator(z)
                out_fake = discriminator(fake_images.detach())

            d_loss_fake = criterion(out_fake, fake_labels)
            d_loss_fake.backward()
            d_optimizer.step()
            d_loss = d_loss_real + d_loss_fake  

            
            # ========== Train Generator ==========
            # TODO: Implement generator training step
            # 1. Zero gradients
            # 2. Generate fake images
            # 3. Forward pass through discriminator
            # 4. Compute adversarial loss
            # 5. Backward and optimize


            g_optimizer.zero_grad()
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z, y_onehot) if is_cond else generator(z)

            out_fake = discriminator(fake_images, y_onehot) if is_cond else discriminator(fake_images)
            g_loss = criterion(out_fake, real_labels)  # We want generator to fool discriminator
            g_loss.backward()
            g_optimizer.step()
            
            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        # Analyze mode collapse every 10 epochs
        if epoch % 10 == 0:
            mode_coverage = analyze_mode_coverage(generator, device)
            history['model_epochs'].append(epoch)
            history['mode_coverage'].append(mode_coverage)
            print(f"Epoch {epoch}: Mode coverage = {mode_coverage:.2f}")
    
    return history

def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.
    
    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate
        
    Returns:
        float: Coverage score (unique letters / 26)
    """
    # TODO: Generate n_samples images
    # Use provided letter classifier to identify generated letters
    # Count unique letters produced
    # Return coverage score (0 to 1)
    is_cond = getattr(generator, "conditional", False)
    generator.eval()
    z = torch.randn(n_samples, 100).to(device)  # Assuming latent dim is 100
    labels = torch.randint(0, 26, (n_samples,)).to(device)  # Random class labels
    with torch.no_grad():
        if is_cond:
            y_onehot = nn.functional.one_hot(labels, num_classes=26).float().to(device)
            fake_images = generator(z, y_onehot)
        else:
            fake_images = generator(z)

    fake_images_01 = (fake_images + 1) / 2  # Scale to [0, 1]
    fake_images_01 = fake_images_01.clamp(0, 1)

    stats = mode_coverage_score(fake_images_01, classifier_fn=_simple_letter_classifier)

    return stats['coverage_score']
           
       

def visualize_mode_collapse(history, save_path):
    """
    Visualize mode collapse progression over training.
    
    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """
    # TODO: Plot mode coverage over time
    # Show which letters survive and which disappear
    plt.figure(figsize=(10, 5))
    plt.plot(history['model_epochs'], history['mode_coverage'], marker='o')
    plt.title('Mode Coverage Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Mode Coverage (Unique Letters / 26)')
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    


def save_grid(generator, device, save_path="results/samples/grid.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    generator.eval()

    z_dim  = getattr(generator, "z_dim", 100)
    is_cond = getattr(generator, "conditional", False)

    if is_cond:
        labels = torch.arange(26, device=device)    # [26], 0..25
        y_onehot = nn.functional.one_hot(labels, num_classes=26).float().to(device)  # [26, 26]
        z = torch.randn(26, z_dim, device=device)
        fake = generator(z, y_onehot)
    else:
        z = torch.randn(64, z_dim, device=device)    
        fake = generator(z)

    fake = (fake + 1) / 2 if fake.min() < 0 else fake
    fake = fake.clamp(0, 1)

    nrow = 13 if is_cond else 8
    grid = make_grid(fake, nrow=nrow, padding=2)
    save_image(grid, save_path)
    print(f"Saved grid to {save_path}")