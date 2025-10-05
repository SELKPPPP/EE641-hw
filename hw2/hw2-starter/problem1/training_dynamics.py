"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from metrics import  _simple_letter_classifier
import matplotlib.pyplot as plt

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
            d_optimizer.zero_grad()
            outputs = discriminator(real_images, labels)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, 100).to(device)  # Assuming latent dim is 100
            fake_images = generator(z, labels)

            outputs = discriminator(fake_images.detach(), labels)
            d_loss_fake = criterion(outputs, fake_labels)
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
            fake_images = generator(z, labels)

            outputs = discriminator(fake_images, labels)
            g_loss = criterion(outputs, real_labels)  # We want generator to fool discriminator
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
    generator.eval()
    z = torch.randn(n_samples, 100).to(device)  # Assuming latent dim is 100
    labels = torch.randint(0, 26, (n_samples,)).to(device)  # Random class labels
    with torch.no_grad():
        fake_images = generator(z, labels)

    fake_images_01 = (fake_images + 1) / 2  # Scale to [0, 1]
    fake_images_01 = fake_images_01.clamp(0, 1)

    predicted_letters = [_simple_letter_classifier(fake_images_01[i].cpu())for i in range(n_samples)]
    unique_letters = set(predicted_letters)
    coverage = len(unique_letters) / 26.0
    generator.train()
    return coverage
           
       

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
    plt.plot(history['epoch'], history['mode_coverage'], marker='o')
    plt.title('Mode Coverage Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Mode Coverage (Unique Letters / 26)')
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    
    