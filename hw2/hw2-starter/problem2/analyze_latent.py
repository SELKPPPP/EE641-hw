"""
Latent space analysis tools for hierarchical VAE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """


def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda'):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """
    model.eval()
    pattern1 = pattern1.to(device).unsqueeze(0)  # [1,16,9]
    pattern2 = pattern2.to(device).unsqueeze(0)  # [1,16,9]
    
    with torch.no_grad():
        mu_low1, logvar_low1, mu_high1, logvar_high1 = model.encode_hierarchy(pattern1)
        z_low1 = model.reparameterize(mu_low1, logvar_low1)
        z_high1 = model.reparameterize(mu_high1, logvar_high1)
        
        mu_low2, logvar_low2, mu_high2, logvar_high2 = model.encode_hierarchy(pattern2)
        z_low2 = model.reparameterize(mu_low2, logvar_low2)
        z_high2 = model.reparameterize(mu_high2, logvar_high2)
        
        # Interpolate z_high
        z_high_interp = [
            (1 - alpha) * z_high1 + alpha * z_high2 
            for alpha in np.linspace(0, 1, n_steps)
        ]
        
        # Interpolate z_low
        z_low_interp = [
            (1 - alpha) * z_low1 + alpha * z_low2 
            for alpha in np.linspace(0, 1, n_steps)
        ]
        
        # Decode interpolations
        patterns_style_interp = [
            torch.sigmoid(model.decode_hierarchy(z_high, z_low1)).cpu().squeeze(0).numpy()
            for z_high in z_high_interp
        ]
        
        patterns_variation_interp = [
            torch.sigmoid(model.decode_hierarchy(z_high1, z_low)).cpu().squeeze(0).numpy()
            for z_low in z_low_interp
        ]
    
    # Visualization code can be added here to plot the interpolated patterns
    return patterns_style_interp, patterns_variation_interp
    

def measure_disentanglement(model, data_loader, device='cuda'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    model.eval()
    z_high_list = []
    z_low_list = []
    style_list = []
    
    with torch.no_grad():
        for patterns, styles, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, logvar_low, mu_high, logvar_high = model.encode_hierarchy(patterns)
            z_low = model.reparameterize(mu_low, logvar_low)
            z_high = model.reparameterize(mu_high, logvar_high)
            
            z_low_list.append(z_low.cpu().numpy())
            z_high_list.append(z_high.cpu().numpy())
            style_list.append(styles.numpy())
    
    z_low_all = np.concatenate(z_low_list, axis=0)
    z_high_all = np.concatenate(z_high_list, axis=0)
    styles_all = np.concatenate(style_list, axis=0)
    
    disentanglement_metrics = {}
    
    # Compute variance of z_high within and across genres
    overall_var_high = np.var(z_high_all, axis=0).mean()
    within_var_high = []
    
    for style in np.unique(styles_all):
        idxs = np.where(styles_all == style)[0]
        var_style = np.var(z_high_all[idxs], axis=0).mean()
        within_var_high.append(var_style)
    
    avg_within_var_high = np.mean(within_var_high)
    disentanglement_metrics['z_high_disentanglement'] = 1 - (avg_within_var_high / overall_var_high)
    
    # Compute variance of z_low within same genre
    overall_var_low = np.var(z_low_all, axis=0).mean()
    within_var_low = []
    
    for style in np.unique(styles_all):
        idxs = np.where(styles_all == style)[0]
        var_style = np.var(z_low_all[idxs], axis=0).mean()
        within_var_low.append(var_style)
    
    avg_within_var_low = np.mean(within_var_low)
    disentanglement_metrics['z_low_variation'] = avg_within_var_low / overall_var_low
    
    return disentanglement_metrics

def controllable_generation(model, genre_labels, device='cuda'):
    """
    Test controllable generation using the hierarchy.
    
    TODO:
    1. Learn genre embeddings in z_high space
    2. Generate patterns with specified genre
    3. Control complexity via z_low sampling temperature
    4. Evaluate genre classification accuracy
    """
    pass