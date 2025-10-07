"""
Main training script for hierarchical VAE experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from training_utils import kl_anneal_schedule, temperature_anneal_schedule, sample_diverse_patterns, save_pianoroll
from analyze_latent import visualize_latent_hierarchy, interpolate_styles, measure_disentanglement

def compute_hierarchical_elbo(recon_x, x, mu_low, logvar_low, mu_high, logvar_high, beta=1.0):
    """
    Compute Evidence Lower Bound (ELBO) for hierarchical VAE.
    
    ELBO = E[log p(x|z_low)] - beta * KL(q(z_low|x) || p(z_low|z_high)) 
           - beta * KL(q(z_high|z_low) || p(z_high))
    
    Args:
        recon_x: Reconstructed pattern logits [batch, 16, 9]
        x: Original patterns [batch, 16, 9]
        mu_low, logvar_low: Low-level latent parameters
        mu_high, logvar_high: High-level latent parameters
        beta: KL weight for beta-VAE
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction component
        kl_low: KL divergence for low-level latent
        kl_high: KL divergence for high-level latent
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_x, x.float(), reduction='sum'
    )
    
    # KL divergence for high-level latent: KL(q(z_high) || p(z_high))
    # where p(z_high) = N(0, I)
    kl_high = -0.5 * torch.sum(1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
    
    # KL divergence for low-level latent: KL(q(z_low) || p(z_low|z_high))
    # For simplicity, we use standard KL with N(0, I) prior
    # In practice, you might want to implement conditional prior p(z_low|z_high)
    kl_low = -0.5 * torch.sum(1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
    
    # Total loss
    total_loss = recon_loss + beta * (kl_low + kl_high)
    
    return total_loss, recon_loss, kl_low, kl_high

def train_epoch(model, data_loader, optimizer, epoch, device, config):
    """
    Train model for one epoch with annealing schedules.
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    
    # Metrics tracking
    metrics = {
        'total_loss': 0,
        'recon_loss': 0,
        'kl_low': 0,
        'kl_high': 0
    }
    
    # Get annealing parameters for this epoch
    beta = kl_anneal_schedule(epoch)
    temperature = temperature_anneal_schedule(epoch)
    
    for batch_idx, (patterns, styles, densities) in enumerate(data_loader):
        patterns = patterns.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu_low, logvar_low, mu_high, logvar_high = model(patterns, beta=beta)
        
        # Compute loss
        loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
            recon, patterns, mu_low, logvar_low, mu_high, logvar_high, beta
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics['total_loss'] += loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['kl_low'] += kl_low.item()
        metrics['kl_high'] += kl_high.item()
        
        # Log progress
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch:3d} [{batch_idx:3d}/{len(data_loader)}] '
                  f'Loss: {loss.item()/len(patterns):.4f} '
                  f'Beta: {beta:.3f} Temp: {temperature:.2f}')
    
    # Average metrics
    n_samples = len(data_loader.dataset)
    for key in metrics:
        metrics[key] /= n_samples
    
    return metrics

def main():
    """
    Main training entry point for hierarchical VAE experiments.
    """
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'z_high_dim': 4,
        'z_low_dim': 12,
        'kl_anneal_method': 'cyclical',  # 'linear', 'cyclical', or 'sigmoid'
        'data_dir': 'data/drums',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results'
    }

    cfg = dict(config)
    cfg['device'] = str(cfg['device'])   
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    gen_dir = Path(config['results_dir']) / "generated_patterns"
    lat_dir = Path(config['results_dir']) / "latent_analysis"
    for d in [gen_dir, lat_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = DrumPatternDataset(config['data_dir'], split='train')
    val_dataset = DrumPatternDataset(config['data_dir'], split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model and optimizer
    model = HierarchicalDrumVAE(
        z_high_dim=config['z_high_dim'],
        z_low_dim=config['z_low_dim']
    ).to(config['device'])


    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'config': config
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch, 
            config['device'], config
        )
        history['train'].append(train_metrics)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_metrics = {
                'total_loss': 0,
                'recon_loss': 0,
                'kl_low': 0,
                'kl_high': 0
            }
            
            with torch.no_grad():
                for patterns, styles, densities in val_loader:
                    patterns = patterns.to(config['device'])
                    recon, mu_low, logvar_low, mu_high, logvar_high = model(patterns)
                    loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
                        recon, patterns, mu_low, logvar_low, mu_high, logvar_high
                    )
                    
                    val_metrics['total_loss'] += loss.item()
                    val_metrics['recon_loss'] += recon_loss.item()
                    val_metrics['kl_low'] += kl_low.item()
                    val_metrics['kl_high'] += kl_high.item()
            
            # Average validation metrics
            n_val = len(val_dataset)
            for key in val_metrics:
                val_metrics[key] /= n_val
            
            history['val'].append(val_metrics)
            
            print(f"Epoch {epoch:3d} Validation - "
                  f"Loss: {val_metrics['total_loss']:.4f} "
                  f"KL_high: {val_metrics['kl_high']:.4f} "
                  f"KL_low: {val_metrics['kl_low']:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")


    history_to_save = dict(history)
    history_to_save['config'] = cfg



    # Save final model and history
    torch.save(model.state_dict(), f"{config['results_dir']}/best_model.pth")
    
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        json.dump(history_to_save, f, indent=2)


    # Results saving

    device= config['device']

    model.eval()
    # ---------- A. 生成样本网格（每种风格多变体） ----------
    # 你已有的函数：sample_diverse_patterns(model, n_styles, n_variations)
    grid = sample_diverse_patterns(model, n_styles=5, n_variations=10, device=device)
    # grid 形状: [n_styles][n_variations][16,9]
    for s_idx, row in enumerate(grid):
        for v_idx, patt in enumerate(row):
            np.save(gen_dir / f"style_{s_idx}_sample_{v_idx:02d}.npy", patt)
            save_pianoroll(patt, gen_dir / f"style_{s_idx}_sample_{v_idx:02d}.png")

    # ---------- B. 潜空间可视化（t-SNE） ----------
    # 你的 visualize_latent_hierarchy 目前只显示图。建议让它 return 数据（z_high_2d 等）。
    # 如果暂时不改函数体，这里直接再次计算并保存图：
    from sklearn.manifold import TSNE

    z_high_list, z_low_list, style_list = [], [], []
    with torch.no_grad():
        for patterns, styles, _ in val_loader:
            patterns = patterns.to(device)
            mu_low, logv_low, mu_high, logv_high = model.encode_hierarchy(patterns)
            z_low  = mu_low.cpu().numpy()
            z_high = mu_high.cpu().numpy()
            z_low_list.append(z_low)
            z_high_list.append(z_high)
            style_list.append(styles.numpy())
    z_low_all   = np.concatenate(z_low_list, axis=0)
    z_high_all  = np.concatenate(z_high_list, axis=0)
    styles_all  = np.concatenate(style_list, axis=0)

    z_high_2d = TSNE(n_components=2, random_state=42).fit_transform(z_high_all)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(z_high_2d[:,0], z_high_2d[:,1], c=styles_all, cmap='tab10', s=10, alpha=0.8)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Style')
    ax.set_title('t-SNE of z_high')
    fig.tight_layout()
    fig.savefig(lat_dir / "z_high_tsne.png", dpi=160)
    plt.close(fig)

    # ---------- C. 分风格看 z_low 的分布 ----------
    unique_styles = np.unique(styles_all)
    for st in unique_styles:
        idx = np.where(styles_all == st)[0]
        if len(idx) < 2:
            continue
        z2d = TSNE(n_components=2, random_state=42).fit_transform(z_low_all[idx])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(z2d[:,0], z2d[:,1], s=10, alpha=0.8)
        ax.set_title(f'z_low t-SNE (style={int(st)})')
        fig.tight_layout()
        fig.savefig(lat_dir / f"z_low_tsne_style_{int(st)}.png", dpi=160)
        plt.close(fig)

    # ---------- D. 插值序列（挑两条验证样本） ----------
    # 从验证集取两条 pattern
    patt1 = val_dataset[0][0]  # (16,9)
    patt2 = val_dataset[1][0]
    style_path = gen_dir / "interpolation_style"
    var_path   = gen_dir / "interpolation_variation"
    style_path.mkdir(exist_ok=True)
    var_path.mkdir(exist_ok=True)

    style_seq, var_seq = interpolate_styles(model, torch.tensor(patt1), torch.tensor(patt2),
                                            n_steps=10, device=device)
    # 保存每一步
    for i, arr in enumerate(style_seq):
        np.save(style_path / f"step_{i:02d}.npy", arr)
        save_pianoroll(arr, style_path / f"step_{i:02d}.png")
    for i, arr in enumerate(var_seq):
        np.save(var_path / f"step_{i:02d}.npy", arr)
        save_pianoroll(arr, var_path / f"step_{i:02d}.png")

    # ---------- E. 解耦指标 ----------
    metrics = measure_disentanglement(model, val_loader, device=device)
    with open(lat_dir / "disentanglement_metrics.json", "w") as f:
        json.dump({k: float(v) for k,v in metrics.items()}, f, indent=2)    
    
    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()