"""
Utility functions for FocusNet.

This module contains:
- FocalLoss: Loss function for handling class imbalance
- Visualization utilities for attention maps and feature analysis
- Data processing utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance, focusing more on hard examples.
    
    Args:
        alpha (float): Weighting factor for rare classes
        gamma (float): Focusing parameter
        reduction (str): Reduction method ('mean' or 'sum')
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """Forward pass computing focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Computed loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def center_crop_2d(img, crop_size=(14,14), offset=(0,0)):
    """Center crop a 2D image with optional offset.
    
    Args:
        img (torch.Tensor): Input image (C,H,W)
        crop_size (tuple): Size of crop (ch,cw)
        offset (tuple): Offset from center (dy,dx)
        
    Returns:
        torch.Tensor: Cropped image
    """
    C,H,W = img.shape
    ch,cw = crop_size
    dy,dx = offset
    
    if ch>H or cw>W:
        return img
        
    top = (H - ch)//2 + dy
    left = (W - cw)//2 + dx
    top = max(0, min(top, H-ch))
    left = max(0, min(left, W-cw))
    
    return img[:, top:top+ch, left:left+cw]


def bound_offset_3d(offset, volume_size=28, max_shift_ratio=0.25):
    """Bound the 3D offset to prevent excessive shifts.
    
    Args:
        offset (torch.Tensor): Original predicted offset (x,y,z)
        volume_size (int): Size of the volume
        max_shift_ratio (float): Maximum allowed shift ratio
        
    Returns:
        torch.Tensor: Bounded offset
    """
    max_shift = int(volume_size * max_shift_ratio)
    return torch.clamp(offset, -max_shift, max_shift)


def center_crop_3d(volume, offset, size=28):
    """Enhanced center crop with bounded offsets for 3D volumes.
    
    Args:
        volume (torch.Tensor): Input volume (B,C,D,H,W)
        offset (torch.Tensor): Offset tensor (B,3)
        size (int): Size of output crop
        
    Returns:
        torch.Tensor: Cropped volume
    """
    B,C,D,H,W = volume.shape
    device = volume.device
    
    # Create sampling grid
    linspace = torch.linspace(-1, 1, size, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(linspace, linspace, linspace)
    
    # Add batch dimension and normalize offset
    grid_d = grid_d.unsqueeze(0).expand(B, -1, -1, -1)
    grid_h = grid_h.unsqueeze(0).expand(B, -1, -1, -1)
    grid_w = grid_w.unsqueeze(0).expand(B, -1, -1, -1)
    
    # Apply offset
    offset_d = 2.0 * offset[:,0].view(B,1,1,1) / D
    offset_h = 2.0 * offset[:,1].view(B,1,1,1) / H
    offset_w = 2.0 * offset[:,2].view(B,1,1,1) / W
    
    grid_d = grid_d + offset_d
    grid_h = grid_h + offset_h
    grid_w = grid_w + offset_w
    
    # Stack and sample
    grid = torch.stack([grid_w, grid_h, grid_d], dim=4)
    return F.grid_sample(volume, grid, align_corners=True)


def plot_attention_maps(model, dataset, save_dir, n_samples=4):
    """Plot attention maps for 2D or 3D data.
    
    Args:
        model: SaccadeNet model
        dataset: Dataset to visualize
        save_dir (str): Directory to save plots
        n_samples (int): Number of samples to visualize
    """
    model.eval()
    is_3d = isinstance(model, EnhancedFocusNet3D)
    
    for i in range(n_samples):
        img, label = dataset[i]
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred, offsets = model(img, return_offsets=True)
            maps = model.get_attention_maps()
        
        # Create figure
        if is_3d:
            fig = plt.figure(figsize=(15,5))
            gs = fig.add_gridspec(1,3)
            
            # Plot orthogonal slices
            for j, (axis, title) in enumerate([
                ('sagittal', 'Sagittal'),
                ('coronal', 'Coronal'),
                ('axial', 'Axial')
            ]):
                ax = fig.add_subplot(gs[0,j])
                slice_idx = img.shape[2+j]//2
                if j == 0:
                    slice_img = img[0,:,slice_idx,:,:]
                elif j == 1:
                    slice_img = img[0,:,:,slice_idx,:]
                else:
                    slice_img = img[0,:,:,:,slice_idx]
                    
                ax.imshow(slice_img.mean(0).cpu())
                ax.set_title(f'{title} View')
                
                # Overlay attention
                for k, offset in enumerate(offsets[0]):
                    ax.plot(offset[2].item(), offset[1].item(), 
                           'r*' if k==0 else 'y*', 
                           label=f'Saccade {k+1}')
                ax.legend()
                
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10,5))
            
            # Original image
            axes[0].imshow(img[0].mean(0).cpu())
            axes[0].set_title('Input Image')
            
            # Attention overlay
            axes[1].imshow(img[0].mean(0).cpu())
            for k, offset in enumerate(offsets[0]):
                axes[1].plot(offset[1].item(), offset[0].item(),
                           'r*' if k==0 else 'y*',
                           label=f'Saccade {k+1}')
            axes[1].set_title('Attention Map')
            axes[1].legend()
            
        plt.tight_layout()
        plt.savefig(f'{save_dir}/attention_sample_{i}.png')
        plt.close()


def measure_receptive_field(model, device='cpu', is_3d=False):
    """Measure effective receptive field using gradient method.
    
    Args:
        model: SaccadeNet model
        device (str): Device to run computation on
        is_3d (bool): Whether model is 3D
        
    Returns:
        dict: Receptive field data and frequency response
    """
    model.eval()
    
    if is_3d:
        input_shape = (1,1,32,32,32)
        center = (16,16,16)
    else:
        input_shape = (1,1,32,32)
        center = (16,16)
        
    x = torch.zeros(input_shape, requires_grad=True, device=device)
    x.retain_grad()
    
    # Forward pass
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
        
    # Backward from center
    grad = torch.zeros_like(out)
    grad[0, 0] = 1
    out.backward(gradient=grad)
    
    # Get gradient magnitude
    rf_data = x.grad.abs().squeeze().cpu().numpy()
    
    # Compute frequency response
    freq_resp = np.fft.fftshift(np.fft.fftn(rf_data))
    freq_resp = np.abs(freq_resp)
    
    return {
        'rf_data': rf_data,
        'freq_resp': freq_resp,
        'center': center
    }
