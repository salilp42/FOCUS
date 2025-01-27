"""
Core model implementations for FocusNet.

This module contains the main model architectures for both 2D and 3D medical image analysis:
- EnhancedFocusNet2D: For 2D medical images (e.g., X-rays, pathology slides)
- EnhancedFocusNet3D: For 3D medical volumes (e.g., CT, MRI)

Both models implement bio-inspired attention mechanisms and M/P/K pathways.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    AdvancedMPKBlock2D,
    AdvancedMPKBlock3D,
    PredictiveCodingGRU
)

class EnhancedFocusNet2D(nn.Module):
    """2D FocusNet with advanced M/P/K, predictive coding, multi-saccade integration.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB)
        num_classes (int): Number of output classes/labels
        is_multilabel (bool): Whether this is a multi-label classification task
        num_saccades (int): Number of saccadic movements
        mpk_out (int): Number of output channels for MPK block
    """
    def __init__(self, in_channels=1, num_classes=10, is_multilabel=False,
                 num_saccades=2, mpk_out=24):
        super().__init__()
        self.is_multilabel = is_multilabel
        self.num_saccades = num_saccades
        self.mpk_block = AdvancedMPKBlock2D(in_channels, mpk_out)
        self.saccade_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mpk_out*7*7, 32),
            nn.ReLU(True),
            nn.Linear(32,2)
        )
        self.pc_gru = PredictiveCodingGRU(input_dim=mpk_out, hidden_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,num_classes)
        )
        
        self.activation_maps = []
        self.hooks = []
        
        def hook_fn(module, inp, out):
            self.activation_maps.append(out.detach())
            
        self.hooks.append(self.mpk_block.M.register_forward_hook(hook_fn))
        self.hooks.append(self.mpk_block.P.register_forward_hook(hook_fn))
        self.hooks.append(self.mpk_block.K.register_forward_hook(hook_fn))

    def forward(self, x, return_offsets=False):
        """Forward pass with optional return of attention offsets.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            return_offsets (bool): Whether to return saccade offsets
            
        Returns:
            tuple: (predictions, offsets) if return_offsets=True else predictions
        """
        B = x.size(0)
        offsets = []
        saccade_feats = []
        saccade_flags = torch.zeros(B, self.num_saccades, device=x.device)
        
        # Initial features
        feats = self.mpk_block(x)
        saccade_feats.append(feats)
        
        # Multiple saccades
        for i in range(self.num_saccades):
            offset = self.saccade_fc(feats)
            offsets.append(offset)
            
            # Center crop with offset
            cropped = center_crop_2d(x, crop_size=(14,14), 
                                   offset=offset.detach().cpu().numpy())
            
            # Get features for cropped region
            feats = self.mpk_block(cropped)
            saccade_feats.append(feats)
            saccade_flags[:,i] = 1
            
        # Stack features and integrate
        saccade_feats = torch.stack(saccade_feats, dim=1)  # (B, num_saccades+1, feat_dim)
        integrated = self.pc_gru(saccade_feats, saccade_flags)
        
        # Final classification
        logits = self.classifier(integrated)
        if self.is_multilabel:
            out = torch.sigmoid(logits)
        else:
            out = F.log_softmax(logits, dim=1)
            
        if return_offsets:
            return out, torch.stack(offsets, dim=1)
        return out

    def get_attention_maps(self):
        """Get activation maps from M/P/K pathways for visualization."""
        maps = self.activation_maps.copy()
        self.activation_maps.clear()
        return maps
        
    def __del__(self):
        """Clean up hooks on deletion."""
        for hook in self.hooks:
            hook.remove()


class EnhancedFocusNet3D(nn.Module):
    """3D FocusNet with advanced M/P/K, predictive coding, multi-saccade integration.
    
    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes/labels
        is_multilabel (bool): Whether this is a multi-label classification task
        num_saccades (int): Number of saccadic movements
        mpk_out (int): Number of output channels for MPK block
    """
    def __init__(self, in_channels=1, num_classes=2, is_multilabel=False,
                 num_saccades=2, mpk_out=16):
        super().__init__()
        self.is_multilabel = is_multilabel
        self.num_saccades = num_saccades
        
        self.mpk_block = AdvancedMPKBlock3D(in_channels, mpk_out)
        self.saccade_fc = nn.Sequential(
            nn.Linear(mpk_out, 32),
            nn.ReLU(True),
            nn.Linear(32,3)
        )
        self.pc_gru = PredictiveCodingGRU(input_dim=mpk_out, hidden_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,num_classes)
        )
        
        self.activation_maps = []
        self.hooks = []
        
        def hook_fn(module, inp, out):
            self.activation_maps.append(out.detach())
            
        self.hooks.append(self.mpk_block.M.register_forward_hook(hook_fn))
        self.hooks.append(self.mpk_block.P.register_forward_hook(hook_fn))
        self.hooks.append(self.mpk_block.K.register_forward_hook(hook_fn))

    def forward(self, x, return_offsets=False):
        """Forward pass with optional return of attention offsets.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)
            return_offsets (bool): Whether to return saccade offsets
            
        Returns:
            tuple: (predictions, offsets) if return_offsets=True else predictions
        """
        B = x.size(0)
        offsets = []
        saccade_feats = []
        saccade_flags = torch.zeros(B, self.num_saccades, device=x.device)
        
        # Initial features
        feats = self.mpk_block(x)
        saccade_feats.append(feats)
        
        # Multiple saccades
        for i in range(self.num_saccades):
            offset = self.saccade_fc(feats.mean([-3,-2,-1]))
            offsets.append(offset)
            
            # Center crop with offset
            offset_bounded = bound_offset_3d(offset)
            cropped = center_crop_3d(x, offset_bounded)
            
            # Get features for cropped region
            feats = self.mpk_block(cropped)
            saccade_feats.append(feats)
            saccade_flags[:,i] = 1
            
        # Stack features and integrate
        saccade_feats = torch.stack(saccade_feats, dim=1)
        integrated = self.pc_gru(saccade_feats, saccade_flags)
        
        # Final classification
        logits = self.classifier(integrated)
        if self.is_multilabel:
            out = torch.sigmoid(logits)
        else:
            out = F.log_softmax(logits, dim=1)
            
        if return_offsets:
            return out, torch.stack(offsets, dim=1)
        return out

    def get_attention_maps(self):
        """Get activation maps from M/P/K pathways for visualization."""
        maps = self.activation_maps.copy()
        self.activation_maps.clear()
        return maps
        
    def __del__(self):
        """Clean up hooks on deletion."""
        for hook in self.hooks:
            hook.remove()
