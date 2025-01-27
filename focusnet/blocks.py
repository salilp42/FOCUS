"""
Neural network building blocks for SaccadeNet.

This module contains the core building blocks:
- CrossNormalization: Cross-stream normalization for M/P/K pathways
- AdvancedMPKBlock2D/3D: M/P/K pathway implementations
- PredictiveCodingGRU: GRU-based predictive coding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNormalization(nn.Module):
    """Cross-normalization among M, P, K streams with enhanced pathway scaling."""
    
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha
        # Initialize pathway scales with slightly different values
        self.pathway_scale = nn.Parameter(torch.tensor([0.5, 0.45, 0.4]))
        
    def forward(self, m, p, k):
        """Forward pass with cross-stream normalization.
        
        Args:
            m: Features from M pathway
            p: Features from P pathway
            k: Features from K pathway
            
        Returns:
            tuple: Normalized (m, p, k) features
        """
        # Compute mean activations
        m_mean = m.mean(dim=1, keepdim=True)
        p_mean = p.mean(dim=1, keepdim=True)
        k_mean = k.mean(dim=1, keepdim=True)
        
        # Cross normalize with learned scales
        m = m / (1 + self.alpha*(self.pathway_scale[1]*p_mean + self.pathway_scale[2]*k_mean))
        p = p / (1 + self.alpha*(self.pathway_scale[0]*m_mean + self.pathway_scale[2]*k_mean))
        k = k / (1 + self.alpha*(self.pathway_scale[0]*m_mean + self.pathway_scale[1]*p_mean))
        
        return m, p, k


class AdvancedMPKBlock2D(nn.Module):
    """Advanced MPK Block for 2D data with enhanced channel capacity.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.cross_norm = CrossNormalization()
        self.out_channels = out_channels
        
        # M pathway - larger kernels, fewer channels for motion/coarse features
        self.M = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(7)
        )
        
        # P pathway - smaller kernels, more channels for fine detail
        self.P = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels*2, out_channels, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(7)
        )
        
        # K pathway - oriented kernels for color/orientation
        self.K = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2, groups=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(7)
        )
        
        self.gate_fc = nn.Linear(out_channels, 3)
        
    def forward(self, x, return_features=False):
        """Forward pass through M/P/K pathways.
        
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W)
            return_features (bool): Whether to return individual pathway features
            
        Returns:
            torch.Tensor or tuple: Combined features or (combined, m, p, k) if return_features=True
        """
        # Get pathway features
        m = self.M(x)
        p = self.P(x)
        k = self.K(x)
        
        # Cross normalize
        m, p, k = self.cross_norm(m, p, k)
        
        # Compute attention weights
        m_pool = m.mean([-2,-1])
        attn = self.gate_fc(m_pool)
        attn = F.softmax(attn, dim=1).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted combination
        combined = (m * attn[:,0] + p * attn[:,1] + k * attn[:,2])
        
        if return_features:
            return combined, m, p, k
        return combined


class AdvancedMPKBlock3D(nn.Module):
    """Advanced MPK Block for 3D data with enhanced channel capacity.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.cross_norm = CrossNormalization()
        self.out_channels = out_channels
        
        # M pathway - larger kernels, fewer channels
        self.M = nn.Sequential(
            nn.Conv3d(in_channels, out_channels*2, 5, padding=2),
            nn.BatchNorm3d(out_channels*2),
            nn.ReLU(True),
            nn.Conv3d(out_channels*2, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.MaxPool3d(2)
        )
        
        # P pathway - smaller kernels, more channels
        self.P = nn.Sequential(
            nn.Conv3d(in_channels, out_channels*2, 3, padding=1),
            nn.BatchNorm3d(out_channels*2),
            nn.ReLU(True),
            nn.Conv3d(out_channels*2, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.MaxPool3d(2)
        )
        
        # K pathway - oriented processing
        self.K = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, groups=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            nn.MaxPool3d(2)
        )
        
        self.gate_fc = nn.Linear(out_channels, 3)
        
    def forward(self, x, return_features=False):
        """Forward pass through M/P/K pathways.
        
        Args:
            x (torch.Tensor): Input tensor (B, C, D, H, W)
            return_features (bool): Whether to return individual pathway features
            
        Returns:
            torch.Tensor or tuple: Combined features or (combined, m, p, k) if return_features=True
        """
        # Get pathway features
        m = self.M(x)
        p = self.P(x)
        k = self.K(x)
        
        # Cross normalize
        m, p, k = self.cross_norm(m, p, k)
        
        # Compute attention weights
        m_pool = m.mean([-3,-2,-1])
        attn = self.gate_fc(m_pool)
        attn = F.softmax(attn, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted combination
        combined = (m * attn[:,0] + p * attn[:,1] + k * attn[:,2])
        
        if return_features:
            return combined, m, p, k
        return combined


class PredictiveCodingGRU(nn.Module):
    """GRU-based integrator with enhanced saccadic suppression and increased memory.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden state dimension
        suppression_scale (float): Scale factor for saccadic suppression
    """
    def __init__(self, input_dim, hidden_dim=96, suppression_scale=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.suppression_scale = suppression_scale
        self.hidden_dim = hidden_dim
        # Add layer normalization for better stability
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, saccade_feats, saccade_flags):
        """Forward pass with saccadic suppression.
        
        Args:
            saccade_feats (torch.Tensor): Features from each saccade (B, num_saccades, input_dim)
            saccade_flags (torch.Tensor): Binary flags indicating saccade (B, num_saccades)
            
        Returns:
            torch.Tensor: Integrated features
        """
        # Apply saccadic suppression
        suppression = 1.0 - (self.suppression_scale * saccade_flags)
        suppression = suppression.unsqueeze(-1)
        saccade_feats = saccade_feats * suppression
        
        # GRU integration
        out, h = self.gru(saccade_feats)
        
        # Use final hidden state with normalization
        return self.norm(h[-1])
