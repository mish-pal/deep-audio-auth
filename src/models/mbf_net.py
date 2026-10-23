import torch
import torch.nn as nn
from src.models.resnet_handcrafted import HandcraftedResNet
from src.models.ssl_branch import SSLAudioBranch

class CrossAttentionFusion(nn.Module):
    """
    Symmetric Cross-Attention Fusion Module (S-CAFM).
    
    Mathematical & Scientific Concept:
    Instead of simple concatenation (which assumes features do not interact) or addition
    (which assumes features lie in the same subspace), we project handcrafted (H) and
    SSL (S) embeddings into Query-Key-Value matrices and run attention in both directions.
    
    1. Handcrafted queries SSL:
       Q_h = H, K_s = S, V_s = S
       H_attn = Softmax( Q_h * K_s^T / sqrt(d_k) ) * V_s
       
    2. SSL queries Handcrafted:
       Q_s = S, K_h = H, V_h = H
       S_attn = Softmax( Q_s * K_h^T / sqrt(d_k) ) * V_h
       
    The outputs (H_attn and S_attn) are concatenated with the original embeddings
    (H and S) to construct a comprehensive 1024-dimensional fused representation:
    Fused = Concat(H, S, H_attn, S_attn)
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        # Symmetric multihead cross attention blocks
        self.cross_attn_h = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_s = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, h_embed, s_embed):
        """
        Args:
            h_embed (torch.Tensor): Handcrafted embedding of shape (batch, embed_dim)
            s_embed (torch.Tensor): SSL embedding of shape (batch, embed_dim)
            
        Returns:
            torch.Tensor: Fused features of shape (batch, 4 * embed_dim)
        """
        # Reshape to (batch, 1, embed_dim) for sequential multihead attention
        h_seq = h_embed.unsqueeze(1)
        s_seq = s_embed.unsqueeze(1)
        
        # 1. Handcrafted Queries SSL
        # query=h_seq, key=s_seq, value=s_seq
        h_attn_out, _ = self.cross_attn_h(query=h_seq, key=s_seq, value=s_seq)
        h_attn_out = h_attn_out.squeeze(1)  # Shape: (batch, embed_dim)
        
        # 2. SSL Queries Handcrafted
        # query=s_seq, key=h_seq, value=h_seq
        s_attn_out, _ = self.cross_attn_s(query=s_seq, key=h_seq, value=h_seq)
        s_attn_out = s_attn_out.squeeze(1)  # Shape: (batch, embed_dim)
        
        # Concatenate original embeddings and attention responses
        fused = torch.cat((h_embed, s_embed, h_attn_out, s_attn_out), dim=-1)
        
        return fused

class MBFNet(nn.Module):
    """
    Multi-Branch Fusion Network (MBF-Net) for Audio Deepfake Detection.
    
    Combines:
    - HandcraftedResNet (LFCC + CQCC features)
    - SSLAudioBranch (Wav2Vec 2.0 raw audio embeddings)
    - CrossAttentionFusion (symmetric attention module)
    - Classification head (standard multi-layer perceptron)
    """
    def __init__(self, config):
        super(MBFNet, self).__init__()
        self.config = config
        
        # 1. Initialize Feature branches
        self.handcrafted_branch = HandcraftedResNet(
            output_dim=256,
            pretrained=True
        )
        
        ssl_cfg = config["model"]["ssl"]
        self.ssl_branch = SSLAudioBranch(
            model_name=ssl_cfg["model_name"],
            output_dim=256,
            fine_tune=ssl_cfg["fine_tune"]
        )
        
        # 2. Cross Attention Fusion
        self.fusion = CrossAttentionFusion(
            embed_dim=256,
            num_heads=4,
            dropout=0.1
        )
        
        # 3. Bottleneck and Classification Head
        classifier_cfg = config["model"]["classifier"]
        hidden_dim = classifier_cfg["hidden_dim"]
        
        # Feature bottleneck layer (used for ArcFace clustering)
        self.bottleneck = nn.Sequential(
            nn.Linear(256 * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_cfg["dropout"])
        )
        
        # Linear classifier (Binary Output: 1 probability logit)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def extract_features(self, lfcc, cqcc, raw_waveform):
        """
        Extract the bottleneck features (useful for ArcFace loss during training
        or feature maps visualization in UI).
        """
        # Forward through branches
        h_feat = self.handcrafted_branch(lfcc, cqcc)
        s_feat = self.ssl_branch(raw_waveform)
        
        # Fuse branches
        fused = self.fusion(h_feat, s_feat)
        
        # Project to bottleneck embedding
        bottleneck_feat = self.bottleneck(fused)
        return bottleneck_feat
        
    def forward(self, lfcc, cqcc, raw_waveform):
        """
        Forward pass for the full network.
        
        Args:
            lfcc (torch.Tensor): LFCC features of shape (batch, 1, 40, time_steps)
            cqcc (torch.Tensor): CQCC features of shape (batch, 1, 40, time_steps)
            raw_waveform (torch.Tensor): Raw audio signal of shape (batch, samples)
            
        Returns:
            torch.Tensor: Classification score in [0, 1] representing deepfake probability.
        """
        # Extract features
        features = self.extract_features(lfcc, cqcc, raw_waveform)
        
        # Standard classification head prediction
        probs = self.classifier_head(features)
        
        return probs
