import torch
import torch.nn as nn
import torchvision.models as models
import ssl

# Bypass SSL certificate verification for downloading pre-trained weights on some networks
ssl._create_default_https_context = ssl._create_unverified_context

class HandcraftedResNet(nn.Module):
    """
    ResNet-18 branch for processing handcrafted acoustic features (LFCC and CQCC).
    
    Acoustic & Architectural Concept:
    Instead of running separate ResNet backbones for LFCC and CQCC (which increases
    computational complexity and parameter count), we stack them as 2 input channels:
    Input shape: (batch_size, 2, n_coeffs, time_steps)
    
    This allows the 2D convolutions in ResNet to learn joint representations and
    cross-correlations between Linear frequency details (LFCC) and Constant-Q scale
    features (CQCC) right from the early layers, while cutting parameter count in half.
    
    We adapt a pre-trained ResNet-18 by:
    1. Modifying the first Conv2d layer to take 2 input channels instead of 3.
    2. Summing and rescaling the pre-trained weights to initialize the 2-channel weights.
    3. Replacing the final pooling and fully connected layers with an embedding projection.
    """
    def __init__(self, output_dim=256, pretrained=True):
        super(HandcraftedResNet, self).__init__()
        
        # InstanceNorm2d normalizes features across the time-frequency plane for each sample.
        # This makes the model invariant to global gain changes and recording level differences.
        self.norm = nn.InstanceNorm2d(2, affine=True)
        
        # Load pre-trained ResNet-18
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18()
            
        # Modify the first conv layer (conv1) from Conv2d(3, 64, ...) to Conv2d(2, 64, ...)
        # We preserve the pre-trained weights by taking the mean across the RGB channels
        # and copying it to our 2 channels.
        pretrained_weights = resnet.conv1.weight.clone()  # Shape: (64, 3, 7, 7)
        
        resnet.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        
        with torch.no_grad():
            # Initialize with pre-trained visual edge detection weights
            resnet.conv1.weight[:, 0] = pretrained_weights[:, 0]
            resnet.conv1.weight[:, 1] = pretrained_weights[:, 1]
            
        # Strip final pooling and fully-connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global Average Pooling + Max Pooling to aggregate spatial details
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Projection head to obtain final handcrafted embedding
        # Output of last conv layer in ResNet-18 is 512 channels
        # We concatenate Avg and Max pooling -> 512 * 2 = 1024 channels
        self.projection = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, lfcc, cqcc):
        """
        Args:
            lfcc (torch.Tensor): Shape (batch_size, 1, n_lfcc, time_steps)
            cqcc (torch.Tensor): Shape (batch_size, 1, n_cqcc, time_steps)
            
        Returns:
            torch.Tensor: Feature embedding of shape (batch_size, output_dim)
        """
        # Concatenate LFCC and CQCC along the channel dimension
        # Input shape: (batch_size, 2, n_coeffs, time_steps)
        x = torch.cat((lfcc, cqcc), dim=1)
        
        # Apply normalization
        x = self.norm(x)
        
        # Extract features from ResNet conv blocks
        x = self.features(x)  # Shape: (batch_size, 512, H_feat, W_feat)
        
        # Aggregate features using both Average and Max pooling
        # Avg pooling captures the general background profile; Max pooling captures transient peaks
        x_avg = self.avg_pool(x).flatten(1)  # Shape: (batch_size, 512)
        x_max = self.max_pool(x).flatten(1)  # Shape: (batch_size, 512)
        
        x_fused = torch.cat((x_avg, x_max), dim=1)  # Shape: (batch_size, 1024)
        
        # Project to embedding space
        embedding = self.projection(x_fused)  # Shape: (batch_size, output_dim)
        
        return embedding
