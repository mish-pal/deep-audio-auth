import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class SSLAudioBranch(nn.Module):
    """
    Self-Supervised Learning (SSL) branch using a pre-trained Wav2Vec 2.0 encoder.
    
    Architectural & Scientific Concepts:
    Instead of only using the output of the last transformer layer (which is optimized
    for speech recognition/semantics), we extract hidden states from ALL 12 layers
    and apply a Learnable Layer-Wise Attention Pooling mechanism.
    
    Formula:
    w = Softmax(alpha)  where alpha are 12 learnable scalar parameters
    H_pooled = sum_{i=1}^12 w_i * H_i
    
    This is extremely powerful because lower layers capture low-level acoustic details
    (e.g., vocoder artifacts, sampling rate glitches) while middle/upper layers capture
    prosodic dynamics and articulation flow (which are unnatural in deepfakes).
    The model automatically learns how to weight these layers.
    """
    def __init__(self, model_name="facebook/wav2vec2-base", output_dim=256, fine_tune=False):
        super(SSLAudioBranch, self).__init__()
        
        # Load Wav2Vec 2.0 base model
        self.ssl_model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Configure fine-tuning parameters
        self.fine_tune = fine_tune
        if not fine_tune:
            # Freeze all parameters to save memory and accelerate training
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        else:
            # Fine-tune only the last 2 transformer layers for target domain adaptation
            for param in self.ssl_model.parameters():
                param.requires_grad = False
            for layer in self.ssl_model.encoder.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        # Learnable layer-wise pooling weights (12 scalar parameters for Wav2Vec2 base)
        num_layers = self.ssl_model.config.num_hidden_layers + 1  # 12 layers + input embedding
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        
        # Projection layer to map Wav2Vec2 embedding (768 dimensions) to target dimension
        ssl_hidden_dim = self.ssl_model.config.hidden_size  # 768
        self.projection = nn.Sequential(
            nn.Linear(ssl_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Raw audio signal of shape (batch_size, samples)
            
        Returns:
            torch.Tensor: SSL representation embedding of shape (batch_size, output_dim)
        """
        # Ensure waveform is 2D: (batch_size, samples)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
            
        if not self.fine_tune:
            # Run inference in eval mode without calculating gradients for the frozen backbone
            self.ssl_model.eval()
            with torch.no_grad():
                outputs = self.ssl_model(waveform, output_hidden_states=True)
        else:
            outputs = self.ssl_model(waveform, output_hidden_states=True)
            
        # Extract hidden states from all transformer layers (tuple of 13 tensors)
        # Each tensor has shape: (batch_size, seq_len, 768)
        hidden_states = outputs.hidden_states
        
        # Stack hidden states: (num_layers, batch_size, seq_len, 768)
        stacked_hidden_states = torch.stack(hidden_states, dim=0)
        
        # Calculate softmax weights across layers
        # Softmax ensures weights sum to 1.0: w_i = e^(alpha_i) / sum_j e^(alpha_j)
        softmax_weights = torch.softmax(self.layer_weights, dim=0)
        
        # Perform weighted layer pooling
        # Multiply weights: (num_layers, 1, 1, 1) * (num_layers, batch_size, seq_len, 768)
        # Sum along the layers dimension
        pooled_states = torch.sum(
            softmax_weights.view(-1, 1, 1, 1) * stacked_hidden_states,
            dim=0
        )  # Shape: (batch_size, seq_len, 768)
        
        # Perform temporal average pooling over the sequence length
        # (batch_size, seq_len, 768) -> (batch_size, 768)
        temporal_pooled = torch.mean(pooled_states, dim=1)
        
        # Project to target dimension
        embedding = self.projection(temporal_pooled)  # Shape: (batch_size, output_dim)
        
        return embedding
