import torch
import torch.nn as nn
import math

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace Loss) module.
    
    Mathematical & Scientific Concept:
    Standard Cross-Entropy Loss optimization only separates classes linearly. It does not
    force intra-class compactness or inter-class margin maximization.
    
    ArcFace maps features onto a hypersphere and optimizes the geodesic distance on it.
    For embedding x_i (normalized) and target class weight W_j (normalized):
    cos(theta_j) = W_j^T x_i
    For the target class y_i, an additive angular margin 'm' is added to theta:
    cos(theta_y_i + m)
    
    The modified logits are scaled by parameter 's' and passed to Softmax:
    Logits_j = s * cos(theta_j)              if j != y_i
    Logits_y_i = s * cos(theta_y_i + m)      if j == y_i
    
    This forces a strict angular margin constraint in the feature embedding space,
    resulting in highly clustered 'Authentic' speech embeddings and pushing 'Fake' speech
    into a completely separate region of the hypersphere. This margin-based clustering
    is what enables the model to generalize well to unseen types of deepfakes (domain shift).
    """
    def __init__(self, in_features, num_classes=2, s=30.0, m=0.50, easy_margin=False):
        """
        Args:
            in_features (int): Dimension of the input bottleneck embedding (e.g. 256).
            num_classes (int): Number of target classes (2 for Authentic vs. Deepfake).
            s (float): Scale factor for the hypersphere radius.
            m (float): Additive angular margin penalty.
            easy_margin (bool): If True, avoids negative cos(theta + m) scaling.
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        
        # Learnable class representative weights on the hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features, label):
        """
        Args:
            input_features (torch.Tensor): Feature embeddings of shape (batch, in_features).
            label (torch.Tensor): Class labels of shape (batch,).
            
        Returns:
            torch.Tensor: Modified logits for Cross Entropy computation.
        """
        # 1. Normalize input features and class weights to project onto unit hypersphere
        # x_norm = x / ||x||
        # W_norm = W / ||W||
        cosine = nn.functional.linear(
            nn.functional.normalize(input_features),
            nn.functional.normalize(self.weight)
        )
        
        # 2. Compute sine using trigonometric identity: sin^2(theta) + cos^2(theta) = 1
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        
        # 3. Compute cosine additions: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            # If cos(theta) > 0, apply margin, else keep standard cosine
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Handle boundaries to ensure stability when theta + m > pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # 4. Construct one-hot representation of labels
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 5. Apply the margin ONLY to the target logits (y_i)
        # Logits = s * (one_hot * cos(theta + m) + (1 - one_hot) * cos(theta))
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output
