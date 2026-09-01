import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalSubAspectNet(nn.Module):
    """
    Conditional Sub-Aspect Predictor Network.
    
    Predicts sub-aspect probabilities conditionally based on the concatenation
    of the product vector E_product (dim=384) and predicted aspect vector E_aspect (dim=384).
    """
    def __init__(self, feature_dim: int = 768, num_subaspects: int = 40):
        super(ConditionalSubAspectNet, self).__init__()
        self.feature_dim = feature_dim
        self.num_subaspects = num_subaspects
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_subaspects)
        )
        
    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        combined_features: Tensor of shape (batch_size, 768)
        returns sub-aspect logits tensor of shape (batch_size, num_subaspects)
        """
        return self.net(combined_features)
