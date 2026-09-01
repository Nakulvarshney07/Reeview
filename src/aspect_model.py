import torch
import torch.nn as nn
import torch.nn.functional as F

class AspectPredictionNet(nn.Module):
    """
    Multi-Label Neural Aspect Classifier.
    
    Takes dense text representation vector E_product (dim=384)
    and predicts multi-label binary logits over experience aspect target space.
    """
    def __init__(self, input_dim: int = 384, num_aspects: int = 20):
        super(AspectPredictionNet, self).__init__()
        self.input_dim = input_dim
        self.num_aspects = num_aspects
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_aspects)
        )
        
    def forward(self, product_embeddings: torch.Tensor) -> torch.Tensor:
        """
        product_embeddings: Tensor of shape (batch_size, input_dim)
        returns logits tensor of shape (batch_size, num_aspects)
        """
        return self.net(product_embeddings)
