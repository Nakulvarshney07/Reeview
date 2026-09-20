import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer

# # The 10 universal aspects our model classifies into.
# # These are the OUTPUT CLASSES of our neural classifier.
# ASPECT_LABELS = [
#     "Comfort and Ergonomics",
#     "Build Quality and Durability",
#     "Performance and Functionality",
#     "Design and Aesthetics",
#     "Assembly and Ease of Use",
#     "Value for Money",
#     "Customer Support and Warranty",
#     "Packaging and Delivery",
#     "Safety and Health",
#     "Sensory Experience",
# ]


class AspectClassifierHead(nn.Module):
    """
    A 2-layer MLP that takes a 384-dim SentenceTransformer embedding
    and outputs a logit score for each of the 10 aspect classes.

    This is the part that is TRAINED on labeled data.
    Architecture:
        Linear(384 -> 256) -> BatchNorm -> ReLU -> Dropout(0.3)
        -> Linear(256 -> 128) -> ReLU -> Dropout(0.2)
        -> Linear(128 -> num_aspects)

    Output: raw logits (not probabilities). Use sigmoid to get [0,1].
    """
    def __init__(self, input_dim: int = 384, num_aspects: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_aspects),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Kept for backwards-compat with old train.py imports
class AspectPredictionNet(nn.Module):
    def __init__(self, input_dim: int, num_aspects: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_aspects)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        return self.fc2(self.relu(self.fc1(x)))

class ConditionalSubAspectNet(nn.Module):
    def __init__(self, feature_dim: int, num_subaspects: int):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_subaspects)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        return self.fc2(self.relu(self.fc1(x)))


class ProductAspectPredictionModel(nn.Module):
    """
    TRUE Trained Multi-Label Aspect Classifier.

    Architecture (two-stage):
    ┌─────────────────────────────────────────────────────────┐
    │  Stage 1: FROZEN Sentence Encoder (SentenceTransformer) │
    │    - 22M pre-trained parameters (all-MiniLM-L6-v2)     │
    │    - Converts any text → 384-dim semantic vector        │
    │    - NOT trained by us, loaded from HuggingFace cache   │
    ├─────────────────────────────────────────────────────────┤
    │  Stage 2: TRAINED Aspect Classifier Head (MLP)          │
    │    - 384 → 256 → 128 → 10  (our trained layers)        │
    │    - Trained with Binary Cross-Entropy loss             │
    │    - Learns: "which aspects does this embedding imply?" │
    │    - Output: 10 probability scores, one per aspect      │
    └─────────────────────────────────────────────────────────┘

    This is NOT cosine similarity. The MLP LEARNS the mapping
    from embedding space to aspect labels through gradient descent
    on labeled training examples.
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        target_aspects: List[str] = None,
        target_subaspects: List[str] = None,
        device: str = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.encoder_name = encoder_name

        # Stage 1: Frozen sentence encoder (pre-trained, not ours to train)
        try:
            self.encoder = SentenceTransformer(encoder_name, local_files_only=True)
        except Exception:
            self.encoder = SentenceTransformer(encoder_name)

        # The aspect labels this model predicts
        # self.aspect_labels = ASPECT_LABELS
        self.num_aspects = len(self.aspect_labels)

        # Stage 2: Our trained MLP classifier head
        # This is registered as a PyTorch module so its parameters are trained
        self.aspect_head = AspectClassifierHead(
            input_dim=384,
            num_aspects=self.num_aspects
        )
        self.aspect_head.to(self.device)

        # Dummy subaspect_head for train.py compatibility
        self.subaspect_head = nn.Linear(768, 1)
        self.subaspect_head.to(self.device)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into a (batch_size, 384) tensor.
        The encoder is frozen — we only train the aspect_head on top.
        """
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=str(self.device)
        )
        return embeddings.to(self.device).detach().clone()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Forward pass: text -> embedding -> logits (batch_size, num_aspects)."""
        embeddings = self.encode_text(texts)
        return self.aspect_head(embeddings)

    def predict(
        self,
        raw_input_text: str,
        top_k: int = 8,
        min_aspects: int = 3,
        confidence_threshold: float = 0.40,
    ) -> Dict[str, Any]:
        """
        Inference using the TRAINED classifier head.

        Flow:
        1. Encode text -> 384-dim vector (frozen encoder)
        2. Pass through trained MLP -> 10 raw logits
        3. Sigmoid -> 10 probabilities in [0, 1]
        4. Return aspects where probability > threshold
        """
        if not raw_input_text or not raw_input_text.strip():
            return {"predictions": []}

        self.eval()
        with torch.no_grad():
            # Step 1: Encode text with frozen SentenceTransformer
            embedding = self.encode_text([raw_input_text])  # (1, 384)

            # Step 2: Pass through TRAINED MLP head -> raw logits
            logits = self.aspect_head(embedding)             # (1, 10)

            # Step 3: Sigmoid converts logits to probabilities [0, 1]
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (10,)

        # Step 4: Build predictions — aspects with prob > threshold
        results = []
        for i, (label, prob) in enumerate(zip(self.aspect_labels, probs)):
            results.append((label, float(prob)))

        # Sort by probability (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        predictions = []
        for label, prob in results:
            if prob < confidence_threshold and len(predictions) >= min_aspects:
                break
            if len(predictions) >= top_k:
                break
            predictions.append({
                "aspect": label,
                "confidence": round(prob, 4),
            })

        return {"predictions": predictions}

    def save_checkpoint(self, path: str):
        """Save only the trained MLP head weights (encoder is pre-trained)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "aspect_head_state": self.aspect_head.state_dict(),
            "encoder_name": self.encoder_name,
            "aspect_labels": self.aspect_labels,
        }, path)
        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, path: str):
        """Load trained MLP head weights from checkpoint."""
        if not os.path.exists(path):
            print(f"[Warning] No checkpoint at {path}. Using untrained weights.")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "aspect_head_state" in ckpt:
            self.aspect_head.load_state_dict(ckpt["aspect_head_state"])
        print(f"[Checkpoint] Loaded from {path}")
