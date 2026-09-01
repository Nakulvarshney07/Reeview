import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from src.aspect_model import AspectPredictionNet
from src.subaspect_model import ConditionalSubAspectNet

class ProductAspectPredictionModel(nn.Module):
    """
    V2 Supervised Neural Product Experience Aspect & Sub-aspect Prediction Model.
    
    Architecture:
    Input Product Document -> Text Encoder -> Product Embedding (dim=384)
       -> Trainable Aspect Prediction Head -> Multi-label Aspects & Confidence Scores
       -> Conditional Sub-Aspect Prediction Head -> Aspect-specific Sub-aspects
    """
    def __init__(self, target_aspects: List[str] = None, target_subaspects: List[str] = None, encoder_name: str = "all-MiniLM-L6-v2", device: str = None):
        super(ProductAspectPredictionModel, self).__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.encoder_name = encoder_name
        try:
            self.encoder = SentenceTransformer(encoder_name, device=self.device, local_files_only=True)
        except Exception:
            self.encoder = SentenceTransformer(encoder_name, device=self.device)

        
        # Freeze sentence encoder weights so training updates our custom neural heads
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        if hasattr(self.encoder, "get_embedding_dimension"):
            self.embed_dim = self.encoder.get_embedding_dimension()
        else:
            self.embed_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Default taxonomy targets if none provided
        if not target_aspects:
            target_aspects = [
                "Protein Content & Nutritional Value", "Mixability & Solubility", "Taste & Flavour",
                "Ingredient Quality & Safety", "Comfort & Fit", "Material Quality & Breathability",
                "Build Quality & Durability", "Style & Aesthetic Appeal", "Sound Quality & Audio Clarity",
                "Performance & Technical Reliability", "Safety & Child Friendliness", "Educational Value",
                "Fragrance Profile & Longevity", "Grooming Performance", "Value for Money",
                "Effectiveness", "Ease of Use", "Battery Performance", "Packaging Quality"
            ]
        if not target_subaspects:
            target_subaspects = [
                "High protein percentage", "Low calories", "Easy to mix", "Smooth consistency",
                "Pleasant taste", "Sweetness control", "Vegetarian", "Clinically tested",
                "Cushioning", "Fit accuracy", "Long-duration comfort", "Sturdy frame",
                "Durability", "Clear audio", "Rich bass", "Fast charging", "Long runtime",
                "Child safety", "Affordable price", "Value", "Ease of cleaning", "Smell longevity"
            ]
            
        self.target_aspects = target_aspects
        self.target_subaspects = target_subaspects
        
        # Prediction Heads
        self.aspect_head = AspectPredictionNet(input_dim=self.embed_dim, num_aspects=len(target_aspects)).to(self.device)
        self.subaspect_head = ConditionalSubAspectNet(feature_dim=self.embed_dim * 2, num_subaspects=len(target_subaspects)).to(self.device)

    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """Encodes list of strings into PyTorch embedding tensor."""
        embeddings = self.encoder.encode(text_list, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        return embeddings.detach().clone().to(self.device)



    def forward(self, input_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for batch of input product texts.
        returns (aspect_logits, subaspect_logits)
        """
        product_embeds = self.encode_text(input_texts)
        aspect_logits = self.aspect_head(product_embeds)
        
        # Default aspect representation vector for batch
        aspect_embeds = torch.zeros_like(product_embeds)
        combined_feats = torch.cat([product_embeds, aspect_embeds], dim=-1)
        subaspect_logits = self.subaspect_head(combined_feats)
        
        return aspect_logits, subaspect_logits

    def predict(self, raw_input_text: str, top_k: int = 8, min_aspects: int = 5, confidence_threshold: float = 0.40) -> Dict[str, Any]:
        """
        Executes end-to-end model inference on raw input text.
        Returns top 5-10 experience aspects and conditionally predicted sub-aspects.
        """
        self.eval()
        with torch.no_grad():
            prod_embed = self.encode_text([raw_input_text]) # (1, 384)
            aspect_logits = self.aspect_head(prod_embed)   # (1, num_aspects)
            aspect_probs = torch.sigmoid(aspect_logits).squeeze(0).cpu().numpy()
            
            # Rank target aspects by predicted confidence
            sorted_indices = np.argsort(aspect_probs)[::-1]
            
            predictions = []
            for idx in sorted_indices:
                prob = float(aspect_probs[idx])
                aspect_name = self.target_aspects[idx]
                
                # Check dynamic confidence threshold or minimum count requirement
                if prob >= confidence_threshold or len(predictions) < min_aspects:
                    if len(predictions) >= top_k:
                        break
                        
                    # Compute conditional sub-aspects for selected aspect
                    asp_embed = self.encode_text([aspect_name]) # (1, 384)
                    comb_feats = torch.cat([prod_embed, asp_embed], dim=-1) # (1, 768)
                    sub_logits = self.subaspect_head(comb_feats)
                    sub_probs = torch.sigmoid(sub_logits).squeeze(0).cpu().numpy()
                    
                    sub_sorted_indices = np.argsort(sub_probs)[::-1]
                    predicted_subs = []
                    
                    # Normalized aspect tensor for cosine similarity domain filtering
                    asp_norm = F.normalize(asp_embed, p=2, dim=-1)
                    
                    for s_idx in sub_sorted_indices:
                        sub_name = self.target_subaspects[s_idx]
                        sub_prob = float(sub_probs[s_idx])
                        
                        # Semantic domain sanity check
                        sub_emb = self.encode_text([sub_name])
                        sub_norm = F.normalize(sub_emb, p=2, dim=-1)
                        sim = float(torch.mm(asp_norm, sub_norm.T).cpu().squeeze())
                        
                        # Sub-aspect must be semantically related to the aspect (similarity >= 0.20)
                        if sim >= 0.20 and (sub_prob >= 0.20 or len(predicted_subs) < 2):
                            if sub_name not in predicted_subs:
                                predicted_subs.append(sub_name)
                        if len(predicted_subs) >= 4:
                            break
                            
                    predictions.append({
                        "aspect": aspect_name,
                        "sub_aspects": predicted_subs,
                        "confidence": round(prob, 2)
                    })

                    
            return {
                "predictions": predictions
            }

    def save_checkpoint(self, path: str):
        """Saves trainable neural weights and taxonomy metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "aspect_head": self.aspect_head.state_dict(),
            "subaspect_head": self.subaspect_head.state_dict(),
            "target_aspects": self.target_aspects,
            "target_subaspects": self.target_subaspects,
            "encoder_name": self.encoder_name
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Loads trained neural weights and taxonomy metadata."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.target_aspects = checkpoint.get("target_aspects", self.target_aspects)
            self.target_subaspects = checkpoint.get("target_subaspects", self.target_subaspects)
            self.aspect_head = AspectPredictionNet(input_dim=self.embed_dim, num_aspects=len(self.target_aspects)).to(self.device)
            self.subaspect_head = ConditionalSubAspectNet(feature_dim=self.embed_dim * 2, num_subaspects=len(self.target_subaspects)).to(self.device)
            self.aspect_head.load_state_dict(checkpoint["aspect_head"])
            self.subaspect_head.load_state_dict(checkpoint["subaspect_head"])
            self.to(self.device)
            self.eval()

