import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any
from src.dataset import DatasetManager, ProductAspectDataset
from src.model import ProductAspectPredictionModel

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_aspect_model.pt")

class Trainer:
    """
    Supervised PyTorch Neural Trainer for Product Experience Aspect Prediction.
    """
    def __init__(self, data_dir: str = None, epochs: int = 25, lr: float = 1e-3):
        self.data_dir = data_dir
        self.epochs = epochs
        self.lr = lr
        self.dm = DatasetManager(data_dir=data_dir)
        
    def calculate_metrics(self, pred_probs: np.ndarray, targets: np.ndarray, threshold: float = 0.40) -> Dict[str, float]:
        """Calculates Precision, Recall, and F1 across prediction matrix."""
        preds = (pred_probs >= threshold).astype(int)
        
        tp = np.sum((preds == 1) & (targets == 1))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4)
        }

    def run(self):
        print("==========================================================")
        print("   STARTING V2 SUPERVISED NEURAL MODEL TRAINING PIPELINE   ")
        print("==========================================================")
        
        train_records, val_records, test_records = self.dm.prepare_splits()
        all_records = train_records + val_records + test_records
        
        target_aspects, target_subaspects = self.dm.get_target_taxonomies(all_records)
        print(f"Loaded {len(all_records)} products across domains.")
        print(f"Train: {len(train_records)} | Validation: {len(val_records)} | Test: {len(test_records)}")
        print(f"Target Aspect Space ({len(target_aspects)} aspects): {target_aspects[:5]}...")
        print(f"Target Sub-aspect Space ({len(target_subaspects)} sub-aspects): {target_subaspects[:5]}...")
        
        train_ds = ProductAspectDataset(train_records, target_aspects, target_subaspects)
        val_ds = ProductAspectDataset(val_records, target_aspects, target_subaspects)
        test_ds = ProductAspectDataset(test_records, target_aspects, target_subaspects)
        
        model = ProductAspectPredictionModel(target_aspects=target_aspects, target_subaspects=target_subaspects)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_f1 = 0.0
        print("\n--- Training Progress ---", flush=True)
        print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Precision':<10} | {'Recall':<10} | {'Val F1':<10} | {'Status':<12}", flush=True)
        print("-" * 78, flush=True)
        
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            
            for item in train_ds:
                optimizer.zero_grad()
                input_text = item["input_text"]
                target = item["aspect_target"].unsqueeze(0).to(model.device) # (1, num_aspects)
                sub_target = item["subaspect_target"].unsqueeze(0).to(model.device) # (1, num_subaspects)
                
                prod_embed = model.encode_text([input_text])
                aspect_logits = model.aspect_head(prod_embed)
                asp_loss = criterion(aspect_logits, target)
                
                comb_feats = torch.cat([prod_embed, prod_embed], dim=-1)
                sub_logits = model.subaspect_head(comb_feats)
                sub_loss = criterion(sub_logits, sub_target)
                
                loss = asp_loss + sub_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= max(1, len(train_ds))
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for item in val_ds:
                    input_text = item["input_text"]
                    target = item["aspect_target"].numpy()
                    sub_target = item["subaspect_target"].unsqueeze(0).to(model.device)
                    
                    prod_embed = model.encode_text([input_text])
                    aspect_logits = model.aspect_head(prod_embed)
                    asp_loss = criterion(aspect_logits, torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(model.device))
                    
                    comb_feats = torch.cat([prod_embed, prod_embed], dim=-1)
                    sub_logits = model.subaspect_head(comb_feats)
                    sub_loss = criterion(sub_logits, sub_target)
                    
                    val_loss += (asp_loss + sub_loss).item()
                    
                    probs = torch.sigmoid(aspect_logits).squeeze(0).cpu().numpy()
                    val_preds.append(probs)
                    val_targets.append(target)

                    
            val_loss /= max(1, len(val_ds))
            metrics = self.calculate_metrics(np.array(val_preds), np.array(val_targets))
            
            status = ""
            if metrics["f1"] >= best_val_f1:
                best_val_f1 = metrics["f1"]
                model.save_checkpoint(CHECKPOINT_PATH)
                status = "[BEST SAVED]"
                
            print(f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f} | {status:<12}", flush=True)

            
        print("\n--- Training Complete ---", flush=True)
        print(f"Best Validation F1: {best_val_f1:.4f}", flush=True)
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}", flush=True)
        
        # Test Evaluation
        print("\n--- Evaluating on Unseen Test Split ---", flush=True)
        model.load_checkpoint(CHECKPOINT_PATH)
        model.eval()
        
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for item in test_ds:
                prod_embed = model.encode_text([item["input_text"]])
                aspect_logits = model.aspect_head(prod_embed)
                probs = torch.sigmoid(aspect_logits).squeeze(0).cpu().numpy()
                test_preds.append(probs)
                test_targets.append(item["aspect_target"].numpy())
                
        test_metrics = self.calculate_metrics(np.array(test_preds), np.array(test_targets))
        print(f"Test Precision: {test_metrics['precision']:.4f}", flush=True)
        print(f"Test Recall:    {test_metrics['recall']:.4f}", flush=True)
        print(f"Test F1 Score:  {test_metrics['f1']:.4f}", flush=True)
        print("==========================================================", flush=True)
        return test_metrics

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()

