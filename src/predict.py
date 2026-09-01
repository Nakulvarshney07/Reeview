import os
import json
import argparse
from typing import Dict, List, Any
from src.model import ProductAspectPredictionModel
from src.dataset import DatasetManager
from src.train import CHECKPOINT_PATH, Trainer

class PredictorEngine:
    """
    High-Level Model Inference Engine for V2 Aspect & Sub-aspect Prediction.
    """
    def __init__(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_PATH
        self.checkpoint_path = checkpoint_path
        
        # Ensure model checkpoint exists; train if necessary
        if not os.path.exists(self.checkpoint_path):
            print(f"[PredictorEngine] No trained checkpoint found at {self.checkpoint_path}. Running training first...")
            trainer = Trainer(epochs=15)
            trainer.run()
            
        # Load dataset targets
        dm = DatasetManager()
        records = dm.load_all_products()
        target_aspects, target_subaspects = dm.get_target_taxonomies(records)
        
        self.model = ProductAspectPredictionModel(target_aspects=target_aspects, target_subaspects=target_subaspects)
        self.model.load_checkpoint(self.checkpoint_path)
        self.model.eval()

    def predict(self, raw_input_text: str) -> Dict[str, Any]:
        """
        Main Prediction Method.
        Accepts arbitrary product text document and returns 5-10 experience aspects with sub-aspects.
        """
        if not raw_input_text or not raw_input_text.strip():
            return {"predictions": []}
            
        output = self.model.predict(raw_input_text, top_k=10, min_aspects=5, confidence_threshold=0.40)
        return output

def main():
    parser = argparse.ArgumentParser(description="Predict product experience aspects & sub-aspects.")
    parser.add_argument("--text", type=str, help="Raw product text input")
    args = parser.parse_args()

    sample_text = args.text or """
    Product Title:
    MuscleBlaze Biozyme Performance Whey Protein Powder (French Vanilla Creme, 1kg)

    Specifications:
    - Flavor: French Vanilla Creme
    - Protein: 25g per scoop
    - Diet Type: Vegetarian
    - Absorption: 50% Higher Protein Absorption

    5-star reviews:
    - "Awesome Taste": Mixes smoothly with water and has great vanilla flavour. Easy on stomach.
    - "Great Muscle Recovery": Noticed good muscle gains after 2 weeks of use.

    1-star reviews:
    - "Too Sweet": Tastes artificial and very sweet. Did not dissolve quickly in cold milk.
    """

    engine = PredictorEngine()
    result = engine.predict(sample_text)
    print("\n--- Model Output JSON ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
