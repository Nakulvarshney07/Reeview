import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from src.dataset import DatasetManager
from src.predict import PredictorEngine
from src.llama_baseline import LlamaBaseline

class AspectEvaluator:
    """
    Comprehensive Evaluation Module comparing:
    - OUR TRAINED V2 MODEL
    - LLAMA 3.2 BASELINE
    against Human Consensus Ground Truth.
    
    Metrics:
    - Aspect Precision, Recall, F1
    - Sub-aspect Precision, Recall, F1
    - Invalid Aspect Rate
    - Redundancy Rate
    - Human Alignment Score
    """
    def __init__(self, sim_threshold: float = 0.55):
        try:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.sim_threshold = sim_threshold


    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between two strings for evaluation matching."""
        if not text1 or not text2:
            return 0.0
        vecs = self.encoder.encode([text1, text2], convert_to_tensor=True, show_progress_bar=False)
        sim = float(np.dot(vecs[0].cpu().numpy(), vecs[1].cpu().numpy()) / (
            np.linalg.norm(vecs[0].cpu().numpy()) * np.linalg.norm(vecs[1].cpu().numpy()) + 1e-8
        ))
        return sim

    def evaluate_predictions(self, predicted_items: List[Dict[str, Any]], ground_truth_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluates a set of predicted aspects & sub-aspects against human consensus ground truth.
        """
        if not predicted_items or not ground_truth_items:
            return {
                "aspect_precision": 0.0, "aspect_recall": 0.0, "aspect_f1": 0.0,
                "subaspect_precision": 0.0, "subaspect_recall": 0.0, "subaspect_f1": 0.0,
                "invalid_aspect_rate": 1.0, "redundancy_rate": 0.0, "human_alignment": 0.0
            }

        pred_aspect_names = [p.get("aspect", "") for p in predicted_items]
        gt_aspect_names = [g.get("aspect", "") for g in ground_truth_items]

        # 1. Aspect Precision & Recall
        matched_gt_indices = set()
        matched_pred_indices = set()

        for p_idx, p_name in enumerate(pred_aspect_names):
            for g_idx, g_name in enumerate(gt_aspect_names):
                sim = self.calculate_similarity(p_name, g_name)
                if sim >= self.sim_threshold:
                    matched_pred_indices.add(p_idx)
                    matched_gt_indices.add(g_idx)

        aspect_precision = len(matched_pred_indices) / max(1, len(pred_aspect_names))
        aspect_recall = len(matched_gt_indices) / max(1, len(gt_aspect_names))
        aspect_f1 = (2 * aspect_precision * aspect_recall / (aspect_precision + aspect_recall)) if (aspect_precision + aspect_recall) > 0 else 0.0

        # 2. Sub-aspect Precision & Recall
        pred_subs = []
        for p in predicted_items:
            pred_subs.extend(p.get("sub_aspects", []) or p.get("subaspects", []))

        gt_subs = []
        for g in ground_truth_items:
            gt_subs.extend(g.get("sub_aspects", []) or g.get("subaspects", []))

        sub_tp = 0
        matched_gt_subs = set()
        for p_sub in pred_subs:
            for g_idx, g_sub in enumerate(gt_subs):
                if self.calculate_similarity(p_sub, g_sub) >= self.sim_threshold:
                    sub_tp += 1
                    matched_gt_subs.add(g_idx)
                    break

        sub_precision = sub_tp / max(1, len(pred_subs)) if pred_subs else 0.0
        sub_recall = len(matched_gt_subs) / max(1, len(gt_subs)) if gt_subs else 0.0
        sub_f1 = (2 * sub_precision * sub_recall / (sub_precision + sub_recall)) if (sub_precision + sub_recall) > 0 else 0.0

        # 3. Invalid Aspect Rate (aspects with < 0.35 similarity to ANY ground truth)
        invalid_count = 0
        for p_name in pred_aspect_names:
            best_sim = max([self.calculate_similarity(p_name, g_name) for g_name in gt_aspect_names], default=0.0)
            if best_sim < 0.35:
                invalid_count += 1
        invalid_aspect_rate = invalid_count / max(1, len(pred_aspect_names))

        # 4. Redundancy Rate (pairwise similarity >= 0.70 among predictions)
        redundant_pairs = 0
        n_preds = len(pred_aspect_names)
        num_pairs = n_preds * (n_preds - 1) / 2
        for i in range(n_preds):
            for j in range(i + 1, n_preds):
                if self.calculate_similarity(pred_aspect_names[i], pred_aspect_names[j]) >= 0.70:
                    redundant_pairs += 1
        redundancy_rate = redundant_pairs / max(1, num_pairs)

        # 5. Human Alignment Score
        human_alignment = max(0.0, 1.0 - (invalid_aspect_rate * 0.5 + redundancy_rate * 0.5)) * aspect_f1

        return {
            "aspect_precision": round(aspect_precision, 4),
            "aspect_recall": round(aspect_recall, 4),
            "aspect_f1": round(aspect_f1, 4),
            "subaspect_precision": round(sub_precision, 4),
            "subaspect_recall": round(sub_recall, 4),
            "subaspect_f1": round(sub_f1, 4),
            "invalid_aspect_rate": round(invalid_aspect_rate, 4),
            "redundancy_rate": round(redundancy_rate, 4),
            "human_alignment": round(human_alignment, 4)
        }

    def run_benchmark_on_test_set(self) -> Dict[str, Any]:
        """
        Executes complete comparative evaluation of OUR MODEL vs LLAMA 3.2 BASELINE.
        """
        dm = DatasetManager()
        train_records, val_records, test_records = dm.prepare_splits()
        predictor = PredictorEngine()
        llama = LlamaBaseline()

        our_model_metrics_list = []
        llama_metrics_list = []

        print("\n==========================================================")
        print("    RUNNING BENCHMARK EVALUATION ON UNSEEN TEST PRODUCTS   ")
        print("==========================================================")

        for record in test_records:
            pid = record["product_id"]
            pname = record["name"]
            input_text = record["input_text"]
            gt = record["ground_truth"]

            # Our Model Predictions
            our_preds = predictor.predict(input_text)["predictions"]

            # Llama Baseline Predictions
            llama_preds = llama.get_baseline_aspects(record)["extracted_aspects"]

            # Evaluate
            our_eval = self.evaluate_predictions(our_preds, gt)
            llama_eval = self.evaluate_predictions(llama_preds, gt)

            our_model_metrics_list.append(our_eval)
            llama_metrics_list.append(llama_eval)

            print(f"\nProduct: [{pid}] {pname[:40]}...")
            print(f"  OUR MODEL -> F1: {our_eval['aspect_f1']} | Sub F1: {our_eval['subaspect_f1']} | Alignment: {our_eval['human_alignment']}")
            print(f"  LLAMA 3.2 -> F1: {llama_eval['aspect_f1']} | Sub F1: {llama_eval['subaspect_f1']} | Alignment: {llama_eval['human_alignment']}")

        # Average Metrics across test set
        def avg_metrics(m_list):
            keys = m_list[0].keys()
            return {k: round(float(np.mean([m[k] for m in m_list])), 4) for k in keys}

        avg_our = avg_metrics(our_model_metrics_list)
        avg_llama = avg_metrics(llama_metrics_list)

        print("\n==========================================================")
        print("                  FINAL BENCHMARK SUMMARY                 ")
        print("==========================================================")
        print(f"{'Metric':<25} | {'OUR MODEL (V2 Neural)':<22} | {'LLAMA 3.2 (Baseline)':<20}")
        print("-" * 75)
        for k in avg_our.keys():
            print(f"{k:<25} | {avg_our[k]:<22} | {avg_llama[k]:<20}")
        print("==========================================================")

        return {
            "our_model_summary": avg_our,
            "llama_baseline_summary": avg_llama
        }

if __name__ == "__main__":
    evaluator = AspectEvaluator()
    evaluator.run_benchmark_on_test_set()
