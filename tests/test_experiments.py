import os
import json
from src.predict import PredictorEngine
from src.evaluator import AspectEvaluator
from src.dataset import DatasetManager

def run_experiment_1_standard_test():
    print("\n--- EXPERIMENT 1: STANDARD TEST (Unseen Test Products) ---")
    evaluator = AspectEvaluator()
    results = evaluator.run_benchmark_on_test_set()
    print("[OK] Experiment 1 Complete.")
    return results

def run_experiment_2_paraphrase_test():
    print("\n--- EXPERIMENT 2: PARAPHRASE SENSITIVITY TEST ---")
    predictor = PredictorEngine()

    text_original = """
    Product: Air Cushion Running Shoes
    Specifications: Breathable Mesh Upper, Rubber Outsole, Cushioned Midsole.
    5-star reviews: Extremely comfortable for long walks and daily jogging. Excellent foot support.
    """

    text_paraphrased = """
    Product: Athletic Jogging Sneakers
    Specifications: Lightweight Woven Fabric, Durable Rubber Sole, Soft Inner Padding.
    5-star reviews: Great shock absorption during extended exercise. Highly supportive for feet after hours of wear.
    """

    res1 = predictor.predict(text_original)["predictions"]
    res2 = predictor.predict(text_paraphrased)["predictions"]

    print("Original Text Aspects:", [r["aspect"] for r in res1])
    print("Paraphrased Text Aspects:", [r["aspect"] for r in res2])

    evaluator = AspectEvaluator()
    sims = [
        max([evaluator.calculate_similarity(a1["aspect"], a2["aspect"]) for a2 in res2], default=0.0)
        for a1 in res1
    ]
    avg_invariance = sum(sims) / max(1, len(sims))
    print(f"Semantic Paraphrase Invariance Score: {avg_invariance:.4f}")
    print("[OK] Experiment 2 Complete.")

def run_experiment_3_unseen_category_test():
    print("\n--- EXPERIMENT 3: UNSEEN CATEGORY GENERALIZATION TEST ---")
    predictor = PredictorEngine()

    unseen_text = """
    Product: Quantum Smartwatch Series 7 with AMOLED Display
    Specifications:
    - Screen: 1.4-inch High Resolution AMOLED
    - Battery: 300mAh, up to 7 days battery life
    - Sensors: Heart Rate Monitor, SpO2, Sleep Tracking
    - Water Resistance: 5 ATM

    5-star reviews:
    - "Battery lasts a full week": Really impressed with the long battery runtime and fast magnetic charger.
    - "Accurate Tracking": Heart rate and sleep metrics are spot on.

    1-star reviews:
    - "Screen Scratched Easily": Glass display scuffed after minor bump. App sync failed twice.
    """

    res = predictor.predict(unseen_text)["predictions"]
    print(f"Predicted Aspects for Unseen Smartwatch Category:", flush=True)
    for item in res:
        print(f"  - Aspect: {item['aspect']:<35} (Conf: {item['confidence']})", flush=True)
        print(f"    Sub-aspects: {', '.join(item['sub_aspects'])}", flush=True)
    print("[OK] Experiment 3 Complete.", flush=True)

def run_experiment_4_ablation_study():
    print("\n--- EXPERIMENT 4: ABLATION STUDY ---", flush=True)
    predictor = PredictorEngine()
    evaluator = AspectEvaluator()

    dm = DatasetManager()
    records = dm.load_all_products()
    if not records:
        return

    sample = records[0]
    gt = sample["ground_truth"]

    # 1. Specs Only
    specs_text = f"Product: {sample['name']}\nSpecifications:\n{json.dumps(sample['specs'])}"
    res_specs = predictor.predict(specs_text)["predictions"]
    eval_specs = evaluator.evaluate_predictions(res_specs, gt)

    # 2. Reviews Only
    rev_text = f"Product: {sample['name']}\nCustomer Reviews:\n{json.dumps(sample['reviews'])}"
    res_revs = predictor.predict(rev_text)["predictions"]
    eval_revs = evaluator.evaluate_predictions(res_revs, gt)

    # 3. Specs + Reviews Combined
    comb_text = sample["input_text"]
    res_comb = predictor.predict(comb_text)["predictions"]
    eval_comb = evaluator.evaluate_predictions(res_comb, gt)

    print(f"{'Input Configuration':<30} | {'Aspect F1':<12} | {'Sub F1':<12} | {'Human Alignment':<15}", flush=True)
    print("-" * 75, flush=True)
    print(f"{'1. Specifications Only':<30} | {eval_specs['aspect_f1']:<12} | {eval_specs['subaspect_f1']:<12} | {eval_specs['human_alignment']:<15}", flush=True)
    print(f"{'2. Customer Reviews Only':<30} | {eval_revs['aspect_f1']:<12} | {eval_revs['subaspect_f1']:<12} | {eval_revs['human_alignment']:<15}", flush=True)
    print(f"{'3. Specs + Reviews Combined':<30} | {eval_comb['aspect_f1']:<12} | {eval_comb['subaspect_f1']:<12} | {eval_comb['human_alignment']:<15}", flush=True)
    print("[OK] Experiment 4 Complete.", flush=True)

if __name__ == "__main__":
    print("==========================================================", flush=True)
    print("       RUNNING ALL EXPERIMENTAL VERIFICATION SUITES        ", flush=True)
    print("==========================================================", flush=True)
    run_experiment_1_standard_test()
    run_experiment_2_paraphrase_test()
    run_experiment_3_unseen_category_test()
    run_experiment_4_ablation_study()
    print("==========================================================", flush=True)

