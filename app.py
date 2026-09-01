import os
import json
from flask import Flask, request, jsonify, send_from_directory
from src.predict import PredictorEngine
from src.evaluator import AspectEvaluator
from src.dataset import DatasetManager
from src.llama_baseline import LlamaBaseline
from src.preprocessing import Preprocessor

app = Flask(__name__, static_folder="frontend", static_url_path="")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

predictor = PredictorEngine()
evaluator = AspectEvaluator()
dataset_manager = DatasetManager(data_dir=DATA_DIR)
llama_baseline = LlamaBaseline()

PRODUCT_MAP = {
    "P001": ("MuscleBlaze Biozyme Whey Protein Powder", "product_1_whey.json"),
    "P002": ("MuscleBlaze Micronised Creatine Monohydrate", "product_2_creatine.json"),
    "P003": ("Wooden Kitchen Set with Refrigerator for Kids", "product_3_kitchen_playset.json"),
    "P004": ("Butterfly Edufields 10-in-1 STEM Robotics Kit", "product_4_robotics_kit.json"),
    "P005": ("Denver Hamilton EDP & Beardo Whisky Smoke Perfume", "product_5_perfume.json"),
    "P006": ("Boldfit Heavy Resistance Band Set", "product_6_resistance_bands.json"),
    "P007": ("Cult Impact Deep Tissue Massage Gun", "product_7_massage_gun.json"),
    "P008": ("Biltoxi Interactive Tool / Roleplay Toy Bench Playset", "product_8_toolset_toy.json"),
    "P009": ("TARBULL SuperBuddy Screen-Free Audio Storyteller", "product_9_audio_storyteller.json")
}

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/api/products", methods=["GET"])
def get_products():
    """Lists pre-loaded e-commerce products."""
    products = [
        {"id": pid, "name": name, "file": fname}
        for pid, (name, fname) in PRODUCT_MAP.items()
    ]
    return jsonify({"status": "success", "products": products})

@app.route("/api/product/<product_id>", methods=["GET"])
def get_product_detail(product_id):
    filename = PRODUCT_MAP.get(product_id, ("Default", "product_1_whey.json"))[1]
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    combined_text = Preprocessor.combine_product_text(data)
    data["combined_text"] = combined_text
    return jsonify({"status": "success", "data": data})

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    V2 Primary Endpoint:
    Accepts raw product text (title + specs + reviews) and outputs predicted 5-10 experience aspects + sub-aspects.
    """
    req = request.json or {}
    raw_text = req.get("raw_text", "").strip()

    if not raw_text:
        return jsonify({"status": "error", "message": "Please paste product title, specifications, and reviews."}), 400

    result = predictor.predict(raw_text)
    return jsonify({
        "status": "success",
        "model_type": "V2 Supervised Neural Network (PyTorch Multi-Label & Conditional Sub-Aspect Heads)",
        "predictions": result["predictions"]
    })

@app.route("/api/benchmark", methods=["GET", "POST"])
def run_benchmark():
    """
    Serves comparative evaluation metrics between Human Ground Truth, Llama 3.2 Baseline, and Our Trained Model.
    """
    benchmark_res = evaluator.run_benchmark_on_test_set()
    return jsonify({
        "status": "success",
        "benchmark": benchmark_res
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
