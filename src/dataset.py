import os
import json
import random
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import Preprocessor
from src.consensus import HumanConsensusEngine

class ProductAspectDataset(Dataset):
    """
    PyTorch Dataset for Product Experience Aspect & Sub-aspect Prediction.
    """
    def __init__(self, data_items: List[Dict[str, Any]], all_aspects: List[str], all_subaspects: List[str]):
        self.items = data_items
        self.all_aspects = all_aspects
        self.all_subaspects = all_subaspects
        self.aspect2idx = {asp: i for i, asp in enumerate(all_aspects)}
        self.sub2idx = {sub: i for i, sub in enumerate(all_subaspects)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        input_text = item["input_text"]
        gt_list = item["ground_truth"]

        # One-hot / Multi-hot target tensor for aspects and sub-aspects
        aspect_target = torch.zeros(len(self.all_aspects), dtype=torch.float32)
        subaspect_target = torch.zeros(len(self.all_subaspects), dtype=torch.float32)
        
        for gt in gt_list:
            asp = gt.get("aspect", "")
            if asp in self.aspect2idx:
                aspect_target[self.aspect2idx[asp]] = 1.0
            for sub in gt.get("sub_aspects", []):
                if sub in self.sub2idx:
                    subaspect_target[self.sub2idx[sub]] = 1.0

        return {
            "product_id": item["product_id"],
            "input_text": input_text,
            "aspect_target": aspect_target,
            "subaspect_target": subaspect_target,
            "ground_truth": gt_list,
            "raw_item": item
        }



class DatasetManager:
    """
    Manages loading product JSON files, building consensus ground truth,
    extracting unique aspect/sub-aspect targets, and splitting data into Train/Val/Test.
    """
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.data_dir = data_dir

    def load_all_products(self) -> List[Dict[str, Any]]:
        """Loads and processes all product JSON files into standardized records."""
        records = []
        if not os.path.exists(self.data_dir):
            return records

        for fname in sorted(os.listdir(self.data_dir)):
            if fname.startswith("product_") and fname.endswith(".json"):
                fpath = os.path.join(self.data_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    pdata = json.load(f)

                combined_text = Preprocessor.combine_product_text(pdata)
                consensus_info = HumanConsensusEngine.process_product_consensus(pdata)

                gt_aspects = []
                for gt in consensus_info["ground_truth"]:
                    gt_aspects.append({
                        "aspect": gt["aspect"],
                        "sub_aspects": gt["sub_aspects"]
                    })

                record = {
                    "product_id": pdata.get("product_id", fname.replace(".json", "")),
                    "name": pdata.get("name") or pdata.get("title", ""),
                    "category": pdata.get("category", "General"),
                    "input_text": combined_text,
                    "specs": pdata.get("specs", {}),
                    "reviews": pdata.get("reviews", {}),
                    "human_annotations": pdata.get("human_annotations", {}),
                    "ground_truth": gt_aspects,
                    "llama_output": pdata.get("llama_output", [])
                }
                records.append(record)
        return records

    def prepare_splits(self, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits products into Train, Validation, and Test splits while holding out an unseen category.
        """
        records = self.load_all_products()
        if not records:
            return [], [], []

        # Hold out product_9 (Audio Storyteller / Smart Device) for Category Generalization Test
        unseen_test = [r for r in records if r["product_id"] in ["P009", "P008"]]
        remaining = [r for r in records if r["product_id"] not in ["P009", "P008"]]

        random.seed(seed)
        random.shuffle(remaining)

        # 5 train, 2 val, 2 test (including holdout)
        val_size = max(1, int(len(remaining) * 0.25))
        val_set = remaining[:val_size]
        train_set = remaining[val_size:]
        test_set = unseen_test + remaining[:1]

        # Write to JSON files
        with open(os.path.join(self.data_dir, "train.json"), "w", encoding="utf-8") as f:
            json.dump(train_set, f, indent=2)
        with open(os.path.join(self.data_dir, "validation.json"), "w", encoding="utf-8") as f:
            json.dump(val_set, f, indent=2)
        with open(os.path.join(self.data_dir, "test.json"), "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2)

        return train_set, val_set, test_set

    def get_target_taxonomies(self, records: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Extracts unique aspect names and sub-aspect strings from ground truth annotations across datasets."""
        aspects = set()
        subaspects = set()
        for r in records:
            for gt in r["ground_truth"]:
                asp = gt["aspect"].strip()
                if asp:
                    aspects.add(asp)
                for sub in gt.get("sub_aspects", []):
                    if sub.strip():
                        subaspects.add(sub.strip())

        return sorted(list(aspects)), sorted(list(subaspects))
