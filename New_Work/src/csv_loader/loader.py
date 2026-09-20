import csv
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.config import PRODUCTS_CSV_PATH, CATEGORIES_CSV_PATH

class CSVProductLoader:
    """
    Efficient, streaming loader for large Amazon product datasets (~1.4M rows).
    Avoids loading the entire CSV into memory by streaming row by row with offset and limit.
    """
    def __init__(self, products_path: Optional[Path] = None, categories_path: Optional[Path] = None):
        self.products_path = products_path or PRODUCTS_CSV_PATH
        self.categories_path = categories_path or CATEGORIES_CSV_PATH
        self.category_map: Dict[int, str] = self._load_category_map()

    def _load_category_map(self) -> Dict[int, str]:
        """Loads category id to category name mapping from amazon_categories.csv."""
        cat_map: Dict[int, str] = {}
        if not os.path.exists(self.categories_path):
            return cat_map
            
        try:
            with open(self.categories_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("id")
                    cname = row.get("category_name")
                    if cid and cname:
                        try:
                            cat_map[int(cid)] = cname.strip()
                        except ValueError:
                            continue
        except Exception as e:
            print(f"[CSVLoader] Warning loading category map: {e}")
            
        return cat_map

    @staticmethod
    def _parse_float(val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            f = float(val)
            return f
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_int(val: Any) -> Optional[int]:
        if val is None or val == "":
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if not val:
            return False
        s = str(val).strip().lower()
        return s in ("true", "1", "t", "yes", "y")

    def load_products(self, start: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Stream rows from the products CSV starting from `start` up to `limit` rows.
        Returns a list of raw parsed dictionary items.
        """
        if not os.path.exists(self.products_path):
            raise FileNotFoundError(f"Amazon products CSV not found at: {self.products_path}")

        results: List[Dict[str, Any]] = []
        current_idx = 0
        end_idx = start + limit

        with open(self.products_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if current_idx < start:
                    current_idx += 1
                    continue
                if current_idx >= end_idx:
                    break

                category_id = self._parse_int(row.get("category_id"))
                category_name = self.category_map.get(category_id, f"Category_{category_id}" if category_id else "General")

                item = {
                    "row_index": current_idx,
                    "asin": (row.get("asin") or "").strip(),
                    "title": (row.get("title") or "").strip(),
                    "img_url": (row.get("imgUrl") or "").strip(),
                    "product_url": (row.get("productURL") or "").strip(),
                    "stars": self._parse_float(row.get("stars")),
                    "reviews": self._parse_int(row.get("reviews")),
                    "price": self._parse_float(row.get("price")),
                    "list_price": self._parse_float(row.get("listPrice")),
                    "category_id": category_id,
                    "category_name": category_name,
                    "is_best_seller": self._parse_bool(row.get("isBestSeller")),
                    "bought_in_last_month": self._parse_int(row.get("boughtInLastMonth")) or 0,
                    "raw_row": dict(row)
                }

                results.append(item)
                current_idx += 1

        return results
