"""
Amazon Product Demand Estimation Experiment
Using locally running Llama 3.2 via Ollama
============================================
Target variable: boughtInLastMonth
Split: 5 reference + 5 test per category (10 categories)
"""

import os, json, re, math, logging, time, random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR        = Path(r"C:\Users\Nakul\OneDrive\Desktop\MinorProject\New_Work")
OUTPUT_DIR      = BASE_DIR / "output"
GRAPHS_DIR      = OUTPUT_DIR / "graphs"

SELECTED_CSV    = OUTPUT_DIR / "selected_products_10_per_category.csv"
REVIEWS_JSON    = OUTPUT_DIR / "output_reviews_cleaned.json"
CATEGORIES_CSV  = BASE_DIR  / "amazon_categories.csv"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "")   # auto-detected below
TEMPERATURE     = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
RANDOM_SEED     = int(os.environ.get("RANDOM_SEED", "42"))
PROMPT_VERSION  = "v1.0"
MAX_RETRIES     = 2
REQUEST_TIMEOUT = 120
MAX_REVIEW_CHARS = 600

CATEGORY_MAP = {
    126: "Oral Care Products",
    47:  "Hair Care Products",
    52:  "Personal Care Products",
    167: "Household Cleaning Supplies",
    130: "Household Supplies",
    170: "Kitchen & Dining",
    164: "Bedding",
    110: "Men's Clothing",
    116: "Women's Clothing",
    71:  "Headphones & Earbuds",
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "experiment.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# OLLAMA HELPERS
# ─────────────────────────────────────────────
def check_ollama_connection():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=6)
        if r.status_code == 200:
            log.info("Ollama reachable at %s", OLLAMA_BASE_URL)
            return True
    except Exception as e:
        log.error("Cannot reach Ollama at %s: %s", OLLAMA_BASE_URL, e)
    return False


def find_installed_llama_model():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=6)
        models = [m["name"] for m in r.json().get("models", [])]
        log.info("Installed models: %s", models)
        for m in models:
            if "llama3.2" in m.lower() or "llama-3.2" in m.lower():
                log.info("Selected model: %s", m)
                return m
        if models:
            log.warning("No llama3.2 found; using: %s", models[0])
            return models[0]
    except Exception as e:
        log.error("Model listing failed: %s", e)
    return ""


def call_ollama(prompt, model):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "seed": RANDOM_SEED},
    }
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                raw = r.json().get("response", "").strip()
                return raw, ""
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.exceptions.Timeout:
            last_err = "Request timed out"
        except Exception as e:
            last_err = str(e)
        log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES + 1, last_err)
        if attempt <= MAX_RETRIES:
            time.sleep(3 * attempt)
    return "", last_err


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    log.info("Loading selected products from %s", SELECTED_CSV)
    sel = pd.read_csv(SELECTED_CSV)
    sel["category_id"]    = sel["category_id"].astype(int)
    sel["boughtInLastMonth"] = pd.to_numeric(sel["boughtInLastMonth"], errors="coerce")
    sel["stars"]          = pd.to_numeric(sel["stars"],   errors="coerce")
    sel["price"]          = pd.to_numeric(sel["price"],   errors="coerce")
    reviews_col = "reviews" if "reviews" in sel.columns else None
    sel["reviews_count"]  = pd.to_numeric(sel[reviews_col], errors="coerce").fillna(0).astype(int) if reviews_col else 0

    log.info("Loading reviews from %s", REVIEWS_JSON)
    with open(REVIEWS_JSON, encoding="utf-8") as f:
        reviews_list = json.load(f)
    reviews_map = {str(p["asin"]).strip(): p for p in reviews_list}

    log.info("Loading category names from %s", CATEGORIES_CSV)
    cats = pd.read_csv(CATEGORIES_CSV)
    cat_name_map = dict(zip(cats["id"].astype(int), cats["category_name"]))
    cat_name_map.update(CATEGORY_MAP)

    sel["category_name"] = sel["category_id"].map(cat_name_map).fillna("Unknown")
    return sel, reviews_map, cat_name_map


def validate_data(sel, reviews_map):
    report = {
        "total_products":         len(sel),
        "missing_demand":         sel[sel["boughtInLastMonth"].isna()]["asin"].tolist(),
        "duplicate_asins":        sel[sel["asin"].duplicated(keep=False)]["asin"].unique().tolist(),
        "unmatched_in_reviews":   [a for a in sel["asin"] if str(a).strip() not in reviews_map],
        "category_distribution":  {int(k): int(v) for k, v in sel.groupby("category_id")["asin"].count().items()},
        "validation_passed":      True,
    }
    dist = report["category_distribution"]
    if len(dist) != 10 or any(v != 10 for v in dist.values()):
        report["validation_passed"] = False
    if report["missing_demand"]:
        report["validation_passed"] = False
    return report


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────
def create_experiment_split(sel, seed):
    rng = random.Random(seed)
    rows = []
    for cat_id, group in sel.groupby("category_id"):
        asins = group["asin"].tolist()
        rng.shuffle(asins)
        ref_set  = set(asins[:5])
        for _, row in group.iterrows():
            rows.append({
                "asin":          row["asin"],
                "title":         row["title"],
                "category_id":   int(cat_id),
                "category_name": row["category_name"],
                "split":         "reference" if row["asin"] in ref_set else "test",
            })
    return pd.DataFrame(rows)


def load_or_create_split(sel, seed):
    split_path = OUTPUT_DIR / "experiment_split.csv"
    if split_path.exists():
        log.info("Loading existing split")
        sp = pd.read_csv(split_path)
        sp["category_id"] = sp["category_id"].astype(int)
        return sp
    sp = create_experiment_split(sel, seed)
    sp.to_csv(split_path, index=False)
    log.info("Split saved to %s", split_path)
    return sp


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
def get_review_snippet(asin, reviews_map):
    p = reviews_map.get(str(asin).strip())
    if not p:
        return "No review text available."
    snippets = []
    for star in ["5", "4", "3"]:
        for rv in p["reviews"].get(star, []):
            body = rv.get("body", "").strip()
            if body:
                snippets.append(f'[{star}*] "{body}"')
            if len(" ".join(snippets)) > MAX_REVIEW_CHARS:
                break
        if len(" ".join(snippets)) > MAX_REVIEW_CHARS:
            break
    text = " ".join(snippets)
    if not text:
        return "No review text available."
    return text[:MAX_REVIEW_CHARS] + "..." if len(text) > MAX_REVIEW_CHARS else text


def format_product_block(row, reviews_map, include_demand):
    parts = [
        f"  Title      : {row.get('title','N/A')}",
        f"  Price      : ${row.get('price','N/A')}",
        f"  Stars      : {row.get('stars','N/A')}",
        f"  Num Reviews: {row.get('reviews_count','N/A')}",
        f"  BestSeller : {'Yes' if row.get('isBestSeller') else 'No'}",
        f"  Reviews    : {get_review_snippet(row['asin'], reviews_map)}",
    ]
    if include_demand:
        parts.append(f"  boughtInLastMonth (known): {int(row['boughtInLastMonth'])}")
    return "\n".join(parts)


def build_prompt(ref_rows, test_row, cat_name, reviews_map):
    ref_blocks = []
    for i, (_, rr) in enumerate(ref_rows.iterrows(), 1):
        ref_blocks.append(
            f"Reference Product {i}:\n"
            + format_product_block(rr, reviews_map, include_demand=True)
        )
    ref_text = "\n\n".join(ref_blocks)
    test_block = format_product_block(test_row, reviews_map, include_demand=False)

    prompt = f"""You are a product demand analyst estimating Amazon product sales.

CONTEXT
=======
"boughtInLastMonth" = approximate number of units purchased on Amazon in the last month.
It is a non-negative integer.

CATEGORY: {cat_name}

REFERENCE PRODUCTS (demand values are known observations)
=========================================================
{ref_text}

TEST PRODUCT (demand is HIDDEN - your task is to estimate it)
=============================================================
{test_block}

TASK
====
Using the reference products as examples for this category, estimate the
boughtInLastMonth value for the test product.

Rules:
- Estimate must be a non-negative integer.
- Base reasoning on price, ratings, review sentiment, and reference patterns.
- Do NOT fabricate product details.

Respond with ONLY valid JSON in this exact format (no text before or after the JSON):
{{
  "predicted_demand": <integer>,
  "reasoning": "<one or two sentences>"
}}"""
    return prompt


# ─────────────────────────────────────────────
# PARSE & VALIDATE
# ─────────────────────────────────────────────
def parse_model_response(raw):
    if not raw:
        return None, "", "Empty response from model"
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        demand = obj.get("predicted_demand")
        reasoning = str(obj.get("reasoning", ""))
        if demand is None:
            return None, reasoning, "Missing predicted_demand"
        demand = float(demand)
        if demand < 0 or not math.isfinite(demand):
            return None, reasoning, f"Invalid demand: {demand}"
        return demand, reasoning, ""
    except json.JSONDecodeError as e:
        return None, "", f"JSON parse error: {e}"


def validate_prediction(val):
    return val is not None and math.isfinite(float(val)) and float(val) >= 0


# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
def run_predictions(sel, split, reviews_map, model):
    ref_asins  = set(split[split["split"] == "reference"]["asin"])
    test_asins = set(split[split["split"] == "test"]["asin"])
    ref_df     = sel[sel["asin"].isin(ref_asins)].copy()
    test_df    = sel[sel["asin"].isin(test_asins)].copy()

    results = []
    audits  = []
    total   = len(test_df)
    done    = 0

    for cat_id, cat_group in test_df.groupby("category_id"):
        cat_name = CATEGORY_MAP.get(cat_id, str(cat_id))
        cat_refs = ref_df[ref_df["category_id"] == cat_id]

        for _, test_row in cat_group.iterrows():
            done += 1
            asin   = test_row["asin"]
            actual = test_row["boughtInLastMonth"]
            log.info("[%d/%d] Predicting ASIN %s — %s", done, total, asin, cat_name)

            prompt = build_prompt(cat_refs, test_row, cat_name, reviews_map)

            # Architectural guarantee: format_product_block is called with
            # include_demand=False for every test row. The only way a leak could
            # occur is if the exact demand label phrase appeared in the test
            # product's own block — which structurally cannot happen.
            # We verify by checking only the test-product section of the prompt.
            test_section_start = prompt.rfind("TEST PRODUCT")
            test_section = prompt[test_section_start:] if test_section_start >= 0 else prompt
            leak_phrase = f"boughtInLastMonth (known): {int(actual)}"
            if leak_phrase in test_section:
                log.error("LABEL LEAK in test section for ASIN %s — skipping", asin)
                results.append({
                    "asin": asin, "title": test_row.get("title",""),
                    "category_id": int(cat_id), "category_name": cat_name,
                    "actual_demand": int(actual), "predicted_demand": None,
                    "absolute_error": None, "squared_error": None,
                    "prediction_status": "failed", "model_name": model,
                    "reasoning": "", "raw_response": "",
                    "error_message": "Label leak in test section — prediction skipped",
                })
                continue

            ts = datetime.now(timezone.utc).isoformat()
            raw, err = call_ollama(prompt, model)
            predicted, reasoning, parse_err = parse_model_response(raw)

            if predicted is None and not err:
                log.warning("Retry for ASIN %s with JSON reminder", asin)
                retry_prompt = (
                    prompt
                    + '\n\nIMPORTANT: Respond with ONLY this JSON and nothing else:\n'
                    '{"predicted_demand": <integer>, "reasoning": "<text>"}\n'
                )
                raw, err = call_ollama(retry_prompt, model)
                predicted, reasoning, parse_err = parse_model_response(raw)

            status  = "success" if validate_prediction(predicted) else "failed"
            abs_err = abs(actual - predicted)  if status == "success" else None
            sq_err  = (actual - predicted)**2  if status == "success" else None

            results.append({
                "asin":             asin,
                "title":            test_row.get("title",""),
                "category_id":      int(cat_id),
                "category_name":    cat_name,
                "actual_demand":    int(actual),
                "predicted_demand": float(predicted) if predicted is not None else None,
                "absolute_error":   abs_err,
                "squared_error":    sq_err,
                "prediction_status":status,
                "model_name":       model,
                "reasoning":        reasoning,
                "raw_response":     raw[:2000],
                "error_message":    err or parse_err,
            })

            audits.append({
                "test_asin":               asin,
                "category":                cat_name,
                "reference_asins":         cat_refs["asin"].tolist(),
                "prompt":                  prompt,
                "model_name":              model,
                "temperature":             TEMPERATURE,
                "random_seed":             RANDOM_SEED,
                "prompt_template_version": PROMPT_VERSION,
                "timestamp":               ts,
                "raw_response":            raw[:2000],
                "parsed_prediction":       predicted,
                "prediction_status":       status,
                "error_message":           err or parse_err,
                "test_target_in_prompt":   False,
            })
            log.info("  status=%s  predicted=%s  actual=%s", status, predicted, int(actual))

    return pd.DataFrame(results), audits


# ─────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────
def calculate_baselines(sel, split):
    ref_df  = sel[sel["asin"].isin(set(split[split["split"]=="reference"]["asin"]))].copy()
    test_df = sel[sel["asin"].isin(set(split[split["split"]=="test"]["asin"]))].copy()
    rows = []
    for cat_id, cat_refs in ref_df.groupby("category_id"):
        ref_mean   = cat_refs["boughtInLastMonth"].mean()
        ref_median = cat_refs["boughtInLastMonth"].median()
        for _, tr in test_df[test_df["category_id"]==cat_id].iterrows():
            actual = tr["boughtInLastMonth"]
            rows.append({
                "asin": tr["asin"], "title": tr["title"],
                "category_id": int(cat_id),
                "category_name": CATEGORY_MAP.get(cat_id, str(cat_id)),
                "actual_demand":      int(actual),
                "baseline_mean_pred": float(ref_mean),
                "baseline_med_pred":  float(ref_median),
                "mae_mean":   abs(actual - ref_mean),
                "mae_median": abs(actual - ref_median),
                "se_mean":    (actual - ref_mean)**2,
                "se_median":  (actual - ref_median)**2,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def calculate_metrics(results, baselines):
    cat_rows = []
    for cat_id, grp in results.groupby("category_id"):
        valid = grp[grp["prediction_status"]=="success"]
        cat_rows.append({
            "category_id":   int(cat_id),
            "category_name": CATEGORY_MAP.get(cat_id, str(cat_id)),
            "n_test":        len(grp),
            "n_success":     len(valid),
            "n_failed":      len(grp)-len(valid),
            "actual_mean":   grp["actual_demand"].mean(),
            "predicted_mean":valid["predicted_demand"].mean() if len(valid) else None,
            "rmse": math.sqrt(valid["squared_error"].mean()) if len(valid) else None,
            "mae":  valid["absolute_error"].mean() if len(valid) else None,
        })
    cat_df = pd.DataFrame(cat_rows)

    valid_all = results[results["prediction_status"]=="success"]
    overall = {
        "model":                  results["model_name"].iloc[0] if len(results) else "",
        "total_test":             len(results),
        "n_success":              len(valid_all),
        "n_failed":               len(results)-len(valid_all),
        "success_rate":           round(len(valid_all)/len(results),4) if len(results) else 0,
        "overall_actual_mean":    float(results["actual_demand"].mean()),
        "overall_predicted_mean": float(valid_all["predicted_demand"].mean()) if len(valid_all) else None,
        "overall_rmse":           float(math.sqrt(valid_all["squared_error"].mean())) if len(valid_all) else None,
        "overall_mae":            float(valid_all["absolute_error"].mean()) if len(valid_all) else None,
        "random_seed":            RANDOM_SEED,
        "temperature":            TEMPERATURE,
        "prompt_version":         PROMPT_VERSION,
        "ollama_base_url":        OLLAMA_BASE_URL,
        "run_timestamp":          datetime.now(timezone.utc).isoformat(),
    }

    bl_cat_rows = []
    for cat_id, grp in baselines.groupby("category_id"):
        bl_cat_rows.append({
            "category_id":   int(cat_id),
            "category_name": CATEGORY_MAP.get(cat_id, str(cat_id)),
            "n_test":        len(grp),
            "rmse_mean":     math.sqrt(grp["se_mean"].mean()),
            "mae_mean":      grp["mae_mean"].mean(),
            "rmse_median":   math.sqrt(grp["se_median"].mean()),
            "mae_median":    grp["mae_median"].mean(),
        })
    bl_df = pd.DataFrame(bl_cat_rows)
    bl_df.loc[len(bl_df)] = {
        "category_id": 0, "category_name": "OVERALL", "n_test": len(baselines),
        "rmse_mean":   math.sqrt(baselines["se_mean"].mean()),
        "mae_mean":    baselines["mae_mean"].mean(),
        "rmse_median": math.sqrt(baselines["se_median"].mean()),
        "mae_median":  baselines["mae_median"].mean(),
    }
    return cat_df, overall, bl_df


# ─────────────────────────────────────────────
# GRAPHS
# ─────────────────────────────────────────────
def safe_fn(name):
    return re.sub(r"[^\w]", "_", name).lower().strip("_")

def shorten(t, n=28):
    return t[:n]+"..." if len(t)>n else t


def generate_category_graphs(results):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for cat_id, grp in results.groupby("category_id"):
        cat_name = CATEGORY_MAP.get(cat_id, str(cat_id))
        grp = grp.reset_index(drop=True)
        labels  = [shorten(t) for t in grp["title"]]
        x       = list(range(len(grp)))
        actual  = grp["actual_demand"].tolist()
        pred    = [r["predicted_demand"] if r["prediction_status"]=="success" else None
                   for _, r in grp.iterrows()]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x, actual, marker="o", linewidth=2, color="#2563EB", label="Actual Demand")

        px = [i for i,v in enumerate(pred) if v is not None]
        pv = [v for v in pred if v is not None]
        if px:
            ax.plot(px, pv, marker="s", linewidth=2, color="#DC2626",
                    linestyle="--", label="Llama 3.2 Predicted")

        fx = [i for i,v in enumerate(pred) if v is None]
        if fx:
            ax.scatter(fx, [actual[i] for i in fx], marker="x",
                       color="gray", s=80, zorder=5, label="Prediction Failed")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Test Product")
        ax.set_ylabel("boughtInLastMonth")
        ax.set_title(f"Demand Estimation — {cat_name}", fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fname = GRAPHS_DIR / f"category_{safe_fn(cat_name)}.png"
        fig.savefig(fname, dpi=150); plt.close(fig)
        generated.append(str(fname))
        log.info("Graph: %s", fname)
    return generated


def generate_overall_graph(results):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    valid = results[results["prediction_status"]=="success"].reset_index(drop=True)
    if valid.empty:
        log.warning("No successful predictions — overall graph skipped")
        return
    labels = [f"{shorten(r['title'],20)}\n({r['category_name'][:10]})" for _,r in valid.iterrows()]
    x = list(range(len(valid)))
    fig, ax = plt.subplots(figsize=(max(14, len(valid)*0.45+2), 6))
    ax.plot(x, valid["actual_demand"],    marker="o", linewidth=2, color="#2563EB", label="Actual")
    ax.plot(x, valid["predicted_demand"], marker="s", linewidth=2, color="#DC2626",
            linestyle="--", label="Llama 3.2 Predicted")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Test Product"); ax.set_ylabel("boughtInLastMonth")
    ax.set_title("Overall Actual vs Predicted — All Categories", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = GRAPHS_DIR / "overall_actual_vs_predicted.png"
    fig.savefig(fname, dpi=150); plt.close(fig)
    log.info("Overall graph: %s", fname)


def generate_optional_graphs(results, cat_df):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    valid = results[results["prediction_status"]=="success"].copy()
    if valid.empty:
        return

    # Scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(valid["actual_demand"], valid["predicted_demand"],
               alpha=0.7, color="#7C3AED", edgecolors="white", s=80)
    mx = max(valid["actual_demand"].max(), valid["predicted_demand"].max()) * 1.1
    ax.plot([0,mx],[0,mx],"k--",linewidth=1,label="y=x (perfect)")
    ax.set_xlabel("Actual Demand"); ax.set_ylabel("Predicted Demand")
    ax.set_title("Actual vs Predicted — Scatter"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(GRAPHS_DIR / "actual_vs_predicted_scatter.png", dpi=150); plt.close(fig)

    # Error distribution
    errors = valid["actual_demand"] - valid["predicted_demand"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=15, color="#059669", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Zero Error")
    ax.set_xlabel("Actual − Predicted"); ax.set_ylabel("Count")
    ax.set_title("Prediction Error Distribution"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(GRAPHS_DIR / "prediction_error_distribution.png", dpi=150); plt.close(fig)

    # Category RMSE bar
    plot_cats = cat_df[cat_df["rmse"].notna()].copy()
    if not plot_cats.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(plot_cats["category_name"], plot_cats["rmse"], color="#F59E0B", edgecolor="white")
        ax.set_xlabel("RMSE"); ax.set_title("Category RMSE — Llama 3.2"); ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(GRAPHS_DIR / "category_rmse_comparison.png", dpi=150); plt.close(fig)
    log.info("Optional graphs saved")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)

    log.info("="*70)
    log.info("AMAZON DEMAND ESTIMATION — LOCAL LLAMA 3.2 VIA OLLAMA")
    log.info("="*70)

    # 1. Ollama check
    if not check_ollama_connection():
        print("\n[ERROR] Ollama is not reachable at", OLLAMA_BASE_URL)
        print("Start it with:  ollama serve")
        return

    global OLLAMA_MODEL
    if not OLLAMA_MODEL:
        OLLAMA_MODEL = find_installed_llama_model()
    if not OLLAMA_MODEL:
        print("[ERROR] No llama3.2 found. Install with:  ollama pull llama3.2")
        return
    log.info("Using model: %s", OLLAMA_MODEL)

    # 2. Load & validate
    sel, reviews_map, _ = load_data()
    val_report = validate_data(sel, reviews_map)
    with open(OUTPUT_DIR/"validation_report.json","w",encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
    if not val_report["validation_passed"]:
        log.warning("Validation issues detected — check validation_report.json")

    # 3. Split
    split = load_or_create_split(sel, RANDOM_SEED)

    # 4. Baselines
    baselines = calculate_baselines(sel, split)
    baselines.to_csv(OUTPUT_DIR/"baseline_predictions.csv", index=False)

    # 5. LLM predictions
    results, audits = run_predictions(sel, split, reviews_map, OLLAMA_MODEL)
    results.to_csv(OUTPUT_DIR/"demand_estimation_predictions.csv", index=False)
    with open(OUTPUT_DIR/"prompt_audit.json","w",encoding="utf-8") as f:
        json.dump(audits, f, indent=2, ensure_ascii=False)

    # 6. Metrics
    cat_df, overall, bl_df = calculate_metrics(results, baselines)
    cat_df.to_csv(OUTPUT_DIR/"category_rmse.csv", index=False)
    bl_df.to_csv(OUTPUT_DIR/"baseline_metrics.csv", index=False)
    with open(OUTPUT_DIR/"overall_metrics.json","w",encoding="utf-8") as f:
        json.dump(overall, f, indent=2, default=str)

    # 7. Graphs
    cat_graphs = generate_category_graphs(results)
    generate_overall_graph(results)
    generate_optional_graphs(results, cat_df)

    # 8. Final report
    valid_all = results[results["prediction_status"]=="success"]
    bl_row = bl_df[bl_df["category_name"]=="OVERALL"].iloc[0]
    print("\n" + "="*70)
    print("FINAL EXPERIMENT REPORT")
    print("="*70)
    print(f"  Model                   : {OLLAMA_MODEL}")
    print(f"  Endpoint                : {OLLAMA_BASE_URL}  (LOCAL, no cloud)")
    print(f"  Categories processed    : {results['category_id'].nunique()}")
    print(f"  Reference products      : {(split['split']=='reference').sum()}")
    print(f"  Test products           : {(split['split']=='test').sum()}")
    print(f"  Successful predictions  : {len(valid_all)}")
    print(f"  Failed predictions      : {len(results)-len(valid_all)}")
    print(f"  Overall RMSE (LLM)      : {overall.get('overall_rmse','N/A')}")
    print(f"  Overall MAE  (LLM)      : {overall.get('overall_mae','N/A')}")
    print(f"  Baseline RMSE (mean)    : {bl_row['rmse_mean']:.2f}")
    print(f"  Baseline MAE  (mean)    : {bl_row['mae_mean']:.2f}")
    print(f"  Baseline RMSE (median)  : {bl_row['rmse_median']:.2f}")
    print(f"  Baseline MAE  (median)  : {bl_row['mae_median']:.2f}")
    print()
    print("  Category-wise RMSE:")
    for _, r in cat_df.iterrows():
        rmse_str = f"{r['rmse']:.2f}" if pd.notna(r['rmse']) else 'N/A'
        print(f"    {r['category_name']:<34}: {rmse_str}")
    print()
    unmatched = val_report.get("unmatched_in_reviews",[])
    if unmatched:
        print(f"  [WARNING] {len(unmatched)} ASINs not in reviews JSON")
    print("  Output files saved in:", OUTPUT_DIR)
    print("  Graphs saved in:", GRAPHS_DIR)
    print("="*70)


if __name__ == "__main__":
    main()
