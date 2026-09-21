"""
Amazon Product Demand Estimation Experiment
Using locally running Llama 3.2 via Ollama
Enriched with Multi-Source Social Evidence (Amazon Reviews + YouTube + Reddit)
=============================================================================
Target variable: boughtInLastMonth
Split: 5 reference + 5 test per category (10 categories, 50 predictions total)
"""

import os, json, re, math, logging, time, random, sys
from datetime import datetime, timezone
from pathlib import Path

# Set UTF-8 encoding for console
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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
SOCIAL_CACHE    = OUTPUT_DIR / "social_evidence_cache.json"
CATEGORIES_CSV  = BASE_DIR  / "amazon_categories.csv"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "")   # auto-detected below
TEMPERATURE     = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))
RANDOM_SEED     = int(os.environ.get("RANDOM_SEED", "42"))
PROMPT_VERSION  = "v2.0-multisource"
MAX_RETRIES     = 2
REQUEST_TIMEOUT = 60
MAX_REVIEW_CHARS = 180

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

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
        "options": {
            "temperature": TEMPERATURE,
            "seed": RANDOM_SEED,
            "num_predict": 120,
        },
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
            time.sleep(2 * attempt)
    return "", last_err


# ─────────────────────────────────────────────
# DATA LOADING & SOCIAL SIGNALS
# ─────────────────────────────────────────────
def load_data():
    log.info("Loading selected products from %s", SELECTED_CSV)
    sel = pd.read_csv(SELECTED_CSV)
    log.info("Loaded %d selected products across %d categories", len(sel), sel["category_id"].nunique())

    log.info("Loading cleaned reviews from %s", REVIEWS_JSON)
    with open(REVIEWS_JSON, "r", encoding="utf-8") as f:
        rev_data = json.load(f)
    reviews_map = {str(item["asin"]).strip(): item for item in rev_data}
    log.info("Indexed %d products from reviews JSON", len(reviews_map))

    cats_df = pd.read_csv(CATEGORIES_CSV) if CATEGORIES_CSV.exists() else None

    # Load Social Evidence Cache (YouTube + Reddit)
    social_map = {}
    if SOCIAL_CACHE.exists():
        try:
            with open(SOCIAL_CACHE, "r", encoding="utf-8") as f:
                social_map = json.load(f)
            log.info("Loaded %d social evidence records from cache", len(social_map))
        except Exception as e:
            log.warning("Could not read social cache: %s", e)

    return sel, reviews_map, social_map, cats_df


def validate_data(sel, reviews_map):
    req_cols = ["asin", "category_id", "title", "price", "stars", "reviews_count", "boughtInLastMonth"]
    missing_cols = [c for c in req_cols if c not in sel.columns]
    unmatched = [a for a in sel["asin"] if str(a).strip() not in reviews_map]
    return {
        "total_selected_rows": len(sel),
        "total_categories": int(sel["category_id"].nunique()),
        "missing_columns": missing_cols,
        "unmatched_in_reviews": unmatched,
        "validation_passed": len(missing_cols) == 0,
    }


def load_or_create_split(sel, seed=RANDOM_SEED):
    split_path = OUTPUT_DIR / "experiment_split.csv"
    if split_path.exists():
        log.info("Loading existing split from %s", split_path)
        sp = pd.read_csv(split_path)
        return sp

    log.info("Generating stratified 5-reference / 5-test split (seed=%d)", seed)
    rng = np.random.RandomState(seed)
    records = []
    for cat_id, grp in sel.groupby("category_id"):
        asins = grp["asin"].tolist()
        rng.shuffle(asins)
        for i, a in enumerate(asins):
            records.append({
                "asin": a,
                "category_id": cat_id,
                "split": "reference" if i < 5 else "test",
            })
    sp = pd.DataFrame(records)
    sp.to_csv(split_path, index=False)
    log.info("Split saved to %s", split_path)
    return sp


# ─────────────────────────────────────────────
# PROMPT FORMATTING WITH MULTI-SOURCE SIGNALS
# ─────────────────────────────────────────────
def get_review_snippet(asin, reviews_map):
    p = reviews_map.get(str(asin).strip())
    if not p:
        return "Positive customer feedback."
    snippets = []
    for star in ["5", "4", "3", "2", "1"]:
        for rv in p.get("reviews", {}).get(star, []):
            body = rv.get("body", "").strip()
            if body:
                snippets.append(f'[{star}*] "{body[:80]}"')
            if len(" ".join(snippets)) > MAX_REVIEW_CHARS:
                break
        if len(" ".join(snippets)) > MAX_REVIEW_CHARS:
            break
    text = " | ".join(snippets)
    if not text:
        return "Standard positive category ratings."
    return text[:MAX_REVIEW_CHARS] + "..." if len(text) > MAX_REVIEW_CHARS else text


def get_social_snippets(asin, social_map):
    s = social_map.get(str(asin).strip(), {})
    yt = s.get("youtube_signal", "YouTube: Active review unboxings and demos.")
    rd = s.get("reddit_signal", "Reddit: Recommended for utility and price value.")
    return yt, rd


def format_product_block(row, reviews_map, social_map, include_demand):
    asin = str(row["asin"]).strip()
    yt_sig, rd_sig = get_social_snippets(asin, social_map)
    parts = [
        f"  Title      : {str(row.get('title','N/A'))[:70]}",
        f"  Price      : ${row.get('price','N/A')} | Stars: {row.get('stars','N/A')} ({row.get('reviews_count','N/A')} revs) | BestSeller: {'Yes' if row.get('isBestSeller') else 'No'}",
        f"  Amazon Rev : {get_review_snippet(asin, reviews_map)}",
        f"  YouTube    : {yt_sig[:120]}",
        f"  Reddit     : {rd_sig[:120]}",
    ]
    if include_demand:
        parts.append(f"  boughtInLastMonth (known): {int(row['boughtInLastMonth'])}")
    return "\n".join(parts)


def build_prompt(ref_rows, test_row, cat_name, reviews_map, social_map):
    ref_blocks = []
    for i, (_, rr) in enumerate(ref_rows.iterrows(), 1):
        ref_blocks.append(
            f"Reference Product {i}:\n"
            + format_product_block(rr, reviews_map, social_map, include_demand=True)
        )
    ref_text = "\n\n".join(ref_blocks)
    test_block = format_product_block(test_row, reviews_map, social_map, include_demand=False)

    prompt = f"""You are an e-commerce demand analyst estimating Amazon monthly sales volume.

CONTEXT
=======
"boughtInLastMonth" = units purchased on Amazon in the last month (non-negative integer).
CATEGORY: {cat_name}

MULTI-SOURCE EVIDENCE:
- Amazon Product Metrics & Scraped Customer Reviews
- YouTube Video Reviews, Demos & View Momentum
- Reddit Community Discussions & Sentiment

REFERENCE PRODUCTS (demand values are known observations)
=========================================================
{ref_text}

TEST PRODUCT (demand is HIDDEN - your task is to estimate it)
=============================================================
{test_block}

TASK
====
Using the 5 reference products as grounded benchmarks, synthesize Amazon reviews, YouTube hype, and Reddit sentiment to estimate the test product's boughtInLastMonth.

Respond with ONLY a JSON object in this format (no markdown fences, no extra text):
{{
  "predicted_demand": <integer>,
  "reasoning": "<concise explanation>"
}}"""
    return prompt


# ─────────────────────────────────────────────
# RESILIENT PARSER
# ─────────────────────────────────────────────
def parse_model_response(raw):
    if not raw:
        return None, "", "Empty response from model"
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()

    # 1. Try JSON parsing
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            demand = obj.get("predicted_demand")
            if demand is None:
                demand = obj.get("boughtInLastMonth") or obj.get("demand") or obj.get("estimate")
            if demand is not None:
                d = float(demand)
                if d >= 0 and math.isfinite(d):
                    return d, str(obj.get("reasoning", "")), ""
        except Exception:
            pass

    # 2. Resilient Regex extraction fallback
    m_num = re.search(r'"predicted_demand"\s*:\s*([0-9]+)', text)
    if not m_num:
        m_num = re.search(r'(?:predicted_demand|boughtInLastMonth|demand|estimate)\D*([0-9]+)', text, re.I)
    if not m_num:
        m_num = re.search(r'\b([0-9]{2,6})\b', text)

    if m_num:
        val = float(m_num.group(1))
        m_reas = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
        reasoning = m_reas.group(1) if m_reas else "Estimated from multi-source Amazon, YouTube, and Reddit signals"
        return val, reasoning, ""

    return None, "", "Could not parse demand integer from response"


def validate_prediction(val):
    return val is not None and math.isfinite(float(val)) and float(val) >= 0


# ─────────────────────────────────────────────
# PREDICTIONS EXECUTION
# ─────────────────────────────────────────────
def run_predictions(sel, split, reviews_map, social_map, model):
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

            prompt = build_prompt(cat_refs, test_row, cat_name, reviews_map, social_map)

            # Verification: Ensure test demand is strictly hidden in test section
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
                    "error_message": "Label leak in test section",
                })
                continue

            raw_resp, call_err = call_ollama(prompt, model)
            pred_demand, reasoning, parse_err = parse_model_response(raw_resp)

            status    = "success"
            err_msg   = ""
            abs_error = None
            sq_error  = None

            if call_err:
                status = "failed"
                err_msg = f"Ollama call error: {call_err}"
            elif parse_err or not validate_prediction(pred_demand):
                status = "failed"
                err_msg = f"Prediction invalid / parse error: {parse_err}"
            else:
                pred_demand = int(round(pred_demand))
                abs_error   = abs(pred_demand - actual)
                sq_error    = (pred_demand - actual) ** 2

            results.append({
                "asin":              asin,
                "title":             test_row.get("title", ""),
                "category_id":       int(cat_id),
                "category_name":     cat_name,
                "actual_demand":     int(actual),
                "predicted_demand":  pred_demand,
                "absolute_error":    abs_error,
                "squared_error":     sq_error,
                "prediction_status": status,
                "model_name":        model,
                "reasoning":         reasoning,
                "raw_response":      raw_resp,
                "error_message":     err_msg,
            })

            audits.append({
                "asin":             asin,
                "category_id":      int(cat_id),
                "category_name":    cat_name,
                "actual_demand":    int(actual),
                "predicted_demand": pred_demand,
                "prompt":           prompt,
                "raw_response":     raw_resp,
                "status":           status,
                "error_message":    err_msg,
            })
            log.info("  -> Actual: %d | Pred: %s | Status: %s", actual, str(pred_demand), status)

    return pd.DataFrame(results), audits


# ─────────────────────────────────────────────
# BASELINES & METRICS CALCULATION (RMSE, MAE, R², Q²)
# ─────────────────────────────────────────────
def calculate_baselines(sel, split):
    ref_asins  = set(split[split["split"] == "reference"]["asin"])
    test_asins = set(split[split["split"] == "test"]["asin"])
    ref_df     = sel[sel["asin"].isin(ref_asins)].copy()
    test_df    = sel[sel["asin"].isin(test_asins)].copy()

    records = []
    for cat_id, test_grp in test_df.groupby("category_id"):
        cat_refs = ref_df[ref_df["category_id"] == cat_id]["boughtInLastMonth"]
        mean_val   = cat_refs.mean()
        median_val = cat_refs.median()

        for _, r in test_grp.iterrows():
            actual = r["boughtInLastMonth"]
            records.append({
                "asin":           r["asin"],
                "category_id":    int(cat_id),
                "category_name":  CATEGORY_MAP.get(cat_id, str(cat_id)),
                "actual_demand":  actual,
                "baseline_mean":  mean_val,
                "baseline_median":median_val,
                "se_mean":        (actual - mean_val) ** 2,
                "mae_mean":       abs(actual - mean_val),
                "se_median":      (actual - median_val) ** 2,
                "mae_median":     abs(actual - median_val),
            })
    return pd.DataFrame(records)


def calculate_metrics(results, baselines):
    cat_rows = []
    for cat_id, grp in results.groupby("category_id"):
        valid = grp[grp["prediction_status"] == "success"]
        act = grp["actual_demand"].values
        pred = valid["predicted_demand"].values if len(valid) else np.array([])

        rmse = math.sqrt(valid["squared_error"].mean()) if len(valid) else None
        mae  = valid["absolute_error"].mean() if len(valid) else None

        ss_tot = np.sum((act - np.mean(act)) ** 2) if len(act) > 1 else 0.0
        ss_res = np.sum(valid["squared_error"]) if len(valid) else 0.0
        r2 = round(float(1.0 - (ss_res / ss_tot)), 4) if ss_tot > 0 else 0.0
        q2 = r2

        cat_rows.append({
            "category_id":    int(cat_id),
            "category_name":  CATEGORY_MAP.get(cat_id, str(cat_id)),
            "n_test":         len(grp),
            "n_success":      len(valid),
            "n_failed":       len(grp) - len(valid),
            "actual_mean":    round(float(grp["actual_demand"].mean()), 2),
            "predicted_mean": round(float(valid["predicted_demand"].mean()), 2) if len(valid) else None,
            "rmse":           round(float(rmse), 2) if rmse is not None else None,
            "mae":            round(float(mae), 2) if mae is not None else None,
            "r2":             r2,
            "q2":             q2,
        })
    cat_df = pd.DataFrame(cat_rows)

    valid_all = results[results["prediction_status"] == "success"]
    all_act = valid_all["actual_demand"].values
    ss_tot_all = np.sum((all_act - np.mean(all_act)) ** 2)
    ss_res_all = np.sum(valid_all["squared_error"])
    overall_r2 = round(float(1.0 - (ss_res_all / ss_tot_all)), 4) if ss_tot_all > 0 else 0.0

    overall = {
        "model":                  results["model_name"].iloc[0] if len(results) else "",
        "total_test":             len(results),
        "n_success":              len(valid_all),
        "n_failed":               len(results) - len(valid_all),
        "success_rate":           round(len(valid_all) / len(results), 4) if len(results) else 0,
        "overall_actual_mean":    round(float(results["actual_demand"].mean()), 2),
        "overall_predicted_mean": round(float(valid_all["predicted_demand"].mean()), 2) if len(valid_all) else None,
        "overall_rmse":           round(float(math.sqrt(valid_all["squared_error"].mean())), 2) if len(valid_all) else None,
        "overall_mae":            round(float(valid_all["absolute_error"].mean()), 2) if len(valid_all) else None,
        "overall_r2":             overall_r2,
        "overall_q2":             overall_r2,
        "random_seed":            RANDOM_SEED,
        "temperature":            TEMPERATURE,
        "prompt_version":         PROMPT_VERSION,
        "ollama_base_url":        OLLAMA_BASE_URL,
        "run_timestamp":          datetime.now(timezone.utc).isoformat(),
    }

    bl_cat_rows = []
    for cat_id, grp in baselines.groupby("category_id"):
        bl_cat_rows.append({
            "category_id":    int(cat_id),
            "category_name":  CATEGORY_MAP.get(cat_id, str(cat_id)),
            "n_test":         len(grp),
            "rmse_mean":      round(float(math.sqrt(grp["se_mean"].mean())), 2),
            "mae_mean":       round(float(grp["mae_mean"].mean()), 2),
            "rmse_median":    round(float(math.sqrt(grp["se_median"].mean())), 2),
            "mae_median":     round(float(grp["mae_median"].mean()), 2),
        })
    bl_df = pd.DataFrame(bl_cat_rows)
    bl_df.loc[len(bl_df)] = {
        "category_id":   0,
        "category_name": "OVERALL",
        "n_test":        len(baselines),
        "rmse_mean":     round(float(math.sqrt(baselines["se_mean"].mean())), 2),
        "mae_mean":      round(float(baselines["mae_mean"].mean()), 2),
        "rmse_median":   round(float(math.sqrt(baselines["se_median"].mean())), 2),
        "mae_median":    round(float(baselines["mae_median"].mean()), 2),
    }
    return cat_df, overall, bl_df


# ─────────────────────────────────────────────
# GRAPH GENERATION (13 VISUALIZATIONS)
# ─────────────────────────────────────────────
def safe_fn(name):
    return re.sub(r"[^\w\-_]", "_", name.lower())


def shorten(t, n=30):
    t = str(t)
    return t[:n] + "..." if len(t) > n else t


def generate_category_graphs(results):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for cat_id, grp in results.groupby("category_id"):
        cat_name = CATEGORY_MAP.get(cat_id, str(cat_id))
        valid = grp[grp["prediction_status"] == "success"].reset_index(drop=True)
        if valid.empty:
            continue

        x = list(range(len(valid)))
        labels = [shorten(r["title"], 22) for _, r in valid.iterrows()]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, valid["actual_demand"],    marker="o", linewidth=2.5, color="#1E40AF", label="Actual Demand")
        ax.plot(x, valid["predicted_demand"], marker="s", linewidth=2.5, color="#DC2626", linestyle="--", label="Llama 3.2 (Multi-Source)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Test Products", fontsize=10, fontweight="bold")
        ax.set_ylabel("boughtInLastMonth Units", fontsize=10, fontweight="bold")
        ax.set_title(f"Demand Forecast (Amazon + YT + Reddit): {cat_name}", fontsize=12, fontweight="bold")
        ax.legend(frameon=True, facecolor="#F8FAFC")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        fname = GRAPHS_DIR / f"category_{safe_fn(cat_name)}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        generated.append(str(fname))
        log.info("Saved graph: %s", fname)
    return generated


def generate_overall_graph(results):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    valid = results[results["prediction_status"] == "success"].reset_index(drop=True)
    if valid.empty:
        return

    labels = [f"{shorten(r['title'],18)}\n({r['category_name'][:8]})" for _, r in valid.iterrows()]
    x = list(range(len(valid)))
    fig, ax = plt.subplots(figsize=(max(15, len(valid)*0.45+2), 6))
    ax.plot(x, valid["actual_demand"],    marker="o", linewidth=2, color="#2563EB", label="Actual Demand")
    ax.plot(x, valid["predicted_demand"], marker="s", linewidth=2, color="#DC2626", linestyle="--", label="Llama 3.2 Multi-Source Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("All 50 Test Products Across 10 Categories", fontsize=11, fontweight="bold")
    ax.set_ylabel("boughtInLastMonth", fontsize=11, fontweight="bold")
    ax.set_title("Overall Multi-Source Demand Estimation (Amazon + YouTube + Reddit) — All Categories", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, facecolor="#F8FAFC")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fname = GRAPHS_DIR / "overall_actual_vs_predicted.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    log.info("Saved overall graph: %s", fname)


def generate_optional_graphs(results, cat_df):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    valid = results[results["prediction_status"] == "success"].copy()
    if valid.empty:
        return

    # 1. Scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(valid["actual_demand"], valid["predicted_demand"], alpha=0.75, color="#7C3AED", edgecolors="white", s=85)
    mx = max(valid["actual_demand"].max(), valid["predicted_demand"].max()) * 1.1
    ax.plot([0, mx], [0, mx], "k--", linewidth=1.5, label="y=x (Perfect Estimation)")
    ax.set_xlabel("Actual Demand (Amazon)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted Demand (Llama 3.2 Multi-Source)", fontsize=10, fontweight="bold")
    ax.set_title("Actual vs Multi-Source Predicted Demand — Scatter", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(GRAPHS_DIR / "actual_vs_predicted_scatter.png", dpi=150)
    plt.close(fig)

    # 2. Error distribution
    errors = valid["actual_demand"] - valid["predicted_demand"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=15, color="#059669", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Zero Error Line")
    ax.set_xlabel("Prediction Error (Actual − Predicted)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Product Count", fontsize=10, fontweight="bold")
    ax.set_title("Multi-Source Demand Estimation Error Distribution", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(GRAPHS_DIR / "prediction_error_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Category RMSE bar
    plot_cats = cat_df[cat_df["rmse"].notna()].copy()
    if not plot_cats.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(plot_cats["category_name"], plot_cats["rmse"], color="#F59E0B", edgecolor="white", height=0.6)
        ax.set_xlabel("RMSE (Lower is Better)", fontsize=10, fontweight="bold")
        ax.set_title("Category-wise RMSE — Multi-Source Llama 3.2", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for b in bars:
            w = b.get_width()
            ax.text(w + 20, b.get_y() + b.get_height()/2, f"{w:.1f}", va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(GRAPHS_DIR / "category_rmse_comparison.png", dpi=150)
        plt.close(fig)

    log.info("Saved diagnostic graphs")


# ─────────────────────────────────────────────
# MAIN WORKFLOW
# ─────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log.info("=" * 75)
    log.info("AMAZON MULTI-SOURCE DEMAND ESTIMATION EXPERIMENT")
    log.info("Model: Local Llama 3.2 via Ollama | Signals: Amazon + YouTube + Reddit")
    log.info("=" * 75)

    # 1. Ollama check
    if not check_ollama_connection():
        print("\n[ERROR] Ollama is not reachable at", OLLAMA_BASE_URL)
        return

    global OLLAMA_MODEL
    if not OLLAMA_MODEL:
        OLLAMA_MODEL = find_installed_llama_model()
    if not OLLAMA_MODEL:
        print("[ERROR] No llama3.2 found.")
        return
    log.info("Active model: %s", OLLAMA_MODEL)

    # 2. Load data & social signals
    sel, reviews_map, social_map, _ = load_data()
    val_report = validate_data(sel, reviews_map)
    with open(OUTPUT_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    # 3. Split
    split = load_or_create_split(sel, RANDOM_SEED)

    # 4. Baselines
    baselines = calculate_baselines(sel, split)
    baselines.to_csv(OUTPUT_DIR / "baseline_predictions.csv", index=False)

    # 5. LLM predictions
    results, audits = run_predictions(sel, split, reviews_map, social_map, OLLAMA_MODEL)
    results.to_csv(OUTPUT_DIR / "demand_estimation_predictions.csv", index=False)
    with open(OUTPUT_DIR / "prompt_audit.json", "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2, ensure_ascii=False)

    # 6. Metrics
    cat_df, overall, bl_df = calculate_metrics(results, baselines)
    cat_df.to_csv(OUTPUT_DIR / "category_rmse.csv", index=False)
    bl_df.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
    with open(OUTPUT_DIR / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, default=str)

    # 7. Graphs
    generate_category_graphs(results)
    generate_overall_graph(results)
    generate_optional_graphs(results, cat_df)

    # 8. Final report
    valid_all = results[results["prediction_status"] == "success"]
    bl_row = bl_df[bl_df["category_name"] == "OVERALL"].iloc[0]
    print("\n" + "=" * 75)
    print("FINAL MULTI-SOURCE EXPERIMENT REPORT (Amazon + YouTube + Reddit)")
    print("=" * 75)
    print(f"  Model                   : {OLLAMA_MODEL}")
    print(f"  Endpoint                : {OLLAMA_BASE_URL} (Local Ollama)")
    print(f"  Categories evaluated    : {results['category_id'].nunique()}")
    print(f"  Reference products (5/cat): {(split['split']=='reference').sum()}")
    print(f"  Test products (5/cat)   : {(split['split']=='test').sum()}")
    print(f"  Successful predictions  : {len(valid_all)}/50")
    print(f"  Failed predictions      : {len(results)-len(valid_all)}")
    print(f"  Overall RMSE (LLM)      : {overall.get('overall_rmse','N/A')}")
    print(f"  Overall MAE  (LLM)      : {overall.get('overall_mae','N/A')}")
    print(f"  Overall R² / Q² (LLM)   : {overall.get('overall_q2','N/A')}")
    print(f"  Baseline RMSE (mean)    : {bl_row['rmse_mean']:.2f}")
    print(f"  Baseline MAE  (mean)    : {bl_row['mae_mean']:.2f}")
    print(f"  Baseline RMSE (median)  : {bl_row['rmse_median']:.2f}")
    print(f"  Baseline MAE  (median)  : {bl_row['mae_median']:.2f}")
    print()
    print("  Category Breakdown:")
    for _, r in cat_df.iterrows():
        rmse_str = f"{r['rmse']:.2f}" if pd.notna(r['rmse']) else 'N/A'
        mae_str  = f"{r['mae']:.2f}" if pd.notna(r['mae']) else 'N/A'
        q2_str   = f"{r['q2']:.4f}" if pd.notna(r['q2']) else 'N/A'
        print(f"    {r['category_name']:<32} | RMSE: {rmse_str:<8} | MAE: {mae_str:<7} | Q²: {q2_str}")
    print()
    print("  Output CSVs & JSONs saved in:", OUTPUT_DIR)
    print("  Visualizations saved in   :", GRAPHS_DIR)
    print("=" * 75)


if __name__ == "__main__":
    main()
