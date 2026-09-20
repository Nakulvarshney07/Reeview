# CSV-Based E-Commerce Demand Forecasting System

An isolated, modular, multi-source demand estimation and listing quantity recommendation engine for e-commerce products (Amazon / Flipkart).

---

## 1. Project Purpose

This system takes a historical Amazon product catalog CSV and processes a configurable batch of products to estimate market demand and recommend an initial inventory listing quantity range for an e-commerce seller.

It synthesizes:
- **Historical Amazon Market Benchmarks** (price, stars, reviews, `boughtInLastMonth` sales signal, bestseller status, category ID lookup).
- **Current YouTube Engagement** (Searches and ranks top-5 most-viewed relevant videos + caption transcript retrieval via `youtube-transcript-api`).
- **Current Reddit Community Discussions** (Pulls real customer discussions, complaints, value perception, and product comparisons).
- **Local Ollama Reasoning Engine** (Synthesizes signals with `llama3.2` or deterministic statistical fallback).

> **Note**: This is a **demand-estimation and listing quantity recommendation system**, not a guaranteed sales prediction or maximum-profit optimization tool (since wholesale product cost/shipping/storage fees are not in the raw product catalog).

---

## 2. Directory Architecture

```
New_work/
├── amazon_products.csv         # Raw Amazon product dataset (~1.4M rows)
├── amazon_categories.csv       # Category ID lookup
├── main.py                     # CLI entrypoint
├── requirements.txt            # Dependencies
├── .env.example                # Environment variable configuration
├── README.md                   # System documentation
│
├── src/
│   ├── config.py               # Centralized configuration & directory paths
│   ├── csv_loader/             # Streaming, memory-efficient CSV row reader
│   ├── product_parser/         # Brand/model/query normalization
│   ├── amazon/                 # Historical Amazon benchmark evaluation
│   ├── youtube/                # YouTube search and transcript extraction
│   ├── reddit/                 # Reddit discussion and comment scraper
│   ├── reviews/                # CSV review & rating health analyzer
│   ├── ollama/                 # Centralized Ollama client & prompt templates
│   ├── forecasting/            # Structured evidence aggregator & demand estimator
│   └── pipeline/               # End-to-end orchestrator & CSV/JSON exporter
│
├── cache/                      # Persistent disk cache
│   ├── youtube/
│   ├── transcripts/
│   ├── reddit/
│   └── ollama/
│
└── output/                     # Generated results
    ├── forecast_results.csv    # Final tabular forecast CSV
    └── forecast_results.json   # Full structured evidence JSON
```

---

## 3. Installation & Setup

1. **Install Dependencies**:
   ```bash
   cd New_work
   pip install -r requirements.txt
   ```

2. **Configure Ollama (Optional Local LLM)**:
   Ensure [Ollama](https://ollama.com/) is installed and running with `llama3.2`:
   ```bash
   ollama pull llama3.2
   ```
   Set environment variables in `.env` if using a custom host:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```
   *(If Ollama is not running, the system automatically falls back to grounded statistical estimation without failing).*

---

## 4. Running the System

### A. Development / Sample Mode (1 product)
```bash
python main.py --limit 1 --start 0
```

### B. Standard Batch (10 products)
```bash
python main.py --limit 10
```

### C. Specific Offset (e.g. Products 100 to 109)
```bash
python main.py --limit 10 --start 100
```

### D. Maximum Batch (100 products max)
```bash
python main.py --limit 100
```

---

## 5. Output Files

All outputs are saved to `New_work/output/`:
- **`output/forecast_results.csv`**: Tabular CSV report containing product details, market benchmarks, YouTube/Reddit view counts and sentiment, demand direction, listing quantity range (`estimated_demand_low`, `estimated_demand`, `estimated_demand_high`, `recommended_quantity`), confidence score, and grounded reasoning.
- **`output/forecast_results.json`**: Complete structured evidence and qualitative factor breakdown for every processed product.

---

## 6. Fault Tolerance & Traceability

- **No Fabrication**: If external sources (transcripts, Reddit threads) are missing or blocked, the system sets `available: false` and adjusts the forecast range and confidence accordingly.
- **Explainable Reasoning**: Every recommendation cites specific quantitative and qualitative evidence collected across Amazon, YouTube, Reddit, and customer review metrics.
