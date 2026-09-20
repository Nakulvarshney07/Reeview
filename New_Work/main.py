import sys
import argparse
from pathlib import Path

# Ensure New_work root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.runner import DemandForecastingPipeline
from src.config import DEFAULT_LIMIT, MAX_LIMIT

def parse_args():
    parser = argparse.ArgumentParser(
        description="E-Commerce Demand Forecasting & Inventory Recommendation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of products to process from CSV (Max: {MAX_LIMIT})"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Row index offset to start processing from (0-indexed)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validation & Safety Capping
    limit = args.limit
    if limit > MAX_LIMIT:
        print(f"[CLI] Warning: Requested limit ({limit}) exceeds max allowed ({MAX_LIMIT}). Automatically capping to {MAX_LIMIT}.")
        limit = MAX_LIMIT
    elif limit < 1:
        print(f"[CLI] Error: Limit must be at least 1.")
        sys.exit(1)

    start = max(0, args.start)

    pipeline = DemandForecastingPipeline()
    pipeline.run(limit=limit, start=start)

if __name__ == "__main__":
    main()
