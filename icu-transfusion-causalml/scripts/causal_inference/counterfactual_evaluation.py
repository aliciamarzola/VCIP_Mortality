from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.causal_utils import simple_predictive_metrics  # noqa: E402
from scripts.common.pipeline_common import counterfactual_dir, load_config, primary_outcome, save_table  # noqa: E402


def evaluate_counterfactual(config) -> None:
    out_dir = counterfactual_dir(config)
    outcome = primary_outcome(config)
    pred = pd.read_parquet(out_dir / "predicted_y0_y1.parquet")
    rows = []
    for label, sub in [("all", pred), ("test", pred[pred["test_split"] == 1])]:
        if sub.empty:
            continue
        metrics = simple_predictive_metrics(sub[outcome].to_numpy(), sub["observed_prediction"].to_numpy())
        metrics["split"] = label
        metrics["n"] = int(len(sub))
        rows.append(metrics)
    save_table(pd.DataFrame(rows), out_dir / "counterfactual_evaluation.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual predictions.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    evaluate_counterfactual(config)
    print("counterfactual_evaluation_complete")


if __name__ == "__main__":
    main()
