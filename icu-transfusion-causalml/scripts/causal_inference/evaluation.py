from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.pipeline_common import causal_dir, counterfactual_dir, evaluation_dir, figures_dir, load_config, primary_outcome, safe_read_table, save_table  # noqa: E402


def run_evaluation(config) -> None:
    eval_dir = evaluation_dir(config)
    fig_dir = figures_dir(config)
    cf_dir = counterfactual_dir(config)
    c_dir = causal_dir(config)

    predictive = safe_read_table(cf_dir / "counterfactual_evaluation.csv")
    if predictive.empty:
        predictive = safe_read_table(cf_dir / "model_metrics.csv")
    save_table(predictive, eval_dir / "predictive_metrics.csv")

    causal_frames = []
    for path in [c_dir / "effect_estimates_psm.csv", c_dir / "cate_estimates.csv", c_dir / "effect_by_phenotype.csv"]:
        df = safe_read_table(path)
        if not df.empty:
            df["source_file"] = path.name
            causal_frames.append(df)
    causal_metrics = pd.concat(causal_frames, ignore_index=True, sort=False) if causal_frames else pd.DataFrame()
    save_table(causal_metrics, eval_dir / "causal_metrics.csv")

    pred_path = cf_dir / "predicted_y0_y1.parquet"
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        outcome = primary_outcome(config)
        bins = pd.qcut(pred["observed_prediction"], q=min(10, pred["observed_prediction"].nunique()), duplicates="drop")
        calib = pred.groupby(bins, observed=True).agg(predicted=("observed_prediction", "mean"), observed=(outcome, "mean"), n=(outcome, "count")).reset_index(drop=True)
        plt.figure(figsize=(5, 5))
        plt.plot(calib["predicted"], calib["observed"], marker="o")
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        plt.xlabel("Predicted risk")
        plt.ylabel("Observed risk")
        plt.title("Calibration Curve")
        plt.tight_layout()
        plt.savefig(fig_dir / "calibration_curve.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_evaluation(config)
    print("evaluation_complete")


if __name__ == "__main__":
    main()
