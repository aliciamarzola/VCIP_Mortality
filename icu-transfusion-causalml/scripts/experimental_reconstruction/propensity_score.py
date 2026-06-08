from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from scripts.common.causal_utils import compute_weights, covariate_columns, fit_propensity, trim_overlap
from scripts.common.pipeline_common import causal_dir, figures_dir, load_feature_metadata, load_config, load_processed_frame, primary_outcome, save_table


def run_propensity(config) -> None:
    out_dir = causal_dir(config)
    fig_dir = figures_dir(config)
    df = load_processed_frame(config, include_clusters=True)
    outcome = primary_outcome(config)
    metadata = load_feature_metadata(config)
    outcome_cols = metadata.get("outcome_columns", [outcome])
    covars = covariate_columns(df, outcome_cols)

    model_type = config.get("propensity", {}).get("model", "logistic")
    scores, _ = fit_propensity(df, "transfused", covars, model_type=model_type)
    result_cols = ["stay_id", "transfused"]
    if outcome in df.columns:
        result_cols.append(outcome)
    if "phenotype" in df.columns:
        result_cols.append("phenotype")
    result = df[result_cols].copy()
    result["propensity_score"] = scores
    result = compute_weights(result)

    trim_cfg = config.get("propensity", {}).get("trim", {})
    result["in_common_support"] = True
    if trim_cfg.get("enabled", True):
        lower = float(trim_cfg.get("lower", 0.02))
        upper = float(trim_cfg.get("upper", 0.98))
        keep_ids = trim_overlap(result, "propensity_score", lower, upper)["stay_id"]
        result["in_common_support"] = result["stay_id"].isin(keep_ids)

    save_table(result, out_dir / "propensity_scores.parquet")
    save_table(pd.DataFrame({"covariate": covars}), out_dir / "propensity_covariates.csv")

    plt.figure(figsize=(7, 4))
    for treatment, label in [(0, "A=0"), (1, "A=1")]:
        subset = result[result["transfused"] == treatment]["propensity_score"]
        plt.hist(subset, bins=30, alpha=0.55, density=True, label=label)
    plt.xlabel("Propensity score")
    plt.ylabel("Density")
    plt.title("Propensity Overlap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "propensity_overlap.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate propensity scores.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_propensity(config)
    print("propensity_score_complete")


if __name__ == "__main__":
    main()

