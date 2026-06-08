from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from scripts.common.causal_utils import effect_from_pairs, nearest_neighbor_match
from scripts.common.pipeline_common import causal_dir, evaluation_dir, figures_dir, load_config, load_processed_frame, primary_outcome, save_table


def run_sensitivity(config) -> None:
    c_dir = causal_dir(config)
    eval_dir = evaluation_dir(config)
    fig_dir = figures_dir(config)
    df = load_processed_frame(config, include_clusters=True)
    ps = pd.read_parquet(c_dir / "propensity_scores.parquet")
    df = df.merge(ps[["stay_id", "propensity_score", "in_common_support"]], on="stay_id", how="left")
    df = df[df["in_common_support"].fillna(True)].copy()
    outcome = primary_outcome(config)

    rows = []
    for caliper in config.get("evaluation", {}).get("sensitivity_calipers", [0.1, 0.2, 0.3, 0.5]):
        pairs = nearest_neighbor_match(df, "transfused", "propensity_score", float(caliper), ratio=1, replace=False)
        est = effect_from_pairs(pairs, df, outcome)
        rows.append({"analysis": "caliper", "caliper": float(caliper), **est})
    results = pd.DataFrame(rows)
    save_table(results, eval_dir / "sensitivity_results.csv")

    plt.figure(figsize=(7, 4))
    plt.plot(results["caliper"], results["effect"], marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Caliper")
    plt.ylabel("Matched risk difference")
    plt.title("Sensitivity Analysis")
    plt.tight_layout()
    plt.savefig(fig_dir / "sensitivity_analysis.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitivity analyses.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_sensitivity(config)
    print("sensitivity_analysis_complete")


if __name__ == "__main__":
    main()

