from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.pipeline_common import figures_dir, load_config, load_processed_frame, phenotypes_dir, primary_outcome, save_table


def analyze_clusters(config) -> None:
    out_dir = phenotypes_dir(config)
    fig_dir = figures_dir(config)
    df = load_processed_frame(config, include_clusters=True)
    outcome = primary_outcome(config)
    if "phenotype" not in df.columns:
        raise FileNotFoundError("Missing phenotype assignments. Run clustering first.")

    summary = (
        df.groupby("phenotype", dropna=False)
        .agg(
            n=("stay_id", "count"),
            treatment_rate=("transfused", "mean"),
            mortality_rate=(outcome, "mean"),
        )
        .reset_index()
    )
    numeric_cols = [c for c in df.columns if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols[:30]:
        summary[col] = df.groupby("phenotype")[col].mean().reindex(summary["phenotype"]).to_numpy()
    save_table(summary, out_dir / "cluster_summary.csv")

    heat_cols = numeric_cols[:20]
    if heat_cols:
        matrix = df.groupby("phenotype")[heat_cols].mean()
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0).replace(0, 1)
        plt.figure(figsize=(10, max(3, 0.4 * len(matrix))))
        plt.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm")
        plt.colorbar(label="Z-score")
        plt.yticks(range(len(matrix.index)), matrix.index.astype(str))
        plt.xticks(range(len(heat_cols)), heat_cols, rotation=90, fontsize=7)
        plt.title("Cluster Heatmap")
        plt.tight_layout()
        plt.savefig(fig_dir / "cluster_heatmap.png", dpi=160)
        plt.close()
    else:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No numeric cluster features", ha="center", va="center")
        plt.axis("off")
        plt.savefig(fig_dir / "cluster_heatmap.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal phenotypes.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    analyze_clusters(config)
    print("cluster_analysis_complete")


if __name__ == "__main__":
    main()

