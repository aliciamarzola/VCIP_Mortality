from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from scripts.common.causal_utils import covariate_columns, smd_table
from scripts.common.pipeline_common import causal_dir, figures_dir, load_feature_metadata, load_config, load_processed_frame, primary_outcome, save_table


def run_balance(config) -> None:
    out_dir = causal_dir(config)
    fig_dir = figures_dir(config)
    df = load_processed_frame(config, include_clusters=True)
    ps = pd.read_parquet(out_dir / "propensity_scores.parquet")
    df = df.merge(ps[["stay_id", "iptw_weight", "overlap_weight", "in_common_support"]], on="stay_id", how="left")
    df = df[df["in_common_support"].fillna(True)].copy()
    metadata = load_feature_metadata(config)
    covars = covariate_columns(df, metadata.get("outcome_columns", [primary_outcome(config)]))

    pre = smd_table(df, covars).rename(columns={"smd": "smd_pre", "abs_smd": "abs_smd_pre"})
    iptw = smd_table(df, covars, weight_col="iptw_weight").rename(columns={"smd": "smd_iptw", "abs_smd": "abs_smd_iptw"})
    overlap = smd_table(df, covars, weight_col="overlap_weight").rename(columns={"smd": "smd_overlap", "abs_smd": "abs_smd_overlap"})

    pairs = pd.read_parquet(out_dir / "matched_population.parquet")
    if not pairs.empty:
        matched_ids = set(pairs["stay_id_transf"]).union(set(pairs["stay_id_ctrl"]))
        matched_df = df[df["stay_id"].isin(matched_ids)].copy()
        matched = smd_table(matched_df, covars).rename(columns={"smd": "smd_matched", "abs_smd": "abs_smd_matched"})
    else:
        matched = pd.DataFrame(columns=["covariate", "smd_matched", "abs_smd_matched"])

    table = pre.merge(iptw, on="covariate", how="outer").merge(overlap, on="covariate", how="outer").merge(matched, on="covariate", how="outer")
    save_table(table, out_dir / "balance_table.csv")

    plot_df = table.copy()
    if "abs_smd_pre" in plot_df.columns:
        plot_df = plot_df.sort_values("abs_smd_pre", ascending=False).head(25)
    y = range(len(plot_df))
    plt.figure(figsize=(8, max(4, 0.28 * len(plot_df))))
    if "abs_smd_pre" in plot_df.columns:
        plt.scatter(plot_df["abs_smd_pre"], y, label="Pre")
    if "abs_smd_matched" in plot_df.columns:
        plt.scatter(plot_df["abs_smd_matched"], y, label="Matched")
    if "abs_smd_overlap" in plot_df.columns:
        plt.scatter(plot_df["abs_smd_overlap"], y, label="Overlap")
    plt.axvline(0.1, color="black", linestyle="--", linewidth=1)
    plt.yticks(list(y), plot_df["covariate"], fontsize=7)
    plt.xlabel("|SMD|")
    plt.title("Love Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "love_plot.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balance diagnostics.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_balance(config)
    print("balance_diagnostics_complete")


if __name__ == "__main__":
    main()

