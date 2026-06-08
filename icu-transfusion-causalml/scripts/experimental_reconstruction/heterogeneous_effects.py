from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from scripts.common.pipeline_common import causal_dir, figures_dir, load_config, phenotypes_dir, save_table


def run_heterogeneous_effects(config) -> None:
    out_dir = causal_dir(config)
    fig_dir = figures_dir(config)
    ite = pd.read_parquet(out_dir / "ite_estimates.parquet")
    clusters_path = phenotypes_dir(config) / "clusters.parquet"
    clusters = pd.read_parquet(clusters_path) if clusters_path.exists() else pd.DataFrame()
    if not clusters.empty:
        ite = ite.merge(clusters[["stay_id", "phenotype"]], on="stay_id", how="left")

    ite_cols = [c for c in ite.columns if c.startswith("ite_")]
    preferred = "ite_causal_forest" if "ite_causal_forest" in ite_cols else ("ite_t_learner" if "ite_t_learner" in ite_cols else ite_cols[0])
    if "phenotype" in ite.columns:
        effect = (
            ite.groupby("phenotype")
            .agg(n=("stay_id", "count"), cate_mean=(preferred, "mean"), cate_median=(preferred, "median"), cate_sd=(preferred, "std"))
            .reset_index()
        )
    else:
        effect = pd.DataFrame([{"phenotype": "all", "n": len(ite), "cate_mean": ite[preferred].mean(), "cate_median": ite[preferred].median(), "cate_sd": ite[preferred].std()}])
    save_table(effect, out_dir / "effect_by_phenotype.csv")

    ranked = ite[["stay_id", preferred] + (["phenotype"] if "phenotype" in ite.columns else [])].sort_values(preferred)
    save_table(ranked, out_dir / "patient_benefit_ranking.csv")

    plt.figure(figsize=(7, 4))
    plt.hist(ite[preferred].dropna(), bins=30)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("ITE risk difference")
    plt.ylabel("N")
    plt.title("ITE Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "ite_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    yerr = effect["cate_sd"].fillna(0) if "cate_sd" in effect.columns else None
    plt.errorbar(effect["cate_mean"], effect["phenotype"].astype(str), xerr=yerr, fmt="o")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("CATE risk difference")
    plt.ylabel("Phenotype")
    plt.title("Effect by Phenotype")
    plt.tight_layout()
    plt.savefig(fig_dir / "forest_plot_effect_by_phenotype.png", dpi=160)
    plt.close()

    if "phenotype" in ite.columns:
        labels = sorted(ite["phenotype"].dropna().unique())
        data = [ite.loc[ite["phenotype"] == label, preferred].dropna() for label in labels]
        plt.figure(figsize=(7, 4))
        plt.boxplot(data, labels=[str(x) for x in labels])
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Phenotype")
        plt.ylabel("ITE risk difference")
        plt.title("CATE by Cluster")
        plt.tight_layout()
        plt.savefig(fig_dir / "cate_by_cluster_boxplot.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize heterogeneous effects.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_heterogeneous_effects(config)
    print("heterogeneous_effects_complete")


if __name__ == "__main__":
    main()

