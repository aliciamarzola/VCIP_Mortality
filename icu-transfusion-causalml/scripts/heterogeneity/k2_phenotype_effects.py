from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import design_matrix, load_trial_config, output_root, read_json, save_table, summarize_effect, write_json  # noqa: E402


def _internal_kmeans_k2(df: pd.DataFrame, feature_cols: list[str], seed: int) -> pd.DataFrame:
    x, _, _ = design_matrix(df, feature_cols)
    km = KMeans(n_clusters=2, n_init=20, random_state=seed)
    labels = km.fit_predict(x)
    return pd.DataFrame({"stay_id": df["stay_id"], "k2_phenotype": labels})


def _enough_overlap(df: pd.DataFrame, labels: pd.DataFrame) -> bool:
    n_overlap = labels["stay_id"].isin(df["stay_id"]).sum()
    return n_overlap >= max(20, int(0.5 * len(df)))


def load_or_create_k2(df: pd.DataFrame, feature_cols: list[str], seed: int) -> tuple[pd.DataFrame, str]:
    candidates = [
        REPO_ROOT / "outputs" / "phenotypes" / "scan_group_assignments.parquet",
        REPO_ROOT / "outputs" / "causal_inference" / "phenotypes" / "scan_group_assignments.parquet",
        REPO_ROOT / "outputs" / "causal_inference" / "phenotypes" / "clusters.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            table = pd.read_parquet(path)
        except Exception:
            continue
        if "stay_id" not in table.columns:
            continue
        if "cluster_k2" in table.columns:
            labels = table[["stay_id", "cluster_k2"]].drop_duplicates("stay_id").rename(columns={"cluster_k2": "k2_phenotype"})
            if _enough_overlap(df, labels):
                return labels, str(path)
        if "k" in table.columns and "cluster" in table.columns:
            sub = table[pd.to_numeric(table["k"], errors="coerce") == 2].copy()
            if not sub.empty:
                labels = sub[["stay_id", "cluster"]].drop_duplicates("stay_id").rename(columns={"cluster": "k2_phenotype"})
                if _enough_overlap(df, labels):
                    return labels, str(path)
        if "phenotype" in table.columns:
            labels = table[["stay_id", "phenotype"]].drop_duplicates("stay_id").rename(columns={"phenotype": "k2_phenotype"})
            if _enough_overlap(df, labels):
                return labels, str(path)

    # If no phenotypes found, fail instead of creating internal K-means
    raise RuntimeError("No K=2 phenotypes found from article/cluster-transfusion. Cannot proceed with internal clustering.")


def run_k2_effects(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"
    het = out_root / "heterogeneity"
    metadata = read_json(proc / "target_trial_metadata.json")
    df = pd.read_parquet(proc / "trial_dataset.parquet")
    nuis = pd.read_parquet(caus / "nuisance_predictions.parquet")
    primary = metadata["primary_outcome"]
    feature_cols = metadata["feature_columns"]
    seed = int(config["target_trial"].get("seed", 42))
    k2, source = load_or_create_k2(df, feature_cols, seed)
    analysis = df.merge(nuis.drop(columns=["transfused", primary, "analysis_split"], errors="ignore"), on="stay_id", how="left").merge(k2, on="stay_id", how="left")
    rows = []
    for split_name, split_df in [("all", analysis), ("validation", analysis[analysis["analysis_split"] == "validation"])]:
        for group, sub in split_df.groupby("k2_phenotype"):
            if sub["transfused"].nunique() < 2:
                continue
            row = summarize_effect(sub, primary, "transfused")
            row.update({"split": split_name, "group": group, "source": source, "outcome": primary})
            rows.append(row)
    columns = [
        "split",
        "group",
        "source",
        "outcome",
        "n",
        "n_treated",
        "n_control",
        "treated_mean",
        "control_mean",
        "unadjusted_diff",
        "aipw_ate",
        "aipw_att",
        "overlap_weighted_diff",
        "iptw_diff",
    ]
    save_table(pd.DataFrame(rows, columns=columns), het / "effect_by_k2.csv")
    write_json(het / "k2_metadata.json", {"source": source, "interpretation": "K=2 is a physiologic macrophenotype, not a benefit/harm label."})
    print("k2_phenotype_effects_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate treatment effects by K=2 physiologic macrophenotype.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    run_k2_effects(args.config_dir)


if __name__ == "__main__":
    main()
