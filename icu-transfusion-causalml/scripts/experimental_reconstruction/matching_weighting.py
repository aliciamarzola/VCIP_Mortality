from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.common.causal_utils import effect_estimate, effect_from_pairs, nearest_neighbor_match
from scripts.common.pipeline_common import causal_dir, load_config, load_processed_frame, primary_outcome, safe_read_table, save_table


def run_matching_weighting(config) -> None:
    out_dir = causal_dir(config)
    df = load_processed_frame(config, include_clusters=True)
    ps = pd.read_parquet(out_dir / "propensity_scores.parquet")
    df = df.merge(ps[["stay_id", "propensity_score", "iptw_weight", "overlap_weight", "in_common_support"]], on="stay_id", how="left")
    df_support = df[df["in_common_support"].fillna(True)].copy()

    outcome = primary_outcome(config)
    prop_cfg = config.get("propensity", {})
    pairs = nearest_neighbor_match(
        df_support,
        treatment_col="transfused",
        score_col="propensity_score",
        caliper=float(prop_cfg.get("caliper", 0.3)),
        ratio=int(prop_cfg.get("matching_ratio", 1)),
        replace=bool(prop_cfg.get("replace", False)),
    )
    save_table(pairs, out_dir / "matched_population.parquet")
    save_table(df_support, out_dir / "weighted_population.parquet")

    rows = []
    unadjusted = effect_estimate(df_support, outcome)
    rows.append({"method": "unadjusted_common_support", "estimand": "ATE", **unadjusted})
    iptw = effect_estimate(df_support, outcome, weight_col="iptw_weight")
    rows.append({"method": "iptw", "estimand": "ATE", **iptw})
    overlap = effect_estimate(df_support, outcome, weight_col="overlap_weight")
    rows.append({"method": "overlap_weight", "estimand": "ATO", **overlap})
    matched = effect_from_pairs(pairs, df_support, outcome)
    rows.append({"method": "nearest_neighbor_psm", "estimand": "ATT", **matched})

    if "phenotype" in df_support.columns:
        for phenotype, sub in df_support.groupby("phenotype"):
            if sub["transfused"].nunique() < 2:
                continue
            est = effect_estimate(sub, outcome, weight_col="overlap_weight")
            rows.append({"method": "overlap_weight", "estimand": "CATE_by_phenotype", "phenotype": phenotype, **est})

    estimates = pd.DataFrame(rows)
    save_table(estimates, out_dir / "effect_estimates_psm.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matching and weighting.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_matching_weighting(config)
    print("matching_weighting_complete")


if __name__ == "__main__":
    main()

