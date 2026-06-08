from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import (  # noqa: E402
    bootstrap_ci,
    crossfit_nuisance,
    load_trial_config,
    output_root,
    read_json,
    save_table,
    summarize_effect,
    weighted_mean,
    write_json,
)


def smd_for_feature(df: pd.DataFrame, feature: str, treatment_col: str, weight_col: str | None = None) -> float:
    x = pd.to_numeric(df[feature], errors="coerce")
    a = df[treatment_col].astype(int)
    tmp = pd.DataFrame({"x": x, "a": a})
    if weight_col:
        tmp["w"] = pd.to_numeric(df[weight_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "a"])
    if tmp.empty or tmp["a"].nunique() < 2:
        return np.nan
    t = tmp[tmp["a"] == 1]
    c = tmp[tmp["a"] == 0]
    if weight_col:
        mt = weighted_mean(t["x"].to_numpy(), t["w"].to_numpy())
        mc = weighted_mean(c["x"].to_numpy(), c["w"].to_numpy())
    else:
        mt = float(t["x"].mean())
        mc = float(c["x"].mean())
    pooled = np.sqrt((float(t["x"].var(ddof=0)) + float(c["x"].var(ddof=0))) / 2.0)
    return float((mt - mc) / pooled) if pooled > 1e-12 else 0.0


def bootstrap_metric(df: pd.DataFrame, metric_col: str, seed: int, n_boot: int) -> tuple[float, float]:
    return bootstrap_ci(df[metric_col].to_numpy(dtype=float), seed=seed, n_boot=n_boot)


def run_doubly_robust(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"
    metadata = read_json(proc / "target_trial_metadata.json")
    df = pd.read_parquet(proc / "trial_dataset.parquet")
    primary = metadata.get("primary_outcome", config["target_trial"]["outcomes"].get("primary", "mortality_anytime"))
    outcome_cols = metadata.get("outcome_columns", [primary])
    feature_cols = metadata.get("feature_columns", [])
    seed = int(config["target_trial"].get("seed", 42))
    n_boot = int(config.get("causal_estimators", {}).get("bootstrap", 300))

    nuisance = crossfit_nuisance(df, feature_cols, primary, "transfused", config)
    analysis = df.merge(nuisance.drop(columns=["transfused", primary, "analysis_split"], errors="ignore"), on="stay_id", how="left")
    save_table(analysis[["stay_id", "analysis_split", "transfused", primary, "propensity_score", "mu0", "mu1", "aipw_score_ate", "overlap_weight", "iptw_weight"]], caus / "nuisance_predictions.parquet")

    rows = []
    for split_name, sub in [("all", analysis), ("discovery", analysis[analysis["analysis_split"] == "discovery"]), ("validation", analysis[analysis["analysis_split"] == "validation"])]:
        if sub.empty or sub["transfused"].nunique() < 2:
            continue
        summary = summarize_effect(sub, primary, "transfused")
        ci_low, ci_high = bootstrap_metric(sub, "aipw_score_ate", seed, n_boot)
        summary.update({"split": split_name, "outcome": primary, "aipw_ate_ci95_low": ci_low, "aipw_ate_ci95_high": ci_high})
        rows.append(summary)
    effects = pd.DataFrame(rows)
    save_table(effects, caus / "main_effects.csv")

    balance_rows = []
    merged_for_balance = analysis.copy()
    for feature in feature_cols:
        if not pd.api.types.is_numeric_dtype(merged_for_balance[feature]):
            continue
        before = smd_for_feature(merged_for_balance, feature, "transfused")
        overlap = smd_for_feature(merged_for_balance, feature, "transfused", "overlap_weight")
        iptw = smd_for_feature(merged_for_balance, feature, "transfused", "iptw_weight")
        balance_rows.append({"feature": feature, "smd_unweighted": before, "smd_overlap": overlap, "smd_iptw": iptw})
    save_table(pd.DataFrame(balance_rows), caus / "balance_smd.csv")

    write_json(
        caus / "causal_estimator_metadata.json",
        {
            "estimator": "cross_fitted_aipw_with_overlap_and_iptw",
            "estimand_primary": "ATT and ATE reported; subgroup scan uses AIPW pseudo-outcome on honest splits",
            "n": int(len(analysis)),
            "n_features": int(len(feature_cols)),
            "bootstrap": n_boot,
        },
    )
    print(f"doubly_robust_complete n={len(analysis)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-fitted propensity/outcome models and doubly robust estimators.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    run_doubly_robust(args.config_dir)


if __name__ == "__main__":
    main()
