from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import bootstrap_ci, load_trial_config, output_root, read_json, rule_mask, rule_to_text, save_table, write_json  # noqa: E402


def numeric_scan_features(df: pd.DataFrame, feature_cols: list[str], max_features: int) -> list[str]:
    candidates = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    suffix_priority = ("_last", "_mean", "_median", "_max", "_min", "_slope", "_std")
    candidates = [c for c in candidates if c.endswith(suffix_priority) or c in {"age", "bmi"}]
    scores = []
    y = df["aipw_score_ate"].astype(float)
    for col in candidates:
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 50 or x[mask].nunique() < 4:
            continue
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        scores.append((col, abs(float(corr)) if np.isfinite(corr) else 0.0))
    return [c for c, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:max_features]]


def evaluate_rule(df: pd.DataFrame, rule: list[dict], seed: int, n_boot: int) -> dict:
    mask = rule_mask(df, rule)
    sub = df[mask].copy()
    effect_values = sub["aipw_score_ate"].astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(effect_values, seed, n_boot)
    treated = int(sub["transfused"].sum()) if "transfused" in sub else 0
    controls = int((1 - sub["transfused"].astype(int)).sum()) if "transfused" in sub else 0
    return {
        "rule_text": rule_to_text(rule),
        "rule_json": json.dumps(rule, sort_keys=True),
        "n": int(len(sub)),
        "n_treated": treated,
        "n_control": controls,
        "effect_aipw": float(np.nanmean(effect_values)) if len(effect_values) else np.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def run_discovery(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"
    het = out_root / "heterogeneity"
    metadata = read_json(proc / "target_trial_metadata.json")
    df = pd.read_parquet(proc / "trial_dataset.parquet")
    nuis = pd.read_parquet(caus / "nuisance_predictions.parquet")
    analysis = df.merge(nuis[["stay_id", "aipw_score_ate", "propensity_score"]], on="stay_id", how="left")
    discovery = analysis[analysis["analysis_split"] == "discovery"].copy()

    scan = config.get("honest_scan", {})
    seed = int(config["target_trial"].get("seed", 42))
    min_n = int(scan.get("min_n", 150))
    min_t = int(scan.get("min_treated", 40))
    min_c = int(scan.get("min_control", 40))
    benefit_threshold = float(scan.get("benefit_threshold", -0.03))
    harm_threshold = float(scan.get("harm_threshold", 0.03))
    n_boot = int(scan.get("bootstrap", 200))
    max_features = int(scan.get("max_features", 24))
    max_rules = int(scan.get("max_rules", 80))
    quantiles = [float(q) for q in scan.get("quantiles", [0.25, 0.5, 0.75])]
    features = numeric_scan_features(discovery, metadata.get("feature_columns", []), max_features)

    rules: list[list[dict]] = []
    for col in features:
        values = pd.to_numeric(discovery[col], errors="coerce").dropna()
        for q in quantiles:
            threshold = float(values.quantile(q))
            rules.append([{"feature": col, "op": "<=", "threshold": threshold}])
            rules.append([{"feature": col, "op": ">", "threshold": threshold}])
    for col1, col2 in combinations(features[:12], 2):
        v1 = pd.to_numeric(discovery[col1], errors="coerce").dropna()
        v2 = pd.to_numeric(discovery[col2], errors="coerce").dropna()
        for q in [0.5]:
            rules.append([{"feature": col1, "op": ">", "threshold": float(v1.quantile(q))}, {"feature": col2, "op": ">", "threshold": float(v2.quantile(q))}])
            rules.append([{"feature": col1, "op": "<=", "threshold": float(v1.quantile(q))}, {"feature": col2, "op": "<=", "threshold": float(v2.quantile(q))}])

    base_columns = [
        "rule_text",
        "rule_json",
        "n",
        "n_treated",
        "n_control",
        "effect_aipw",
        "ci95_low",
        "ci95_high",
        "direction",
        "rank_abs_effect",
    ]
    rows = []
    for rule in rules:
        row = evaluate_rule(discovery, rule, seed, n_boot)
        if row["n"] < min_n or row["n_treated"] < min_t or row["n_control"] < min_c:
            continue
        if row["effect_aipw"] <= benefit_threshold:
            row["direction"] = "benefit_apparent"
        elif row["effect_aipw"] >= harm_threshold:
            row["direction"] = "harm_apparent"
        else:
            row["direction"] = "neutral_apparent"
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank_abs_effect"] = out["effect_aipw"].abs().rank(method="first", ascending=False)
        selected = out[out["direction"] != "neutral_apparent"].sort_values("rank_abs_effect").head(max_rules)
    else:
        out = pd.DataFrame(columns=base_columns)
        selected = pd.DataFrame(columns=base_columns)
    save_table(out.sort_values("rank_abs_effect") if "rank_abs_effect" in out else out, het / "honest_scan_discovery_all_rules.csv")
    save_table(selected, het / "honest_scan_discovery_selected_rules.csv")
    write_json(het / "honest_scan_metadata.json", {"split": "discovery", "rule_count": int(len(out)), "selected_count": int(len(selected)), "features_scanned": features})
    print(f"honest_scan_discovery_complete selected={len(selected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover candidate benefit/harm rules on the discovery split only.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    run_discovery(args.config_dir)


if __name__ == "__main__":
    main()
