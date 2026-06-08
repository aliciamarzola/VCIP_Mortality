from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import bootstrap_ci, load_trial_config, output_root, read_json, rule_mask, rule_to_text, save_table  # noqa: E402


def run_validation(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"
    het = out_root / "heterogeneity"
    df = pd.read_parquet(proc / "trial_dataset.parquet")
    nuis = pd.read_parquet(caus / "nuisance_predictions.parquet")
    rules_path = het / "honest_scan_discovery_selected_rules.csv"
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing discovery rules: {rules_path}")
    try:
        rules = pd.read_csv(rules_path)
    except pd.errors.EmptyDataError:
        rules = pd.DataFrame()
    analysis = df.merge(nuis[["stay_id", "aipw_score_ate", "propensity_score"]], on="stay_id", how="left")
    validation = analysis[analysis["analysis_split"] == "validation"].copy()
    seed = int(config["target_trial"].get("seed", 42))
    n_boot = int(config.get("honest_scan", {}).get("bootstrap", 200))

    columns = [
        "discovery_direction",
        "validation_direction",
        "rule_text",
        "rule_json",
        "n_validation",
        "n_treated_validation",
        "n_control_validation",
        "effect_aipw_validation",
        "ci95_low_validation",
        "ci95_high_validation",
        "effect_aipw_discovery",
        "n_discovery",
        "validated_same_sign",
    ]
    rows = []
    if rules.empty:
        save_table(pd.DataFrame(columns=columns), het / "honest_scan_validation.csv")
        print("honest_scan_validation_complete rules=0")
        return
    for _, rule_row in rules.iterrows():
        rule = json.loads(rule_row["rule_json"])
        mask = rule_mask(validation, rule)
        sub = validation[mask].copy()
        values = sub["aipw_score_ate"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(values, seed, n_boot)
        effect = float(np.nanmean(values)) if len(values) else np.nan
        rows.append(
            {
                "discovery_direction": rule_row.get("direction"),
                "validation_direction": "benefit_apparent" if effect < 0 else "harm_apparent" if effect > 0 else "neutral_apparent",
                "rule_text": rule_to_text(rule),
                "rule_json": rule_row["rule_json"],
                "n_validation": int(len(sub)),
                "n_treated_validation": int(sub["transfused"].sum()) if not sub.empty else 0,
                "n_control_validation": int((1 - sub["transfused"].astype(int)).sum()) if not sub.empty else 0,
                "effect_aipw_validation": effect,
                "ci95_low_validation": ci_low,
                "ci95_high_validation": ci_high,
                "effect_aipw_discovery": rule_row.get("effect_aipw"),
                "n_discovery": rule_row.get("n"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["validated_same_sign"] = np.sign(out["effect_aipw_validation"]) == np.sign(out["effect_aipw_discovery"].astype(float))
    save_table(out, het / "honest_scan_validation.csv")
    print(f"honest_scan_validation_complete rules={len(out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate discovered scan rules on the held-out validation split.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    run_validation(args.config_dir)


if __name__ == "__main__":
    main()
