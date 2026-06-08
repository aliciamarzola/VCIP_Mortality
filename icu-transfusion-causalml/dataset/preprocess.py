from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .feature_engineering import build_static_feature_table, build_temporal_tensor, pre_t0_window, select_static_temporal_columns
from .load_data import load_config, resolve_path


def _processed_dir(config: Dict[str, Any], output_dir: Optional[str] = None) -> Path:
    if output_dir:
        return resolve_path(output_dir)
    return resolve_path(config.get("paths", {}).get("processed_dir", "outputs/causal_inference/processed"))


def preprocess(config: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
    out_dir = _processed_dir(config, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temporal_path = out_dir / "raw_temporal.parquet"
    cohort_path = out_dir / "cohort.parquet"
    if not temporal_path.exists() or not cohort_path.exists():
        raise FileNotFoundError("Missing raw_temporal.parquet or cohort.parquet. Run dataset.build_cohort first.")

    temporal = pd.read_parquet(temporal_path)
    cohort = pd.read_parquet(cohort_path)

    preprocessing = config.get("preprocessing", {})
    window_hours = int(preprocessing.get("window_hours", config.get("cohort", {}).get("decision_window_hours", 48)))
    step_minutes = int(preprocessing.get("time_step_minutes", 60))
    seed = int(config.get("run", {}).get("seed", 42))
    train_fraction = float(preprocessing.get("normalization", {}).get("train_fraction", 0.8))

    pre = pre_t0_window(temporal, cohort, window_hours)
    static_features = build_static_feature_table(temporal, pre, cohort, config)
    _, temporal_cols = select_static_temporal_columns(temporal, config)
    tensor, missing_mask, tensor_ids, tensor_metadata = build_temporal_tensor(
        pre=pre,
        cohort=cohort,
        temporal_cols=temporal_cols,
        window_hours=window_hours,
        step_minutes=step_minutes,
        seed=seed,
        train_fraction=train_fraction,
    )

    outcome_cols = []
    outcomes_cfg = config.get("variables", {}).get("outcomes", {})
    outcome_cols.extend(outcomes_cfg.get("primary", []))
    outcome_cols.extend(outcomes_cfg.get("secondary", []))
    outcome_cols = [col for col in outcome_cols if col in cohort.columns]

    outcomes = cohort[["stay_id"] + outcome_cols].copy()
    treatment = cohort[["stay_id", "transfused", "t0", "pseudo_t0", "data_source", "is_synthetic"]].copy()

    static_features.to_parquet(out_dir / "static_features.parquet", index=False)
    outcomes.to_parquet(out_dir / "outcomes.parquet", index=False)
    treatment.to_parquet(out_dir / "treatment.parquet", index=False)
    tensor_ids.to_parquet(out_dir / "temporal_tensor_ids.parquet", index=False)
    np.save(out_dir / "temporal_tensor.npy", tensor)
    np.save(out_dir / "temporal_missing_mask.npy", missing_mask)

    analysis = static_features.merge(treatment[["stay_id", "transfused"]], on="stay_id", how="left").merge(outcomes, on="stay_id", how="left")
    analysis.to_parquet(out_dir / "analysis_features.parquet", index=False)

    metadata = {
        "n_stays": int(len(cohort)),
        "n_pre_rows": int(len(pre)),
        "static_feature_columns": [col for col in static_features.columns if col != "stay_id"],
        "outcome_columns": outcome_cols,
        "primary_outcome": config.get("columns", {}).get("primary_outcome", "mortality_anytime"),
        "treatment_column": "transfused",
        "temporal_tensor": tensor_metadata,
        "no_post_t0_features": True,
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess pre-t0 temporal features.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    metadata = preprocess(config, output_dir=args.output_dir)
    print(f"processed_stays={metadata['n_stays']} temporal_shape={metadata['temporal_tensor']['shape']}")


if __name__ == "__main__":
    main()
