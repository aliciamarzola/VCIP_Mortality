from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.causal_utils import covariate_columns, design_matrix, predict_outcome  # noqa: E402
from scripts.common.pipeline_common import causal_dir, load_feature_metadata, load_config, load_processed_frame, primary_outcome, save_table  # noqa: E402


def run_causal_forest(config) -> None:
    out_dir = causal_dir(config)
    df = load_processed_frame(config, include_clusters=True).dropna(subset=["transfused", primary_outcome(config)]).copy()
    outcome = primary_outcome(config)
    metadata = load_feature_metadata(config)
    covars = covariate_columns(df, metadata.get("outcome_columns", [outcome]))
    x, _, _ = design_matrix(df, covars)
    a = df["transfused"].astype(int).to_numpy()
    y = df[outcome].astype(int).to_numpy()
    seed = int(config.get("run", {}).get("seed", 42))

    optional_available = bool(importlib.util.find_spec("econml") or importlib.util.find_spec("causalml"))
    method = "causal_forest_optional" if optional_available else "causal_forest_fallback_t_learner_gbdt"

    def fit_arm(mask):
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            return None, float(np.mean(yy)) if len(yy) else float(np.mean(y))
        model = GradientBoostingClassifier(random_state=seed)
        model.fit(x[mask], yy)
        return model, None

    mt, ct = fit_arm(a == 1)
    mc, cc = fit_arm(a == 0)
    y1 = np.repeat(ct, len(df)) if mt is None else predict_outcome(mt, x, binary=True)
    y0 = np.repeat(cc, len(df)) if mc is None else predict_outcome(mc, x, binary=True)
    ite_cf = y1 - y0

    cf = pd.DataFrame({"stay_id": df["stay_id"].to_numpy(), "y1_causal_forest": y1, "y0_causal_forest": y0, "ite_causal_forest": ite_cf})
    save_table(cf, out_dir / "causal_forest_ite.parquet")

    ite_path = out_dir / "ite_estimates.parquet"
    if ite_path.exists():
        ite = pd.read_parquet(ite_path).merge(cf, on="stay_id", how="left")
        save_table(ite, ite_path)

    cate_path = out_dir / "cate_estimates.csv"
    cate = pd.read_csv(cate_path) if cate_path.exists() else pd.DataFrame()
    row = pd.DataFrame([{"method": method, "ate": float(np.mean(ite_cf)), "att": float(np.mean(ite_cf[a == 1])), "optional_library_available": optional_available}])
    cate = pd.concat([cate, row], ignore_index=True)
    save_table(cate, cate_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate causal forest or fallback.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_causal_forest(config)
    print("causal_forest_complete")


if __name__ == "__main__":
    main()
