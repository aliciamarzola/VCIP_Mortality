from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from scripts.common.causal_utils import covariate_columns, design_matrix, predict_outcome
from scripts.common.pipeline_common import causal_dir, load_feature_metadata, load_config, load_processed_frame, primary_outcome, save_table


def _fit_binary_model(x: np.ndarray, y: np.ndarray, seed: int):
    if len(np.unique(y)) < 2:
        return None, float(np.mean(y))
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    model.fit(x, y)
    return model, None


def _predict_binary(model, constant, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.repeat(float(constant), x.shape[0])
    return predict_outcome(model, x, binary=True)


def run_meta_learners(config) -> None:
    out_dir = causal_dir(config)
    df = load_processed_frame(config, include_clusters=True).dropna(subset=["transfused", primary_outcome(config)]).copy()
    outcome = primary_outcome(config)
    metadata = load_feature_metadata(config)
    covars = covariate_columns(df, metadata.get("outcome_columns", [outcome]))
    x, _, _ = design_matrix(df, covars)
    a = df["transfused"].astype(int).to_numpy()
    y = df[outcome].astype(int).to_numpy()
    seed = int(config.get("run", {}).get("seed", 42))

    ite = pd.DataFrame({"stay_id": df["stay_id"].to_numpy(), "transfused": a, outcome: y})
    estimates = []

    xs = np.column_stack([x, a])
    s_model, s_constant = _fit_binary_model(xs, y, seed)
    mu1_s = _predict_binary(s_model, s_constant, np.column_stack([x, np.ones(len(df))]))
    mu0_s = _predict_binary(s_model, s_constant, np.column_stack([x, np.zeros(len(df))]))
    ite["y1_s_learner"] = mu1_s
    ite["y0_s_learner"] = mu0_s
    ite["ite_s_learner"] = mu1_s - mu0_s
    estimates.append({"method": "s_learner", "ate": float(ite["ite_s_learner"].mean()), "att": float(ite.loc[a == 1, "ite_s_learner"].mean())})

    t_model, t_const = _fit_binary_model(x[a == 1], y[a == 1], seed)
    c_model, c_const = _fit_binary_model(x[a == 0], y[a == 0], seed)
    mu1_t = _predict_binary(t_model, t_const, x)
    mu0_t = _predict_binary(c_model, c_const, x)
    ite["y1_t_learner"] = mu1_t
    ite["y0_t_learner"] = mu0_t
    ite["ite_t_learner"] = mu1_t - mu0_t
    estimates.append({"method": "t_learner", "ate": float(ite["ite_t_learner"].mean()), "att": float(ite.loc[a == 1, "ite_t_learner"].mean())})

    tau_t = y[a == 1] - mu0_t[a == 1]
    tau_c = mu1_t[a == 0] - y[a == 0]
    reg_t = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    reg_c = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed + 1, n_jobs=-1)
    if len(tau_t) >= 5:
        reg_t.fit(x[a == 1], tau_t)
        tau_hat_t = reg_t.predict(x)
    else:
        tau_hat_t = np.repeat(np.nanmean(tau_t) if len(tau_t) else 0.0, len(df))
    if len(tau_c) >= 5:
        reg_c.fit(x[a == 0], tau_c)
        tau_hat_c = reg_c.predict(x)
    else:
        tau_hat_c = np.repeat(np.nanmean(tau_c) if len(tau_c) else 0.0, len(df))
    ps_path = out_dir / "propensity_scores.parquet"
    if ps_path.exists():
        ps = pd.read_parquet(ps_path)[["stay_id", "propensity_score"]]
        g = df[["stay_id"]].merge(ps, on="stay_id", how="left")["propensity_score"].fillna(a.mean()).to_numpy()
    else:
        g = np.repeat(a.mean(), len(df))
    tau_x = g * tau_hat_c + (1 - g) * tau_hat_t
    ite["ite_x_learner"] = tau_x
    ite["y1_x_learner"] = np.clip(mu0_t + tau_x, 0, 1)
    ite["y0_x_learner"] = mu0_t
    estimates.append({"method": "x_learner_simplified", "ate": float(np.mean(tau_x)), "att": float(np.mean(tau_x[a == 1]))})

    save_table(ite, out_dir / "ite_estimates.parquet")
    save_table(pd.DataFrame(estimates), out_dir / "cate_estimates.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate CATE/ITE with meta-learners.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_meta_learners(config)
    print("cate_meta_learners_complete")


if __name__ == "__main__":
    main()

