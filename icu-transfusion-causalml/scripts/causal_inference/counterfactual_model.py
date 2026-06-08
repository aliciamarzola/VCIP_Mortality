from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.causal_utils import covariate_columns, design_matrix, predict_outcome, simple_predictive_metrics  # noqa: E402
from scripts.common.pipeline_common import counterfactual_dir, figures_dir, load_feature_metadata, load_config, load_processed_frame, primary_outcome, save_table  # noqa: E402


def _fit_arm_model(x, y, seed):
    if len(y) == 0:
        return None, 0.0
    if len(np.unique(y)) < 2:
        return None, float(np.mean(y))
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(x, y)
    return model, None


def _predict(model, constant, x):
    if model is None:
        return np.repeat(float(constant), x.shape[0])
    return predict_outcome(model, x, binary=True)


def run_counterfactual(config) -> None:
    out_dir = counterfactual_dir(config)
    fig_dir = figures_dir(config)
    df = load_processed_frame(config, include_clusters=True).dropna(subset=["transfused", primary_outcome(config)]).copy()
    outcome = primary_outcome(config)
    metadata = load_feature_metadata(config)
    covars = covariate_columns(df, metadata.get("outcome_columns", [outcome]))
    x, _, _ = design_matrix(df, covars)
    a = df["transfused"].astype(int).to_numpy()
    y = df[outcome].astype(int).to_numpy()
    seed = int(config.get("run", {}).get("seed", 42))

    stratify = y if len(np.unique(y)) > 1 else None
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=float(config.get("counterfactual", {}).get("test_fraction", 0.25)), random_state=seed, stratify=stratify)

    mt, ct = _fit_arm_model(x[train_idx][a[train_idx] == 1], y[train_idx][a[train_idx] == 1], seed)
    mc, cc = _fit_arm_model(x[train_idx][a[train_idx] == 0], y[train_idx][a[train_idx] == 0], seed + 1)
    y1 = _predict(mt, ct, x)
    y0 = _predict(mc, cc, x)
    observed_pred = np.where(a == 1, y1, y0)

    pred = pd.DataFrame({"stay_id": df["stay_id"].to_numpy(), "transfused": a, outcome: y, "predicted_y1": y1, "predicted_y0": y0})
    pred["ite"] = pred["predicted_y1"] - pred["predicted_y0"]
    pred["observed_prediction"] = observed_pred
    pred["test_split"] = 0
    pred.loc[test_idx, "test_split"] = 1
    save_table(pred, out_dir / "predicted_y0_y1.parquet")
    save_table(pred[["stay_id", "ite", "predicted_y0", "predicted_y1"]], out_dir / "individual_treatment_effects.parquet")

    metrics = simple_predictive_metrics(y[test_idx], observed_pred[test_idx])
    metrics["n_test"] = int(len(test_idx))
    metrics["model"] = "two_model_gradient_boosting"
    save_table(pd.DataFrame([metrics]), out_dir / "model_metrics.csv")

    plt.figure(figsize=(5, 5))
    plt.scatter(pred["predicted_y0"], pred["predicted_y1"], c=pred["transfused"], s=18, alpha=0.7)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Y(0)")
    plt.ylabel("Predicted Y(1)")
    plt.title("Counterfactual Risk")
    plt.tight_layout()
    plt.savefig(fig_dir / "counterfactual_risk_scatter.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit initial counterfactual model.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_counterfactual(config)
    print("counterfactual_model_complete")


if __name__ == "__main__":
    main()
