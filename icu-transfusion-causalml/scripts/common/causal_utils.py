from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_FEATURES = {
    "stay_id",
    "subject_id",
    "transfused",
    "t0",
    "pseudo_t0",
    "pseudo_t0_offset_minutes",
    "data_source",
    "is_synthetic",
    "transfused_source",
    "min_time",
    "max_time",
    "t0_transf",
    "phenotype",
    "selected_k",
}


def covariate_columns(df: pd.DataFrame, outcome_cols: Iterable[str]) -> list[str]:
    exclude = EXCLUDE_FEATURES | set(outcome_cols)
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("synthetic_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == object or str(df[col].dtype).startswith("category"):
            cols.append(col)
    return cols


def design_matrix(df: pd.DataFrame, covariates: list[str]) -> tuple[np.ndarray, list[str], Pipeline]:
    x_df = df[covariates].copy() if covariates else pd.DataFrame(index=df.index)
    x_df = pd.get_dummies(x_df, dummy_na=True)
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    if x_df.shape[1] == 0:
        x_df["intercept_only"] = 1.0
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x = pipeline.fit_transform(x_df)
    return x, x_df.columns.tolist(), pipeline


def fit_propensity(df: pd.DataFrame, treatment_col: str, covariates: list[str], model_type: str = "logistic") -> tuple[np.ndarray, object]:
    x, _, _ = design_matrix(df, covariates)
    treatment = df[treatment_col].astype(int).to_numpy()
    if treatment.min() == treatment.max():
        scores = np.repeat(float(treatment.mean()), len(treatment))
        return scores, None
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(x, treatment)
    scores = model.predict_proba(x)[:, 1]
    return np.clip(scores, 1e-4, 1 - 1e-4), model


def trim_overlap(df: pd.DataFrame, score_col: str, lower: float, upper: float) -> pd.DataFrame:
    return df[(df[score_col] >= lower) & (df[score_col] <= upper)].copy()


def compute_weights(df: pd.DataFrame, treatment_col: str = "transfused", score_col: str = "propensity_score") -> pd.DataFrame:
    out = df.copy()
    a = out[treatment_col].astype(int).to_numpy()
    p = np.clip(out[score_col].to_numpy(dtype=float), 1e-4, 1 - 1e-4)
    p_treat = max(a.mean(), 1e-4)
    p_ctrl = max(1 - a.mean(), 1e-4)
    out["iptw_weight"] = np.where(a == 1, p_treat / p, p_ctrl / (1 - p))
    out["overlap_weight"] = np.where(a == 1, 1 - p, p)
    return out


def nearest_neighbor_match(
    df: pd.DataFrame,
    treatment_col: str,
    score_col: str,
    caliper: float,
    ratio: int = 1,
    replace: bool = False,
) -> pd.DataFrame:
    treated = df[df[treatment_col] == 1].copy()
    controls = df[df[treatment_col] == 0].copy()
    if treated.empty or controls.empty:
        return pd.DataFrame()

    logits = logit(np.clip(df[score_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    caliper_value = float(caliper) * float(np.nanstd(logits))
    treated["_logit"] = logit(np.clip(treated[score_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    controls["_logit"] = logit(np.clip(controls[score_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6))

    n_neighbors = min(len(controls), max(1, ratio * 5))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(controls[["_logit"]].to_numpy())
    distances, indices = nn.kneighbors(treated[["_logit"]].to_numpy())

    used_controls = set()
    rows = []
    controls_reset = controls.reset_index(drop=False)
    for i, (_, trow) in enumerate(treated.reset_index(drop=True).iterrows()):
        matched = 0
        for dist, idx in zip(distances[i], indices[i]):
            if dist > caliper_value:
                continue
            crow = controls_reset.iloc[int(idx)]
            ctrl_index = int(crow["index"])
            if not replace and ctrl_index in used_controls:
                continue
            used_controls.add(ctrl_index)
            rows.append(
                {
                    "stay_id_transf": trow["stay_id"],
                    "stay_id_ctrl": crow["stay_id"],
                    "propensity_score_transf": trow[score_col],
                    "propensity_score_ctrl": crow[score_col],
                    "distance": float(dist),
                    "caliper_value": caliper_value,
                }
            )
            matched += 1
            if matched >= ratio:
                break
    return pd.DataFrame(rows)


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.nanmean(values))
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not mask.any() or np.sum(weights[mask]) <= 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def effect_estimate(df: pd.DataFrame, outcome_col: str, treatment_col: str = "transfused", weight_col: Optional[str] = None) -> dict[str, float]:
    treated = df[df[treatment_col] == 1]
    controls = df[df[treatment_col] == 0]
    wt = treated[weight_col].to_numpy() if weight_col else None
    wc = controls[weight_col].to_numpy() if weight_col else None
    mt = weighted_mean(treated[outcome_col].to_numpy(), wt)
    mc = weighted_mean(controls[outcome_col].to_numpy(), wc)
    return {
        "n": int(len(df)),
        "n_treated": int(len(treated)),
        "n_control": int(len(controls)),
        "treated_mean": mt,
        "control_mean": mc,
        "effect": float(mt - mc) if np.isfinite(mt) and np.isfinite(mc) else np.nan,
    }


def effect_from_pairs(pairs: pd.DataFrame, df: pd.DataFrame, outcome_col: str) -> dict[str, float]:
    if pairs.empty:
        return {"n_pairs": 0, "effect": np.nan}
    outcomes = df[["stay_id", outcome_col]].drop_duplicates("stay_id")
    merged = pairs.merge(outcomes, left_on="stay_id_transf", right_on="stay_id", how="left").rename(columns={outcome_col: "y_t"})
    merged = merged.drop(columns=["stay_id"])
    merged = merged.merge(outcomes, left_on="stay_id_ctrl", right_on="stay_id", how="left").rename(columns={outcome_col: "y_c"})
    diff = merged["y_t"] - merged["y_c"]
    return {
        "n_pairs": int(len(merged)),
        "treated_mean": float(merged["y_t"].mean()),
        "control_mean": float(merged["y_c"].mean()),
        "effect": float(diff.mean()),
    }


def smd_table(df: pd.DataFrame, covariates: list[str], treatment_col: str = "transfused", weight_col: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for col in covariates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({"x": series, "a": df[treatment_col].astype(int)})
        if weight_col:
            tmp["w"] = pd.to_numeric(df[weight_col], errors="coerce")
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "a"])
        if tmp.empty or tmp["a"].nunique() < 2:
            continue
        treated = tmp[tmp["a"] == 1]
        controls = tmp[tmp["a"] == 0]
        wt = treated["w"].to_numpy() if weight_col and "w" in treated else None
        wc = controls["w"].to_numpy() if weight_col and "w" in controls else None
        mt = weighted_mean(treated["x"].to_numpy(), wt)
        mc = weighted_mean(controls["x"].to_numpy(), wc)
        vt = np.nanvar(treated["x"].to_numpy())
        vc = np.nanvar(controls["x"].to_numpy())
        pooled = np.sqrt((vt + vc) / 2.0)
        smd = (mt - mc) / pooled if pooled > 1e-12 else 0.0
        rows.append({"covariate": col, "smd": float(smd), "abs_smd": float(abs(smd))})
    return pd.DataFrame(rows)


def fit_outcome_model(x: np.ndarray, y: np.ndarray, binary: bool = True, seed: int = 42):
    if binary:
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    model.fit(x, y)
    return model


def predict_outcome(model, x: np.ndarray, binary: bool = True) -> np.ndarray:
    if binary and hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.asarray(model.predict(x), dtype=float)


def simple_predictive_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    metrics = {}
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auc"] = np.nan
        metrics["average_precision"] = np.nan
    metrics["brier_score"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
    y_pred = (y_score >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    metrics["accuracy"] = float((tp + tn) / max(len(y_true), 1))
    metrics["precision"] = float(tp / max(tp + fp, 1))
    metrics["recall"] = float(tp / max(tp + fn, 1))
    metrics["f1"] = float(2 * metrics["precision"] * metrics["recall"] / max(metrics["precision"] + metrics["recall"], 1e-12))
    return metrics
