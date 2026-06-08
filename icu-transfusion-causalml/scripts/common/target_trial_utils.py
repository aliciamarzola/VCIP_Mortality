from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.load_data import deep_update, load_config as load_base_config  # noqa: E402


def load_trial_config(config_dir: str | Path = "configs") -> dict[str, Any]:
    config = load_base_config(config_dir)
    path = Path(config_dir)
    if not path.is_absolute():
        path = ROOT / path
    trial_path = path / "target_trial_config.yaml"
    if trial_path.exists():
        with trial_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        deep_update(config, payload)
    return config


def output_root(config: dict[str, Any]) -> Path:
    value = config.get("target_trial", {}).get("output_dir", "outputs/target_trial")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input format: {path}")


def existing_table(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def stable_split_id(value: Any, seed: int, discovery_fraction: float) -> str:
    token = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    number = int(digest[:12], 16) / float(16**12)
    return "discovery" if number < discovery_fraction else "validation"


def make_synthetic_longitudinal(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial = config.get("target_trial", {})
    seed = int(trial.get("seed", 42))
    n = int(trial.get("synthetic_n", 600))
    rng = np.random.default_rng(seed)
    times = np.arange(-48 * 60, 13 * 60, 60)
    rows = []
    outcome_rows = []
    for i in range(n):
        stay_id = 30_000_000 + i
        subject_id = 10_000_000 + i
        age = int(np.clip(rng.normal(66, 14), 18, 95))
        sex = "F" if rng.random() < 0.46 else "M"
        bmi = float(np.clip(rng.normal(28, 6), 15, 55))
        frailty = rng.normal()
        base_hb = rng.normal(9.2 - 0.35 * frailty, 1.2)
        low_flow = rng.binomial(1, 1 / (1 + np.exp(-(frailty - 0.2))))
        first_low_time = None
        treated_time = None
        for t in times:
            hb = base_hb - 0.00035 * (t + 48 * 60) + rng.normal(0, 0.25)
            if first_low_time is None and hb <= 8.0 and t >= -36 * 60:
                first_low_time = float(t)
                p_tx = 1 / (1 + np.exp(-(-0.5 + 0.9 * low_flow + 0.5 * (8.0 - hb) + 0.2 * frailty)))
                if rng.random() < p_tx:
                    treated_time = float(t + rng.choice([60, 120, 180, 240]))
            rows.append(
                {
                    "stay_id": stay_id,
                    "subject_id": subject_id,
                    "time_min": float(t),
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "hemoglobin": float(hb),
                    "lactate": float(np.exp(rng.normal(0.4 + 0.35 * low_flow + 0.12 * frailty, 0.35))),
                    "creatinine": float(np.exp(rng.normal(0.2 + 0.25 * low_flow + 0.15 * frailty, 0.35))),
                    "platelets": float(np.clip(rng.normal(210 - 25 * low_flow, 55), 20, 600)),
                    "heart_rate": float(np.clip(rng.normal(88 + 10 * low_flow + 3 * frailty, 12), 35, 170)),
                    "mbp": float(np.clip(rng.normal(78 - 6 * low_flow, 10), 35, 130)),
                    "dbp": float(np.clip(rng.normal(55 - 5 * low_flow, 8), 20, 100)),
                    "sbp": float(np.clip(rng.normal(118 - 8 * low_flow, 18), 60, 220)),
                    "resp_rate": float(np.clip(rng.normal(19 + 3 * low_flow, 4), 6, 45)),
                    "spo2": float(np.clip(rng.normal(97 - 1.5 * low_flow, 2), 70, 100)),
                    "sofa": float(np.clip(rng.normal(5 + 2.5 * low_flow + frailty, 2), 0, 22)),
                    "any_vasopressor": float(rng.random() < (0.2 + 0.35 * low_flow)),
                    "ventilation_on": float(rng.random() < (0.15 + 0.4 * low_flow)),
                    "rbc_transfusion_flag": int(treated_time is not None and abs(float(t) - treated_time) < 1e-6),
                }
            )
        if first_low_time is None:
            first_low_time = float(rng.choice(times[12:36]))
        treated = int(treated_time is not None)
        true_effect = -0.08 if low_flow == 0 else 0.05
        logit_y = -1.2 + 0.04 * (age - 65) + 0.7 * low_flow + 0.35 * frailty + true_effect * treated * 4
        mortality = int(rng.random() < 1 / (1 + np.exp(-logit_y)))
        outcome_rows.append(
            {
                "stay_id": stay_id,
                "mortality_anytime": mortality,
                "ventilation_hours": float(rng.gamma(2 + low_flow + treated, 28)),
                "rrt_on": int(rng.random() < (0.04 + 0.12 * low_flow + 0.03 * treated)),
                "any_vasopressor": int(rng.random() < (0.25 + 0.35 * low_flow)),
                "nee_mcgkgmin_max": float(max(0, rng.normal(0.08 + 0.18 * low_flow, 0.12))),
                "icu_los_hours": float(rng.gamma(3 + low_flow + treated, 90)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(outcome_rows)


def design_matrix(df: pd.DataFrame, feature_cols: list[str], fit_pipeline: Pipeline | None = None) -> tuple[np.ndarray, list[str], Pipeline]:
    x_df = df[feature_cols].copy()
    x_df = pd.get_dummies(x_df, dummy_na=True)
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    if x_df.shape[1] == 0:
        x_df["intercept"] = 1.0
    if fit_pipeline is None:
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        x = pipe.fit_transform(x_df)
    else:
        pipe = fit_pipeline
        x = pipe.transform(x_df)
    return x, x_df.columns.tolist(), pipe


def fit_classifier(x: np.ndarray, y: np.ndarray, model_type: str, seed: int):
    if len(np.unique(y)) < 2:
        return None, float(np.mean(y))
    if model_type == "logistic":
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
    else:
        model = GradientBoostingClassifier(random_state=seed)
    model.fit(x, y)
    return model, None


def fit_regressor_or_classifier(x: np.ndarray, y: np.ndarray, binary: bool, model_type: str, seed: int):
    if binary:
        return fit_classifier(x, y.astype(int), model_type, seed)
    if np.nanstd(y) < 1e-12:
        return None, float(np.nanmean(y))
    model = GradientBoostingRegressor(random_state=seed)
    model.fit(x, y)
    return model, None


def predict_model(model: Any, constant: float | None, x: np.ndarray, binary: bool = True) -> np.ndarray:
    if model is None:
        return np.repeat(float(constant if constant is not None else 0.0), x.shape[0])
    if binary and hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def crossfit_nuisance(df: pd.DataFrame, feature_cols: list[str], outcome_col: str, treatment_col: str, config: dict[str, Any]) -> pd.DataFrame:
    est = config.get("causal_estimators", {})
    seed = int(config.get("target_trial", {}).get("seed", 42))
    n_splits = int(est.get("n_splits", 2))
    ps_clip = float(est.get("ps_clip", 0.02))
    ps_model = est.get("propensity_model", "gradient_boosting")
    y_model = est.get("outcome_model", "gradient_boosting")

    out = df[["stay_id", treatment_col, outcome_col, "analysis_split"]].copy()
    out["propensity_score"] = np.nan
    out["mu0"] = np.nan
    out["mu1"] = np.nan

    y = df[outcome_col].astype(float).to_numpy()
    a = df[treatment_col].astype(int).to_numpy()
    binary = sorted(pd.Series(y).dropna().unique().tolist()) in [[0.0, 1.0], [0], [1]]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if len(np.unique(a)) > 1 else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_iter = splitter.split(df, a) if len(np.unique(a)) > 1 else splitter.split(df)

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        x_train, cols, pipe = design_matrix(train, feature_cols)
        x_test, _, _ = design_matrix(pd.get_dummies(test[feature_cols], dummy_na=True).reindex(columns=cols, fill_value=np.nan), cols, pipe) if False else (None, [], pipe)
        test_x_df = pd.get_dummies(test[feature_cols].copy(), dummy_na=True).reindex(columns=cols, fill_value=np.nan)
        x_test = pipe.transform(test_x_df.replace([np.inf, -np.inf], np.nan))
        ps_fit, ps_const = fit_classifier(x_train, train[treatment_col].astype(int).to_numpy(), ps_model, seed + fold)
        ps = predict_model(ps_fit, ps_const, x_test, binary=True)
        out.loc[out.index[test_idx], "propensity_score"] = np.clip(ps, ps_clip, 1 - ps_clip)
        for arm in [0, 1]:
            arm_train = train[train[treatment_col].astype(int) == arm]
            if arm_train.empty:
                pred = np.repeat(float(np.nanmean(train[outcome_col])), len(test))
            else:
                x_arm, arm_cols, arm_pipe = design_matrix(arm_train, feature_cols)
                test_arm_df = pd.get_dummies(test[feature_cols].copy(), dummy_na=True).reindex(columns=arm_cols, fill_value=np.nan)
                x_arm_test = arm_pipe.transform(test_arm_df.replace([np.inf, -np.inf], np.nan))
                fit, const = fit_regressor_or_classifier(x_arm, arm_train[outcome_col].astype(float).to_numpy(), binary, y_model, seed + fold + arm)
                pred = predict_model(fit, const, x_arm_test, binary=binary)
            out.loc[out.index[test_idx], f"mu{arm}"] = pred

    a = out[treatment_col].astype(int).to_numpy()
    yy = out[outcome_col].astype(float).to_numpy()
    e = np.clip(out["propensity_score"].to_numpy(dtype=float), ps_clip, 1 - ps_clip)
    mu0 = out["mu0"].to_numpy(dtype=float)
    mu1 = out["mu1"].to_numpy(dtype=float)
    out["aipw_score_ate"] = mu1 - mu0 + a * (yy - mu1) / e - (1 - a) * (yy - mu0) / (1 - e)
    n1 = max(float(a.sum()), 1.0)
    out["att_y1_component"] = a * yy / n1
    out["att_y0_component"] = (a * mu0 + (1 - a) * e / (1 - e) * (yy - mu0)) / n1
    out["overlap_weight"] = np.where(a == 1, 1 - e, e)
    out["iptw_weight"] = np.where(a == 1, 1 / e, 1 / (1 - e))
    return out


def summarize_effect(df: pd.DataFrame, outcome_col: str, treatment_col: str = "transfused") -> dict[str, Any]:
    a = df[treatment_col].astype(int).to_numpy()
    y = df[outcome_col].astype(float).to_numpy()
    e = np.clip(df["propensity_score"].astype(float).to_numpy(), 1e-4, 1 - 1e-4)
    mu0 = df["mu0"].astype(float).to_numpy()
    mu1 = df["mu1"].astype(float).to_numpy()
    ate_aipw = float(np.nanmean(df["aipw_score_ate"]))
    treated_mean = float(np.nanmean(y[a == 1])) if (a == 1).any() else np.nan
    control_mean = float(np.nanmean(y[a == 0])) if (a == 0).any() else np.nan
    n1 = max(float(a.sum()), 1.0)
    att = float(np.nansum(a * y) / n1 - np.nansum(a * mu0 + (1 - a) * e / (1 - e) * (y - mu0)) / n1)
    ow_t = np.where(a == 1, 1 - e, 0)
    ow_c = np.where(a == 0, e, 0)
    ow_eff = weighted_mean(y, ow_t) - weighted_mean(y, ow_c)
    iptw_eff = weighted_mean(y, np.where(a == 1, 1 / e, 0)) - weighted_mean(y, np.where(a == 0, 1 / (1 - e), 0))
    return {
        "n": int(len(df)),
        "n_treated": int(a.sum()),
        "n_control": int((1 - a).sum()),
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "unadjusted_diff": treated_mean - control_mean if np.isfinite(treated_mean) and np.isfinite(control_mean) else np.nan,
        "aipw_ate": ate_aipw,
        "aipw_att": att,
        "overlap_weighted_diff": float(ow_eff),
        "iptw_diff": float(iptw_eff),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any() or weights[mask].sum() <= 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(np.mean(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def candidate_feature_columns(df: pd.DataFrame, outcome_cols: list[str]) -> list[str]:
    exclude = {
        "stay_id",
        "subject_id",
        "t0",
        "transfused",
        "analysis_split",
        "source",
        "treatment_source",
        "treatment_time",
        "eligibility_rule",
        "eligible",
        "first_observed_time",
        "pre_observation_minutes",
        "_fallback_treatment_time",
    } | set(outcome_cols)
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("post_") or col.endswith("_post_mean"):
            continue
        if col.startswith("_") or col.endswith("_n"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == object:
            cols.append(col)
    return cols


def rule_mask(df: pd.DataFrame, rule: list[dict[str, Any]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for item in rule:
        col = item["feature"]
        op = item["op"]
        threshold = float(item["threshold"])
        x = pd.to_numeric(df[col], errors="coerce")
        if op == "<=":
            mask &= x <= threshold
        elif op == ">":
            mask &= x > threshold
        else:
            raise ValueError(f"Unsupported op: {op}")
    return mask.fillna(False)


def rule_to_text(rule: list[dict[str, Any]]) -> str:
    return " AND ".join(f"{r['feature']} {r['op']} {float(r['threshold']):.4g}" for r in rule)
