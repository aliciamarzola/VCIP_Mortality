from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def available_columns(df: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    return [col for col in requested if col in df.columns]


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    excluded = set(exclude)
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def select_static_temporal_columns(temporal: pd.DataFrame, config: Dict[str, Any]) -> tuple[list[str], list[str]]:
    variables = config.get("variables", {})
    static_cols = available_columns(temporal, variables.get("static", []))
    temporal_cols = available_columns(temporal, variables.get("temporal", []))

    if not temporal_cols:
        exclude = {"stay_id", "subject_id", "time_min", "rbc_transfusion_flag", "transfused", "true_phenotype"}
        temporal_cols = numeric_columns(temporal, exclude=exclude | set(static_cols))
    return static_cols, temporal_cols


def pre_t0_window(temporal: pd.DataFrame, cohort: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    window_minutes = float(window_hours) * 60.0
    anchors = cohort[["stay_id", "t0"]].drop_duplicates("stay_id")
    merged = temporal.merge(anchors, on="stay_id", how="inner")
    start = merged["t0"] - window_minutes
    mask = (merged["time_min"] >= start) & (merged["time_min"] < merged["t0"])
    out = merged.loc[mask].copy()
    out["time_to_t0_min"] = out["time_min"] - out["t0"]
    return out


def _slope(group: pd.DataFrame, value_col: str) -> float:
    sub = group[["time_to_t0_min", value_col]].dropna()
    if len(sub) < 2:
        return np.nan
    x = sub["time_to_t0_min"].to_numpy(dtype=float)
    y = sub[value_col].to_numpy(dtype=float)
    var_x = np.var(x)
    if var_x <= 0:
        return np.nan
    return float(np.cov(x, y, ddof=0)[0, 1] / var_x)


def aggregate_temporal_features(pre: pd.DataFrame, temporal_cols: list[str]) -> pd.DataFrame:
    rows = []
    for stay_id, group in pre.sort_values(["stay_id", "time_min"]).groupby("stay_id"):
        row: dict[str, Any] = {"stay_id": stay_id, "n_rows_pre": int(len(group))}
        for col in temporal_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"{col}_min"] = float(values.min()) if values.notna().any() else np.nan
            row[f"{col}_max"] = float(values.max()) if values.notna().any() else np.nan
            row[f"{col}_std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
            first = values.dropna().iloc[0] if values.notna().any() else np.nan
            last = values.dropna().iloc[-1] if values.notna().any() else np.nan
            row[f"{col}_last"] = float(last) if pd.notna(last) else np.nan
            row[f"{col}_first_last_diff"] = float(last - first) if pd.notna(first) and pd.notna(last) else np.nan
            mean_abs = abs(row[f"{col}_mean"]) if pd.notna(row[f"{col}_mean"]) else np.nan
            row[f"{col}_variability"] = float(row[f"{col}_std"] / mean_abs) if mean_abs and mean_abs > 0 else np.nan
            row[f"{col}_n_measures"] = int(values.notna().sum())
            row[f"{col}_slope"] = _slope(group, col)
        rows.append(row)
    return pd.DataFrame(rows)


def build_static_feature_table(temporal: pd.DataFrame, pre: pd.DataFrame, cohort: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    static_cols, temporal_cols = select_static_temporal_columns(temporal, config)
    static = temporal.sort_values("time_min").groupby("stay_id", as_index=False)[static_cols].first() if static_cols else pd.DataFrame({"stay_id": cohort["stay_id"]})
    aggregates = aggregate_temporal_features(pre, temporal_cols)
    features = cohort[["stay_id"]].merge(static, on="stay_id", how="left").merge(aggregates, on="stay_id", how="left")
    return features


def build_temporal_tensor(
    pre: pd.DataFrame,
    cohort: pd.DataFrame,
    temporal_cols: list[str],
    window_hours: int,
    step_minutes: int,
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    stay_ids = cohort["stay_id"].drop_duplicates().to_numpy()
    n_steps = max(1, int(np.ceil(window_hours * 60 / step_minutes)))
    tensor = np.full((len(stay_ids), n_steps, len(temporal_cols)), np.nan, dtype=float)
    mask = np.zeros_like(tensor, dtype=np.int8)
    stay_to_idx = {stay_id: idx for idx, stay_id in enumerate(stay_ids)}

    for stay_id, group in pre.groupby("stay_id"):
        if stay_id not in stay_to_idx:
            continue
        idx = stay_to_idx[stay_id]
        t0 = float(cohort.loc[cohort["stay_id"] == stay_id, "t0"].iloc[0])
        start = t0 - window_hours * 60
        bins = np.floor((group["time_min"].to_numpy(dtype=float) - start) / step_minutes).astype(int)
        group = group.assign(_bin=bins)
        group = group[(group["_bin"] >= 0) & (group["_bin"] < n_steps)]
        if group.empty:
            continue
        binned = group.groupby("_bin")[temporal_cols].mean(numeric_only=True)
        for bin_idx, row in binned.iterrows():
            values = row.to_numpy(dtype=float)
            tensor[idx, int(bin_idx), :] = values
            mask[idx, int(bin_idx), :] = np.isfinite(values).astype(np.int8)

    rng = np.random.default_rng(seed)
    train_flag = rng.random(len(stay_ids)) < train_fraction
    if not train_flag.any():
        train_flag[:] = True

    flat_train = tensor[train_flag].reshape(-1, len(temporal_cols))
    medians = np.nanmedian(flat_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    imputed = tensor.copy()
    for i in range(imputed.shape[0]):
        frame = pd.DataFrame(imputed[i], columns=temporal_cols)
        frame = frame.ffill()
        for j, col in enumerate(temporal_cols):
            frame[col] = frame[col].fillna(medians[j])
        imputed[i] = frame.to_numpy(dtype=float)

    flat_train_imp = imputed[train_flag].reshape(-1, len(temporal_cols))
    means = flat_train_imp.mean(axis=0)
    stds = flat_train_imp.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    normalized = (imputed - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)

    ids = pd.DataFrame({"stay_id": stay_ids, "train_split": train_flag.astype(int)})
    metadata = {
        "shape": list(normalized.shape),
        "temporal_columns": temporal_cols,
        "window_hours": int(window_hours),
        "step_minutes": int(step_minutes),
        "normalization": "zscore_train_split",
        "train_fraction": float(train_fraction),
        "medians": dict(zip(temporal_cols, medians.astype(float))),
        "means": dict(zip(temporal_cols, means.astype(float))),
        "stds": dict(zip(temporal_cols, stds.astype(float))),
    }
    return normalized, mask, ids, metadata
