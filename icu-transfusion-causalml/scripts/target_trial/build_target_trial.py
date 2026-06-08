from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import (  # noqa: E402
    candidate_feature_columns,
    load_trial_config,
    make_synthetic_longitudinal,
    output_root,
    read_table,
    save_table,
    stable_split_id,
    write_json,
)


TABLE_SUFFIXES = {".parquet", ".csv", ".pkl", ".pickle"}
BIN_TIME_COLUMNS = {"bin_idx", "bin", "time_bin", "time_idx", "tbin", "charttime_bin"}


def _path_from_config(config: dict[str, Any], key: str, default: str) -> Path:
    value = config.get("paths", {}).get(key, default)
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate and candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _table_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES:
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES)


def _longitudinal_candidates(config: dict[str, Any]) -> list[Path]:
    processed = _path_from_config(config, "processed_dir", "outputs/causal_inference/processed")
    timegrid = _path_from_config(config, "timegrid_dir", "dataset/timegrid_features")
    priority = [
        processed / "raw_temporal.parquet",
        REPO_ROOT / "outputs" / "processed" / "raw_temporal.parquet",
        REPO_ROOT / "outputs" / "causal_inference" / "processed" / "raw_temporal.parquet",
    ]
    candidates = [p for p in priority if p.exists()]
    candidates.extend(_table_files(timegrid))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _outcome_candidates(config: dict[str, Any]) -> list[Path]:
    processed = _path_from_config(config, "processed_dir", "outputs/causal_inference/processed")
    outcomes_file = _path_from_config(config, "outcomes_file", "dataset/outputs_outcomes/outcomes_by_stay_full.csv")
    priority = [
        processed / "outcomes.parquet",
        REPO_ROOT / "outputs" / "processed" / "outcomes.parquet",
        REPO_ROOT / "outputs" / "causal_inference" / "processed" / "outcomes.parquet",
        outcomes_file,
    ]
    return [p for p in priority if p.exists()]


def _treatment_candidates(config: dict[str, Any]) -> list[Path]:
    processed = _path_from_config(config, "processed_dir", "outputs/causal_inference/processed")
    priority = [
        processed / "treatment.parquet",
        processed / "cohort.parquet",
        REPO_ROOT / "outputs" / "processed" / "treatment.parquet",
        REPO_ROOT / "outputs" / "processed" / "cohort.parquet",
        REPO_ROOT / "outputs" / "causal_inference" / "processed" / "treatment.parquet",
        REPO_ROOT / "outputs" / "causal_inference" / "processed" / "cohort.parquet",
    ]
    return [p for p in priority if p.exists()]


def _normalize_longitudinal_columns(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    trial = config.get("target_trial", {})
    tt = trial.get("time_zero", {})
    elig = trial.get("eligibility", {})
    trt = trial.get("treatment", {})
    columns_cfg = config.get("columns", {})

    id_col = tt.get("id_col", "stay_id")
    subject_col = tt.get("subject_col", "subject_id")
    time_col = tt.get("time_col", "time_min")
    hb_col = elig.get("hemoglobin_col", "hemoglobin")
    event_col = trt.get("event_col", "rbc_transfusion_flag")

    normalized = df.copy()
    notes: dict[str, Any] = {}

    id_source = _first_column(normalized, [id_col, columns_cfg.get("id", ""), "stay_id", "icustay_id", "icu_stay_id", "stay"])
    if id_source and id_source != id_col:
        normalized[id_col] = normalized[id_source]
        notes["id_alias"] = id_source

    subject_source = _first_column(normalized, [subject_col, columns_cfg.get("subject_id", ""), "subject_id", "patient_id"])
    if subject_source and subject_source != subject_col:
        normalized[subject_col] = normalized[subject_source]
        notes["subject_alias"] = subject_source

    time_source = _first_column(
        normalized,
        [
            time_col,
            columns_cfg.get("time", ""),
            "time_min",
            "time_minutes",
            "minutes",
            "minute",
            "offset_min",
            "offset_minutes",
            "minutes_from_t0",
            "relative_time_min",
            "relative_time_minutes",
            "bin_start_min",
            "bin_start_minutes",
            "tbin",
            "charttime_bin",
            "bin_idx",
            "bin",
            "time_bin",
            "time_idx",
            "chartoffset",
            "offset",
        ],
    )
    if time_source:
        values = pd.to_numeric(normalized[time_source], errors="coerce")
        if time_source.lower() in BIN_TIME_COLUMNS:
            step = float(config.get("preprocessing", {}).get("time_step_minutes", 60))
            values = values * step
            notes["time_alias"] = f"{time_source}*{step:g}"
        elif time_source != time_col:
            notes["time_alias"] = time_source
        normalized[time_col] = values
    else:
        hour_source = _first_column(
            normalized,
            [
                "time_hour",
                "time_hours",
                "hours",
                "hour",
                "offset_hour",
                "offset_hours",
                "hours_from_admission",
                "hour_from_admission",
                "relative_time_hours",
            ],
        )
        if hour_source:
            normalized[time_col] = pd.to_numeric(normalized[hour_source], errors="coerce") * 60.0
            notes["time_alias"] = f"{hour_source}*60"
        else:
            datetime_source = _first_column(normalized, ["charttime", "timestamp", "time", "datetime"])
            if datetime_source and id_col in normalized.columns:
                dt = pd.to_datetime(normalized[datetime_source], errors="coerce")
                first_dt = dt.groupby(normalized[id_col]).transform("min")
                normalized[time_col] = (dt - first_dt).dt.total_seconds() / 60.0
                notes["time_alias"] = f"{datetime_source}-first_observed"

    hb_source = _first_column(
        normalized,
        [
            hb_col,
            "hemoglobin",
            "haemoglobin",
            "hb",
            "hgb",
            "hemoglobin_mean",
            "hemoglobin_last",
            "hb_mean",
            "hgb_mean",
        ],
    )
    if hb_source and hb_source != hb_col:
        normalized[hb_col] = normalized[hb_source]
        notes["hemoglobin_alias"] = hb_source

    event_source = _first_column(
        normalized,
        [
            event_col,
            "rbc_transfusion_flag",
            "rbc_transfusion",
            "transfusion_flag",
            "transfusion",
            "any_rbc_transfusion",
            "packed_rbc",
            "prbc",
        ],
    )
    if event_source and event_source != event_col:
        normalized[event_col] = normalized[event_source]
        notes["event_alias"] = event_source

    required = [id_col, time_col, hb_col]
    missing = [col for col in required if col not in normalized.columns]
    return normalized, notes, missing


def _normalize_treatment_columns(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    tt = config.get("target_trial", {}).get("time_zero", {})
    trt = config.get("target_trial", {}).get("treatment", {})
    columns_cfg = config.get("columns", {})
    id_col = tt.get("id_col", "stay_id")
    fallback_col = trt.get("fallback_treatment_col", "transfused")
    fallback_time_col = trt.get("fallback_treatment_time_col", "t0")

    out = df.copy()
    id_source = _first_column(out, [id_col, columns_cfg.get("id", ""), "stay_id", "icustay_id", "icu_stay_id", "stay"])
    if id_source and id_source != id_col:
        out[id_col] = out[id_source]

    treatment_source = _first_column(
        out,
        [
            fallback_col,
            columns_cfg.get("treatment_name", ""),
            columns_cfg.get("treatment", ""),
            "transfused",
            "treated",
            "treatment",
            "A",
            "rbc_transfusion_flag",
            "rbc_transfusion",
            "any_rbc_transfusion",
        ],
    )
    if treatment_source and treatment_source != fallback_col:
        out[fallback_col] = out[treatment_source]

    time_source = _first_column(
        out,
        [
            fallback_time_col,
            "t0",
            "t0_transf",
            "treatment_time",
            "first_transfusion_time",
            "first_rbc_time",
            "pseudo_t0",
        ],
    )
    if time_source and time_source != fallback_time_col:
        out[fallback_time_col] = out[time_source]

    missing = [col for col in [id_col, fallback_col] if col not in out.columns]
    return out, missing


def _merge_imported_treatment(longitudinal: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    tt = config.get("target_trial", {}).get("time_zero", {})
    trt = config.get("target_trial", {}).get("treatment", {})
    id_col = tt.get("id_col", "stay_id")
    fallback_col = trt.get("fallback_treatment_col", "transfused")
    fallback_time_col = trt.get("fallback_treatment_time_col", "t0")
    if fallback_col in longitudinal.columns:
        return longitudinal, {}

    for path in _treatment_candidates(config):
        try:
            treatment, missing = _normalize_treatment_columns(read_table(path), config)
        except Exception:
            continue
        if missing:
            continue
        keep = [id_col, fallback_col]
        if fallback_time_col in treatment.columns:
            keep.append(fallback_time_col)
        if "pseudo_t0" in treatment.columns and "pseudo_t0" not in keep:
            keep.append("pseudo_t0")
        imported = treatment[keep].drop_duplicates(id_col)
        merged = longitudinal.merge(imported, on=id_col, how="left")
        return merged, {"imported_treatment_path": str(path), "imported_treatment_columns": keep}
    return longitudinal, {}


def _format_source_errors(errors: list[dict[str, Any]]) -> str:
    lines = []
    for item in errors[:12]:
        columns = ", ".join(item.get("columns", [])[:25])
        lines.append(f"- {item['path']}: missing={item.get('missing', [])}; columns=[{columns}]")
    extra = len(errors) - len(lines)
    if extra > 0:
        lines.append(f"- ... {extra} more candidate tables not shown")
    return "\n".join(lines)


def load_longitudinal_and_outcomes(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trial = config.get("target_trial", {})
    allow_synth = bool(trial.get("allow_synthetic_fallback", True))
    metadata: dict[str, Any] = {"synthetic": False}
    source_errors: list[dict[str, Any]] = []
    source_path: Path | None = None
    longitudinal: pd.DataFrame | None = None
    normalization_notes: dict[str, Any] = {}

    for candidate in _longitudinal_candidates(config):
        try:
            raw = read_table(candidate)
            normalized, notes, missing = _normalize_longitudinal_columns(raw, config)
            if not missing:
                source_path = candidate
                longitudinal = normalized
                normalization_notes = notes
                break
            source_errors.append({"path": str(candidate), "missing": missing, "columns": [str(c) for c in raw.columns]})
        except Exception as exc:
            source_errors.append({"path": str(candidate), "missing": ["read_error"], "columns": [str(exc)]})

    outcome_path = next(iter(_outcome_candidates(config)), None)

    if longitudinal is None or source_path is None or outcome_path is None:
        if source_errors:
            details = _format_source_errors(source_errors)
            raise RuntimeError(
                "No usable longitudinal table found for target-trial emulation. "
                "The table must contain stay_id, a time column, and hemoglobin before t0. "
                f"Checked candidates:\n{details}"
            )
        if not allow_synth:
            raise FileNotFoundError("No longitudinal/outcome source found and synthetic fallback is disabled.")
        longitudinal, outcomes = make_synthetic_longitudinal(config)
        metadata.update({"synthetic": True, "source": "synthetic"})
        return longitudinal, outcomes, metadata

    try:
        longitudinal, treatment_notes = _merge_imported_treatment(longitudinal, config)
        normalization_notes.update(treatment_notes)
        outcomes = read_table(outcome_path)
        metadata.update(
            {
                "source": "real_or_imported",
                "longitudinal_path": str(source_path),
                "outcome_path": str(outcome_path),
                "normalization_notes": normalization_notes,
            }
        )
        return longitudinal, outcomes, metadata
    except Exception as exc:
        if not allow_synth:
            raise
        longitudinal, outcomes = make_synthetic_longitudinal(config)
        metadata.update({"synthetic": True, "source": "synthetic_after_read_error", "read_error": str(exc)})
        return longitudinal, outcomes, metadata


def first_eligible_times(longitudinal: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    elig = config["target_trial"]["eligibility"]
    tt = config["target_trial"]["time_zero"]
    id_col = tt.get("id_col", "stay_id")
    time_col = tt.get("time_col", "time_min")
    subject_col = tt.get("subject_col", "subject_id")
    hb_col = elig.get("hemoglobin_col", "hemoglobin")
    threshold = float(elig.get("hemoglobin_threshold", 8.0))
    min_age = float(elig.get("min_age", 18))
    min_pre_min = float(elig.get("min_pre_hours", 6)) * 60.0
    use_imported_t0 = bool(tt.get("use_imported_t0_if_available", True))

    required = {id_col, time_col, hb_col}
    missing = required - set(longitudinal.columns)
    if missing:
        preview = [str(c) for c in longitudinal.columns[:80]]
        raise RuntimeError(f"Longitudinal table missing target-trial columns: {sorted(missing)}. Available columns: {preview}")

    df = longitudinal.copy()
    if "age" in df.columns:
        df = df[pd.to_numeric(df["age"], errors="coerce") >= min_age]
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[hb_col] = pd.to_numeric(df[hb_col], errors="coerce")
    df = df.dropna(subset=[id_col, time_col, hb_col])

    if use_imported_t0 and "t0" in df.columns and "transfused" in df.columns:
        df["t0"] = pd.to_numeric(df["t0"], errors="coerce")
        df = df.dropna(subset=["t0"])
        agg_spec: dict[str, Any] = {"t0": ("t0", "first"), "first_observed_time": (time_col, "min")}
        if subject_col in df.columns:
            agg_spec[subject_col] = (subject_col, "first")
        first = df.groupby(id_col, as_index=False).agg(**agg_spec)
        first["pre_observation_minutes"] = first["t0"] - first["first_observed_time"]
        first = first[first["pre_observation_minutes"] >= min_pre_min].copy()
        first["eligible"] = 1
        first["eligibility_rule"] = "imported_t0_from_existing_cohort"
        return first

    low = df[df[hb_col] <= threshold].sort_values([id_col, time_col])
    first = low.groupby(id_col, as_index=False).first()
    first = first[[id_col, time_col] + ([subject_col] if subject_col in first.columns else [])].rename(columns={time_col: "t0"})
    starts = df.groupby(id_col, as_index=False)[time_col].min().rename(columns={time_col: "first_observed_time"})
    first = first.merge(starts, on=id_col, how="left")
    first["pre_observation_minutes"] = first["t0"] - first["first_observed_time"]
    first = first[first["pre_observation_minutes"] >= min_pre_min].copy()
    first["eligible"] = 1
    first["eligibility_rule"] = f"{hb_col} <= {threshold:g}"
    return first


def assign_treatment(longitudinal: pd.DataFrame, eligibility: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    trt = config["target_trial"]["treatment"]
    tt = config["target_trial"]["time_zero"]
    id_col = tt.get("id_col", "stay_id")
    time_col = tt.get("time_col", "time_min")
    event_col = trt.get("event_col", "rbc_transfusion_flag")
    fallback_col = trt.get("fallback_treatment_col", "transfused")
    fallback_time_col = trt.get("fallback_treatment_time_col", "t0")
    window_min = float(trt.get("window_hours", 6)) * 60.0
    out = eligibility.copy()

    if event_col in longitudinal.columns:
        events = longitudinal[[id_col, time_col, event_col]].copy()
        events[event_col] = pd.to_numeric(events[event_col], errors="coerce").fillna(0)
        events = events[events[event_col] > 0]
        merged = out[[id_col, "t0"]].merge(events, on=id_col, how="left")
        merged = merged[(merged[time_col] >= merged["t0"]) & (merged[time_col] <= merged["t0"] + window_min)]
        tx = merged.groupby(id_col, as_index=False)[time_col].min().rename(columns={time_col: "treatment_time"})
        out = out.merge(tx, on=id_col, how="left")
        out["transfused"] = out["treatment_time"].notna().astype(int)
        out["treatment_source"] = event_col
        return out

    fallback_cols = [c for c in [id_col, fallback_col] if c in longitudinal.columns]
    if fallback_col in fallback_cols:
        tmp = longitudinal[fallback_cols].drop_duplicates(id_col)
        fallback_time_merge_col = "_fallback_treatment_time"
        if fallback_time_col in longitudinal.columns:
            times = longitudinal[[id_col, fallback_time_col]].drop_duplicates(id_col).rename(columns={fallback_time_col: fallback_time_merge_col})
            tmp = tmp.merge(times, on=id_col, how="left")
        out = out.merge(tmp, on=id_col, how="left")
        out["transfused"] = pd.to_numeric(out[fallback_col], errors="coerce").fillna(0).astype(int)
        if fallback_time_merge_col in out.columns:
            tx_time = pd.to_numeric(out[fallback_time_merge_col], errors="coerce")
            out.loc[(out["transfused"] == 1) & ((tx_time < out["t0"]) | (tx_time > out["t0"] + window_min)), "transfused"] = 0
            out["treatment_time"] = tx_time.where(out["transfused"] == 1, np.nan)
        out["treatment_source"] = fallback_col
        return out

    raise RuntimeError("No transfusion event or fallback treatment column found.")


def aggregate_pre_t0(longitudinal: pd.DataFrame, trial: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    tt = config["target_trial"]["time_zero"]
    feat = config["target_trial"]["features"]
    elig = config["target_trial"]["eligibility"]
    id_col = tt.get("id_col", "stay_id")
    time_col = tt.get("time_col", "time_min")
    lookback_min = float(elig.get("lookback_hours", 48)) * 60.0
    temporal = [c for c in feat.get("temporal", []) if c in longitudinal.columns]
    static = [c for c in feat.get("static", []) if c in longitudinal.columns]
    aggregations = feat.get("aggregations", [])

    merged = longitudinal.merge(trial[[id_col, "t0"]], on=id_col, how="inner")
    merged[time_col] = pd.to_numeric(merged[time_col], errors="coerce")
    pre = merged[(merged[time_col] < merged["t0"]) & (merged[time_col] >= merged["t0"] - lookback_min)].copy()
    rows = []
    for stay_id, group in pre.groupby(id_col):
        row: dict[str, Any] = {id_col: stay_id}
        group = group.sort_values(time_col)
        for col in static:
            values = group[col].dropna()
            row[col] = values.iloc[-1] if len(values) else np.nan
        for col in temporal:
            x = pd.to_numeric(group[col], errors="coerce")
            t = pd.to_numeric(group[time_col], errors="coerce") / 60.0
            finite = x.notna() & t.notna()
            vals = x[finite]
            tt_vals = t[finite]
            prefix = col
            if "mean" in aggregations:
                row[f"{prefix}_mean"] = float(vals.mean()) if len(vals) else np.nan
            if "median" in aggregations:
                row[f"{prefix}_median"] = float(vals.median()) if len(vals) else np.nan
            if "min" in aggregations:
                row[f"{prefix}_min"] = float(vals.min()) if len(vals) else np.nan
            if "max" in aggregations:
                row[f"{prefix}_max"] = float(vals.max()) if len(vals) else np.nan
            if "std" in aggregations:
                row[f"{prefix}_std"] = float(vals.std(ddof=0)) if len(vals) else np.nan
            if "first" in aggregations:
                row[f"{prefix}_first"] = float(vals.iloc[0]) if len(vals) else np.nan
            if "last" in aggregations:
                row[f"{prefix}_last"] = float(vals.iloc[-1]) if len(vals) else np.nan
            if "delta" in aggregations:
                row[f"{prefix}_delta"] = float(vals.iloc[-1] - vals.iloc[0]) if len(vals) > 1 else 0.0
            if "n" in aggregations:
                row[f"{prefix}_n"] = int(len(vals))
            if "slope" in aggregations and len(vals) > 1 and float(tt_vals.max() - tt_vals.min()) > 1e-8:
                row[f"{prefix}_slope"] = float(np.polyfit(tt_vals.to_numpy(), vals.to_numpy(), 1)[0])
            elif "slope" in aggregations:
                row[f"{prefix}_slope"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_target_trial(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    longitudinal, outcomes, metadata = load_longitudinal_and_outcomes(config)
    eligibility = first_eligible_times(longitudinal, config)
    treatment = assign_treatment(longitudinal, eligibility, config)
    features = aggregate_pre_t0(longitudinal, treatment, config)

    id_col = config["target_trial"]["time_zero"].get("id_col", "stay_id")
    primary = config["target_trial"]["outcomes"].get("primary", "mortality_anytime")
    secondary = [c for c in config["target_trial"]["outcomes"].get("secondary", []) if c in outcomes.columns]
    keep_outcomes = [id_col] + [c for c in [primary] + secondary if c in outcomes.columns]
    trial = treatment.merge(features, on=id_col, how="inner").merge(outcomes[keep_outcomes].drop_duplicates(id_col), on=id_col, how="left")
    trial = trial.dropna(subset=[primary, "transfused"]).copy()
    n_treated = int(trial["transfused"].sum()) if "transfused" in trial.columns else 0
    n_control = int((1 - trial["transfused"]).sum()) if "transfused" in trial.columns else 0
    if n_treated == 0 or n_control == 0:
        save_table(trial, proc / "trial_dataset.parquet")
        raise RuntimeError(
            "Target-trial dataset has no treatment contrast after anchoring "
            f"(treated={n_treated}, controls={n_control}). "
            "Use a longitudinal source with both transfused and control stays, or import the matched control cohort from the paper pipeline."
        )

    split = config["target_trial"].get("split", {})
    seed = int(config["target_trial"].get("seed", 42))
    discovery_fraction = float(split.get("discovery_fraction", 0.7))
    split_col = "subject_id" if "subject_id" in trial.columns else id_col
    trial["analysis_split"] = [stable_split_id(v, seed, discovery_fraction) for v in trial[split_col]]
    trial["source"] = metadata.get("source", "unknown")

    outcome_cols = [primary] + secondary
    feature_cols = candidate_feature_columns(trial, outcome_cols)
    save_table(eligibility, proc / "eligibility.parquet")
    save_table(treatment, proc / "treatment_assignment.parquet")
    save_table(trial, proc / "trial_dataset.parquet")
    write_json(
        proc / "target_trial_metadata.json",
        {
            **metadata,
            "n_eligible": int(len(eligibility)),
            "n_trial": int(len(trial)),
            "n_treated": n_treated,
            "n_control": n_control,
            "primary_outcome": primary,
            "feature_columns": feature_cols,
            "outcome_columns": outcome_cols,
            "design": "target_trial_emulation",
            "warning": "Use only as confirmatory if all included features are pre-t0.",
        },
    )
    print(f"target_trial_complete n={len(trial)} treated={n_treated} controls={n_control}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-trial emulation dataset with strict pre-t0 features.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    build_target_trial(args.config_dir)


if __name__ == "__main__":
    main()
