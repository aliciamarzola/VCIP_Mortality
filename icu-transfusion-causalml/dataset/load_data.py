from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class ProjectData:
    temporal: pd.DataFrame
    outcomes: pd.DataFrame
    metadata: Dict[str, Any]


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / "configs").exists() and (parent / "dataset").exists():
            return parent
    return Path.cwd()


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_dir: Path | str = "configs") -> Dict[str, Any]:
    root = repo_root()
    config_path = Path(config_dir)
    if not config_path.is_absolute():
        config_path = root / config_path

    config: Dict[str, Any] = {}
    for name in ["config.yaml", "paths.yaml", "model_config.yaml", "causal_config.yaml"]:
        path = config_path / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            deep_update(config, payload)
    return config


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def _candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    suffixes = {".csv", ".parquet", ".pkl", ".pickle"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _valid_parquet_files(files: Iterable[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    valid: list[Path] = []
    invalid: list[dict[str, Any]] = []
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return [], [{"path": str(p), "error": f"pyarrow unavailable: {exc}"} for p in files]

    for path in files:
        try:
            pq.ParquetFile(path)
            valid.append(path)
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    return valid, invalid


def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_table_collection(path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    files = _candidate_files(path)
    metadata: Dict[str, Any] = {"path": str(path), "n_files": len(files), "invalid_files": []}
    if not files:
        return pd.DataFrame(), metadata

    parquet_files = [p for p in files if p.suffix.lower() == ".parquet"]
    non_parquet = [p for p in files if p.suffix.lower() != ".parquet"]
    readable = non_parquet[:]
    if parquet_files:
        valid, invalid = _valid_parquet_files(parquet_files)
        readable.extend(valid)
        metadata["invalid_files"] = invalid
        metadata["valid_parquet_files"] = len(valid)

    frames = []
    errors = []
    for file_path in readable:
        try:
            frames.append(read_any_table(file_path))
        except Exception as exc:
            errors.append({"path": str(file_path), "error": str(exc)})
    metadata["read_errors"] = errors

    if not frames:
        return pd.DataFrame(), metadata
    return pd.concat(frames, ignore_index=True, sort=False), metadata


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def standardize_temporal_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    columns_cfg = config.get("columns", {})
    id_col = _find_column(df.columns, [columns_cfg.get("id", "stay_id"), "icustay_id", "icu_stay_id"])
    if id_col and id_col != "stay_id":
        df = df.rename(columns={id_col: "stay_id"})

    subject_col = _find_column(df.columns, [columns_cfg.get("subject_id", "subject_id"), "patient_id"])
    if subject_col and subject_col != "subject_id":
        df = df.rename(columns={subject_col: "subject_id"})

    time_col = _find_column(
        df.columns,
        [columns_cfg.get("time", "time_min"), "time_min", "tbin", "minutes_from_t0", "charttime_bin", "charttime"],
    )
    if time_col is None:
        raise RuntimeError("Could not infer temporal time column.")
    if time_col != "time_min":
        series = df[time_col]
        if pd.api.types.is_numeric_dtype(series):
            multiplier = 60.0 if "hour" in time_col.lower() else 1.0
            df["time_min"] = pd.to_numeric(series, errors="coerce") * multiplier
        else:
            dt = pd.to_datetime(series, errors="coerce")
            if "stay_id" not in df.columns:
                raise RuntimeError("Datetime time columns require stay_id.")
            first_dt = dt.groupby(df["stay_id"]).transform("min")
            df["time_min"] = (dt - first_dt).dt.total_seconds() / 60.0

    return df


def standardize_outcome_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    id_col = _find_column(df.columns, [config.get("columns", {}).get("id", "stay_id"), "icustay_id", "icu_stay_id"])
    if id_col and id_col != "stay_id":
        df = df.rename(columns={id_col: "stay_id"})
    return df


def _logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_data(config: Dict[str, Any]) -> ProjectData:
    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 42))
    n_stays = int(run_cfg.get("synthetic_n_stays", 400))
    step_minutes = int(run_cfg.get("synthetic_time_step_minutes", 60))
    rng = np.random.default_rng(seed)

    stay_ids = np.arange(100000, 100000 + n_stays)
    subject_ids = np.arange(200000, 200000 + n_stays)
    age = rng.integers(18, 90, size=n_stays)
    sex = rng.integers(0, 2, size=n_stays)
    bmi = np.clip(rng.normal(28, 6, size=n_stays), 16, 55)
    latent = rng.normal(0, 1, size=n_stays)
    phenotype_true = np.digitize(latent, [-0.5, 0.8])

    baseline_hb = np.clip(rng.normal(9.2 - 0.6 * latent, 1.2, size=n_stays), 5.5, 14.5)
    baseline_lactate = np.clip(rng.lognormal(0.25 + 0.35 * latent, 0.45, size=n_stays), 0.5, 9.0)
    baseline_sofa = np.clip(np.round(rng.normal(6 + 2.2 * latent, 2.0, size=n_stays)), 0, 20)
    treatment_prob = _logit(-0.8 - 0.55 * baseline_hb + 0.35 * baseline_lactate + 0.18 * baseline_sofa)
    treatment = rng.binomial(1, np.clip(treatment_prob, 0.05, 0.9))

    treated_offsets = rng.integers(30 * 60, 72 * 60 + 1, size=n_stays) // step_minutes * step_minutes
    t0 = treated_offsets.astype(float)
    max_time = 96 * 60
    time_points = np.arange(0, max_time + step_minutes, step_minutes)

    rows = []
    for i, stay_id in enumerate(stay_ids):
        drift = rng.normal(0, 0.03)
        for t in time_points:
            rel_hours = (t - t0[i]) / 60.0
            pre_hours = t / 24.0
            hb = baseline_hb[i] - 0.025 * pre_hours * 24 + 0.12 * np.sin(t / 480) + rng.normal(0, 0.25)
            lactate = baseline_lactate[i] + 0.018 * rel_hours + rng.normal(0, 0.25)
            creatinine = np.clip(0.8 + 0.25 * latent[i] + 0.005 * t / 60 + rng.normal(0, 0.12), 0.25, 8)
            platelets = np.clip(230 - 35 * latent[i] - 0.4 * t / 60 + rng.normal(0, 18), 20, 600)
            sofa = np.clip(baseline_sofa[i] + 0.03 * rel_hours + rng.normal(0, 0.8), 0, 24)
            heart_rate = np.clip(86 + 7 * latent[i] + 1.4 * lactate + rng.normal(0, 7), 40, 170)
            mbp = np.clip(78 - 4 * latent[i] - 1.3 * lactate + rng.normal(0, 6), 35, 130)
            resp_rate = np.clip(18 + 1.4 * latent[i] + rng.normal(0, 3), 6, 42)
            spo2 = np.clip(96 - 1.2 * latent[i] + rng.normal(0, 2), 70, 100)
            vasopressor = int(rng.random() < _logit(-1.0 + 0.5 * latent[i] + 0.12 * sofa))
            ventilation = int(rng.random() < _logit(-1.3 + 0.35 * latent[i] + 0.1 * sofa))
            rbc_event = int(treatment[i] == 1 and t == t0[i])
            rows.append(
                {
                    "subject_id": int(subject_ids[i]),
                    "stay_id": int(stay_id),
                    "time_min": float(t),
                    "age": int(age[i]),
                    "sex": int(sex[i]),
                    "bmi": float(bmi[i]),
                    "hemoglobin": float(hb + drift * rel_hours),
                    "lactate": float(lactate),
                    "creatinine": float(creatinine),
                    "platelets": float(platelets),
                    "heart_rate": float(heart_rate),
                    "mbp": float(mbp),
                    "resp_rate": float(resp_rate),
                    "spo2": float(spo2),
                    "sofa": float(sofa),
                    "any_vasopressor": int(vasopressor),
                    "ventilation_on": int(ventilation),
                    "rbc_transfusion_flag": int(rbc_event),
                    "true_phenotype": int(phenotype_true[i]),
                }
            )

    temporal = pd.DataFrame(rows)

    baseline_risk = _logit(-3.1 + 0.03 * age + 0.28 * baseline_lactate + 0.16 * baseline_sofa + 0.35 * latent)
    true_ite = np.where(phenotype_true == 0, -0.08, np.where(phenotype_true == 1, 0.01, 0.09))
    y0_prob = np.clip(baseline_risk, 0.02, 0.85)
    y1_prob = np.clip(y0_prob + true_ite, 0.01, 0.95)
    observed_prob = np.where(treatment == 1, y1_prob, y0_prob)
    mortality = rng.binomial(1, observed_prob)

    outcomes = pd.DataFrame(
        {
            "stay_id": stay_ids.astype(int),
            "mortality_anytime": mortality.astype(int),
            "vm_time_hours": np.clip(rng.normal(35 + 18 * latent + 18 * mortality, 18), 0, 300),
            "ventilation_hours": np.clip(rng.normal(28 + 14 * latent + 15 * mortality, 14), 0, 260),
            "rrt_on": rng.binomial(1, np.clip(_logit(-2.5 + 0.45 * latent + 0.12 * baseline_sofa), 0.01, 0.6)),
            "any_vasopressor": rng.binomial(1, np.clip(_logit(-1.1 + 0.55 * latent + 0.12 * baseline_sofa), 0.02, 0.9)),
            "nee_mcgkgmin_max": np.clip(rng.lognormal(-2.8 + 0.4 * latent, 0.7), 0, 2.5),
            "icu_los_hours": np.clip(rng.normal(140 + 30 * latent + 55 * mortality, 36), 24, 600),
            "transfused": treatment.astype(int),
            "synthetic_y0_risk": y0_prob,
            "synthetic_y1_risk": y1_prob,
            "synthetic_true_ite": true_ite,
            "synthetic_true_phenotype": phenotype_true,
        }
    )

    metadata = {
        "source": "synthetic",
        "synthetic": True,
        "n_stays": int(n_stays),
        "n_temporal_rows": int(len(temporal)),
        "seed": int(seed),
        "note": "Synthetic data are demonstrative and do not represent real clinical estimates.",
    }
    return ProjectData(temporal=temporal, outcomes=outcomes, metadata=metadata)


def load_real_data(config: Dict[str, Any]) -> ProjectData:
    paths = config.get("paths", {})
    timegrid_path = resolve_path(paths.get("timegrid_dir", "dataset/timegrid_features"))
    outcomes_path = resolve_path(paths.get("outcomes_file", "dataset/outputs_outcomes/outcomes_by_stay_full.csv"))

    temporal, temporal_meta = read_table_collection(timegrid_path)
    outcomes = pd.DataFrame()
    outcome_meta: Dict[str, Any] = {"path": str(outcomes_path), "exists": outcomes_path.exists()}
    if outcomes_path.exists():
        outcomes = read_any_table(outcomes_path)

    if not temporal.empty:
        temporal = standardize_temporal_columns(temporal, config)
    if not outcomes.empty:
        outcomes = standardize_outcome_columns(outcomes, config)

    metadata = {
        "source": "real",
        "synthetic": False,
        "timegrid": temporal_meta,
        "outcomes": outcome_meta,
        "n_temporal_rows": int(len(temporal)),
        "n_outcome_rows": int(len(outcomes)),
    }
    return ProjectData(temporal=temporal, outcomes=outcomes, metadata=metadata)


def load_project_data(config: Optional[Dict[str, Any]] = None, source_override: Optional[str] = None) -> ProjectData:
    config = config or load_config()
    run_cfg = config.get("run", {})
    source = source_override or run_cfg.get("data_source", "auto")
    allow_fallback = bool(run_cfg.get("allow_synthetic_fallback", True))

    if source == "synthetic":
        return generate_synthetic_data(config)

    real_data = load_real_data(config)
    has_real = not real_data.temporal.empty and not real_data.outcomes.empty
    invalid_files = real_data.metadata.get("timegrid", {}).get("invalid_files", [])

    if source == "real" and has_real:
        return real_data
    if source == "real" and not has_real:
        raise RuntimeError("Real data were requested, but readable temporal or outcome data were not found.")
    if source == "auto" and has_real and not invalid_files:
        return real_data
    if source == "auto" and has_real and invalid_files and not allow_fallback:
        return real_data

    synthetic = generate_synthetic_data(config)
    synthetic.metadata["fallback_reason"] = {
        "requested_source": source,
        "real_data_available": bool(has_real),
        "invalid_timegrid_files": len(invalid_files),
    }
    return synthetic


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    cfg = load_config()
    data = load_project_data(cfg)
    print(json.dumps(data.metadata, indent=2, default=str))
