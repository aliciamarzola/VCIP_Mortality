from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.load_data import load_config as _load_config  # noqa: E402


def load_config(config_dir: str | Path = "configs") -> Dict[str, Any]:
    return _load_config(config_dir)


def resolve_path(config: Dict[str, Any], key: str, default: str) -> Path:
    value = config.get("paths", {}).get(key, default)
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def processed_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, "processed_dir", "outputs/causal_inference/processed"))


def figures_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, "figures_dir", "outputs/causal_inference/figures"))


def phenotypes_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, "phenotypes_dir", "outputs/causal_inference/phenotypes"))


def causal_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, "causal_dir", "outputs/causal_inference/causal"))


def counterfactual_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, "counterfactual_dir", "outputs/causal_inference/counterfactual"))


def evaluation_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, "evaluation_dir", "outputs/causal_inference/evaluation"))


def logs_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, "logs_dir", "outputs/causal_inference/logs"))


def primary_outcome(config: Dict[str, Any]) -> str:
    return config.get("columns", {}).get("primary_outcome", "mortality_anytime")


def load_feature_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    return read_json(processed_dir(config) / "feature_metadata.json")


def load_processed_frame(config: Dict[str, Any], include_clusters: bool = True) -> pd.DataFrame:
    out_dir = processed_dir(config)
    features = pd.read_parquet(out_dir / "static_features.parquet")
    treatment = pd.read_parquet(out_dir / "treatment.parquet")
    outcomes = pd.read_parquet(out_dir / "outcomes.parquet")
    df = features.merge(treatment, on="stay_id", how="left").merge(outcomes, on="stay_id", how="left")

    if include_clusters:
        clusters_path = phenotypes_dir(config) / "clusters.parquet"
        if clusters_path.exists():
            clusters = pd.read_parquet(clusters_path)
            keep = [c for c in ["stay_id", "phenotype"] if c in clusters.columns]
            if keep:
                df = df.merge(clusters[keep].drop_duplicates("stay_id"), on="stay_id", how="left")
    return df


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output table format: {path}")


def safe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def mark_demonstrative_if_synthetic(config: Dict[str, Any]) -> bool:
    metadata = read_json(processed_dir(config) / "data_source_metadata.json")
    return bool(metadata.get("synthetic", False) or metadata.get("source") == "synthetic")
