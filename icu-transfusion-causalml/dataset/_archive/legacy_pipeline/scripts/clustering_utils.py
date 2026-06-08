from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - runtime guard
    raise RuntimeError("pyarrow is required to run clustering scripts.") from exc


BIN_MINUTES = 5


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "src").exists() and (parent / "outputs").exists():
            return parent
    return start


def get_dataset(root: Path) -> ds.Dataset:
    env_dir = os.getenv("PARQUET_DIR")
    if env_dir:
        parquet_dir = Path(env_dir)
    else:
        cand_primary = root / "outputs" / "engineered_features"
        cand_fallback = root / "outputs" / "intermediate" / "engineered_features"
        parquet_dir = cand_primary if cand_primary.exists() else cand_fallback
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet dir not found: {parquet_dir.resolve()}")
    return ds.dataset(str(parquet_dir), format="parquet", partitioning="hive")


def select_feature_cols(
    dataset: ds.Dataset,
    exclude_cols: Iterable[str],
    drop_imputed: bool = True,
) -> List[str]:
    exclude = set(exclude_cols)
    cols = []
    for field in dataset.schema:
        name = field.name
        if name in exclude:
            continue
        if drop_imputed and name.endswith("_imputed"):
            continue
        if pa.types.is_boolean(field.type) or pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            cols.append(name)
    return cols


def scan_t0_map(
    dataset: ds.Dataset,
    id_col: str,
    tbin_col: str,
    flag_col: str,
    batch_size: int,
    progress_every: int = 50,
    progress_label: str = "scan_t0",
) -> Tuple[Dict[int, int], List[int]]:
    t0_map: Dict[int, int] = {}
    stay_ids: set[int] = set()
    start = time.time()
    batches = 0
    rows_total = 0
    scanner = dataset.scanner(columns=[id_col, tbin_col, flag_col], batch_size=batch_size)
    for batch in scanner.to_batches():
        batches += 1
        df = batch.to_pandas()
        rows_total += len(df)
        if df.empty:
            continue
        stay_ids.update(df[id_col].dropna().unique().tolist())
        if flag_col not in df.columns:
            raise ValueError(f"Missing column: {flag_col}")
        df_flag = df[df[flag_col].fillna(0) > 0].copy()
        if df_flag.empty:
            continue
        df_flag[tbin_col] = pd.to_numeric(df_flag[tbin_col], errors="coerce")
        df_flag = df_flag.dropna(subset=[id_col, tbin_col])
        grouped = df_flag.groupby(id_col, as_index=True)[tbin_col].min()
        for stay_id, t0 in grouped.items():
            t0_int = int(t0)
            prev = t0_map.get(stay_id)
            if prev is None or t0_int < prev:
                t0_map[stay_id] = t0_int
        if progress_every and batches % progress_every == 0:
            elapsed = time.time() - start
            print(
                f"[{progress_label}] batches={batches} rows={rows_total} stays={len(stay_ids)} "
                f"transfused={len(t0_map)} elapsed={elapsed:.1f}s"
            )
    return t0_map, sorted(stay_ids)


def iter_window_batches(
    dataset: ds.Dataset,
    t0_map: Dict[int, int],
    stay_to_idx: Dict[int, int],
    feature_cols: List[str],
    id_col: str,
    tbin_col: str,
    pre_bins: int,
    post_bins: int,
    batch_size: int,
    progress_every: int = 50,
    progress_label: str = "window",
):
    seq_len = pre_bins + post_bins
    pre_minutes = pre_bins * BIN_MINUTES
    post_minutes = post_bins * BIN_MINUTES

    cols = [id_col, tbin_col] + feature_cols
    scanner = dataset.scanner(columns=cols, batch_size=batch_size)
    start = time.time()
    batches = 0
    rows_total = 0
    kept_total = 0
    for batch in scanner.to_batches():
        batches += 1
        df = batch.to_pandas()
        rows_total += len(df)
        if df.empty:
            continue
        t0_series = df[id_col].map(t0_map)
        mask = t0_series.notna()
        if not mask.any():
            continue

        df = df.loc[mask].copy()
        t0_vals = t0_series[mask].astype(int)

        tbin = pd.to_numeric(df[tbin_col], errors="coerce")
        df = df.loc[tbin.notna()].copy()
        tbin = tbin.loc[tbin.notna()].astype(int)
        t0_vals = t0_vals.loc[tbin.index]

        rel = ((tbin - t0_vals) // BIN_MINUTES) + pre_bins
        in_window = (rel >= 0) & (rel < seq_len)
        if not in_window.any():
            continue

        df = df.loc[in_window].copy()
        kept_total += int(in_window.sum())
        rel = rel.loc[in_window].astype(int)
        t0_vals = t0_vals.loc[in_window]
        tbin = tbin.loc[in_window]

        stay_idx = df[id_col].map(stay_to_idx).astype(int).to_numpy()
        rel_idx = rel.to_numpy()
        values = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        is_pre = (tbin < t0_vals).to_numpy()

        if progress_every and batches % progress_every == 0:
            elapsed = time.time() - start
            print(
                f"[{progress_label}] batches={batches} rows={rows_total} kept={kept_total} elapsed={elapsed:.1f}s"
            )
        yield stay_idx, rel_idx, values, is_pre


def update_metrics_json(path: Path, key: str, payload: dict) -> None:
    data = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                data = {}
    data[key] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
