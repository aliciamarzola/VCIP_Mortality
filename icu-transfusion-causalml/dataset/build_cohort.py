from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from .load_data import load_config, load_project_data, read_any_table, resolve_path, write_metadata
from .time_anchor import create_time_anchors


def _processed_dir(config: Dict[str, Any], output_dir: Optional[str] = None) -> Path:
    if output_dir:
        return resolve_path(output_dir)
    return resolve_path(config.get("paths", {}).get("processed_dir", "outputs/causal_inference/processed"))


def filter_adult_first_stays(temporal: pd.DataFrame, anchors: pd.DataFrame, config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_cfg = config.get("cohort", {})
    adult_min_age = float(cohort_cfg.get("adult_min_age", 18))
    if "age" in temporal.columns:
        age_by_stay = temporal.groupby("stay_id", as_index=False)["age"].first()
        anchors = anchors.merge(age_by_stay, on="stay_id", how="left")
        anchors = anchors[anchors["age"].fillna(adult_min_age) >= adult_min_age].copy()

    if bool(cohort_cfg.get("first_icu_stay_only", True)) and {"subject_id", "stay_id"}.issubset(temporal.columns):
        stay_subject = temporal.groupby("stay_id", as_index=False).agg(subject_id=("subject_id", "first"))
        anchors = anchors.merge(stay_subject, on="stay_id", how="left")
        anchors = anchors.sort_values(["subject_id", "min_time", "stay_id"]).drop_duplicates("subject_id", keep="first")

    min_obs = float(cohort_cfg.get("min_observation_hours", 24)) * 60.0
    anchors = anchors[(anchors["max_time"] - anchors["min_time"]) >= min_obs].copy()
    anchors = anchors[(anchors["t0"] - anchors["min_time"]) >= min_obs].copy()
    temporal = temporal[temporal["stay_id"].isin(anchors["stay_id"])].copy()
    return temporal, anchors


def build_cohort(config: Dict[str, Any], source_override: Optional[str] = None, output_dir: Optional[str] = None) -> pd.DataFrame:
    source = source_override or config.get("run", {}).get("data_source", "auto")
    if source == "real":
        return build_real_cohort_duckdb(config, output_dir=output_dir)

    out_dir = _processed_dir(config, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_project_data(config, source_override=source_override)
    temporal = data.temporal.copy()
    outcomes = data.outcomes.copy()
    anchors = create_time_anchors(temporal, outcomes, config)
    temporal, anchors = filter_adult_first_stays(temporal, anchors, config)

    treatment_name = config.get("columns", {}).get("treatment_name", "transfused")
    if treatment_name in outcomes.columns:
        outcomes = outcomes.drop(columns=[treatment_name])
    cohort = anchors.merge(outcomes, on="stay_id", how="left")
    cohort["data_source"] = data.metadata.get("source", "unknown")
    cohort["is_synthetic"] = bool(data.metadata.get("synthetic", False))

    temporal.to_parquet(out_dir / "raw_temporal.parquet", index=False)
    cohort.to_parquet(out_dir / "cohort.parquet", index=False)
    write_metadata(out_dir / "data_source_metadata.json", data.metadata)

    treatment = cohort[["stay_id", "transfused", "t0", "pseudo_t0", "data_source", "is_synthetic"]].copy()
    treatment.to_parquet(out_dir / "treatment.parquet", index=False)

    return cohort


def _timegrid_glob(config: Dict[str, Any]) -> Path:
    timegrid_dir = resolve_path(config.get("paths", {}).get("timegrid_dir", "dataset/timegrid_features"))
    if timegrid_dir.is_file():
        return timegrid_dir
    return timegrid_dir / "*" / "*.parquet"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _progress(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] build_cohort: {message}", flush=True)


def _duckdb_columns(con: duckdb.DuckDBPyConnection, relation: str) -> list[tuple[str, str]]:
    return [(row[0], row[1]) for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]


def _lookup(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _infer_time_expression(time_col: str, time_type: str, alias: str = "tg") -> str:
    col = f"{alias}.{_quote(time_col)}"
    if "TIMESTAMP" in time_type.upper() or "DATE" in time_type.upper():
        return f"(epoch({col}) / 60.0)"
    multiplier = 60.0 if "hour" in time_col.lower() else 1.0
    return f"(CAST({col} AS DOUBLE) * {multiplier})"


def _select_real_columns(columns_with_types: list[tuple[str, str]], config: Dict[str, Any]) -> tuple[list[str], list[str], dict[str, str]]:
    names = [name for name, _ in columns_with_types]
    lookup = {name.lower(): name for name in names}
    alias_map: dict[str, str] = {}

    static_requested = list(config.get("variables", {}).get("static", []))
    static_cols = []
    for col in static_requested:
        source = lookup.get(col.lower())
        if source is None and col == "sex":
            source = lookup.get("gender")
        if source is not None:
            static_cols.append(source)
            alias_map[source] = col

    temporal_cols = []
    for col in config.get("variables", {}).get("temporal", []):
        source = lookup.get(col.lower())
        if source is not None:
            temporal_cols.append(source)
            alias_map[source] = col

    return static_cols, temporal_cols, alias_map


def build_real_cohort_duckdb(config: Dict[str, Any], output_dir: Optional[str] = None) -> pd.DataFrame:
    """Build real-data cohort without loading the full timegrid into pandas."""
    out_dir = _processed_dir(config, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _progress("starting real-data cohort build with DuckDB")
    timegrid_path = _timegrid_glob(config)
    outcomes_path = resolve_path(config.get("paths", {}).get("outcomes_file", "dataset/outputs_outcomes/outcomes_by_stay_full.csv"))
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Missing outcomes file: {outcomes_path}")
    _progress(f"timegrid source: {timegrid_path}")
    _progress(f"outcomes source: {outcomes_path}")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_object_cache=true")
    con.execute("PRAGMA memory_limit='16GB'")
    if os.getenv("DUCKDB_PROGRESS", "1") != "0":
        con.execute("PRAGMA enable_progress_bar=true")
    timegrid_rel = f"read_parquet('{timegrid_path.as_posix()}', union_by_name=true)"

    _progress("inspecting timegrid schema")
    columns_with_types = _duckdb_columns(con, timegrid_rel)
    names = [name for name, _ in columns_with_types]
    stay_col = _lookup(names, [config.get("columns", {}).get("id", "stay_id"), "stay_id", "icustay_id", "icu_stay_id"])
    subject_col = _lookup(names, [config.get("columns", {}).get("subject_id", "subject_id"), "subject_id"])
    time_col = _lookup(names, [config.get("columns", {}).get("time", "time_min"), "time_min", "tbin", "minutes_from_t0", "charttime_bin", "charttime"])
    treatment_col = _lookup(names, [config.get("columns", {}).get("treatment", "rbc_transfusion_flag"), "rbc_transfusion_flag", "rbc_amount_ml_event", "transfusion_flag"])
    age_col = _lookup(names, ["age"])
    if not stay_col or not time_col or not treatment_col:
        raise RuntimeError(
            f"Could not infer required columns. stay_col={stay_col}, time_col={time_col}, treatment_col={treatment_col}"
        )
    _progress(
        "inferred columns: "
        f"stay_id={stay_col}, subject_id={subject_col}, time={time_col}, treatment={treatment_col}, age={age_col}"
    )

    time_type = dict(columns_with_types)[time_col]
    time_expr = _infer_time_expression(time_col, time_type)
    subject_expr = f"any_value(tg.{_quote(subject_col)}) AS subject_id" if subject_col else "NULL::BIGINT AS subject_id"
    age_expr = f"max(CAST(tg.{_quote(age_col)} AS DOUBLE)) AS age" if age_col else "NULL::DOUBLE AS age"
    min_obs = float(config.get("cohort", {}).get("min_observation_hours", 24)) * 60.0
    window_hours = int(config.get("preprocessing", {}).get("window_hours", config.get("cohort", {}).get("decision_window_hours", 48)))
    window_minutes = float(window_hours * 60)
    step_minutes = int(config.get("preprocessing", {}).get("time_step_minutes", 60))
    adult_min_age = float(config.get("cohort", {}).get("adult_min_age", 18))

    _progress("building t0 anchors and treatment labels")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW anchor_base AS
        WITH base AS (
            SELECT
                tg.{_quote(stay_col)} AS stay_id,
                {subject_expr},
                {age_expr},
                min({time_expr}) AS min_time,
                max({time_expr}) AS max_time,
                max(CASE WHEN COALESCE(CAST(tg.{_quote(treatment_col)} AS DOUBLE), 0) > 0 THEN 1 ELSE 0 END) AS transfused,
                min(CASE WHEN COALESCE(CAST(tg.{_quote(treatment_col)} AS DOUBLE), 0) > 0 THEN {time_expr} END) AS t0_transf
            FROM {timegrid_rel} tg
            GROUP BY tg.{_quote(stay_col)}
        ), med AS (
            SELECT median(t0_transf - min_time) AS median_offset
            FROM base
            WHERE t0_transf IS NOT NULL
        ), anchored AS (
            SELECT
                base.*,
                CASE
                    WHEN base.t0_transf IS NOT NULL THEN base.t0_transf
                    WHEN med.median_offset IS NOT NULL THEN LEAST(base.min_time + med.median_offset, base.max_time)
                    ELSE base.min_time + {min_obs}
                END AS t0,
                CASE WHEN base.t0_transf IS NULL THEN 1 ELSE 0 END AS pseudo_t0
            FROM base, med
        )
        SELECT *
        FROM anchored
        WHERE (max_time - min_time) >= {min_obs}
          AND (t0 - min_time) >= {min_obs}
          AND (age IS NULL OR age >= {adult_min_age})
        """
    )

    if bool(config.get("cohort", {}).get("first_icu_stay_only", True)) and subject_col:
        _progress("filtering first ICU stay per subject")
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW anchors AS
            SELECT * EXCLUDE(rn)
            FROM (
                SELECT anchor_base.*, row_number() OVER (PARTITION BY subject_id ORDER BY min_time, stay_id) AS rn
                FROM anchor_base
            )
            WHERE rn = 1
            """
        )
    else:
        con.execute("CREATE OR REPLACE TEMP VIEW anchors AS SELECT * FROM anchor_base")

    anchor_counts = con.execute(
        "SELECT count(*) AS n, sum(transfused) AS n_treated FROM anchors"
    ).fetchone()
    _progress(f"anchors ready: n={int(anchor_counts[0])}, transfused={int(anchor_counts[1] or 0)}")

    _progress("loading outcomes table")
    outcomes = read_any_table(outcomes_path)
    if "stay_id" not in outcomes.columns:
        id_col = _lookup(list(outcomes.columns), ["stay_id", "icustay_id", "icu_stay_id"])
        if id_col is None:
            raise RuntimeError("Could not infer stay_id in outcomes file.")
        outcomes = outcomes.rename(columns={id_col: "stay_id"})
    treatment_name = config.get("columns", {}).get("treatment_name", "transfused")
    if treatment_name in outcomes.columns:
        outcomes = outcomes.drop(columns=[treatment_name])
    con.register("outcomes_df", outcomes)

    cohort_path = out_dir / "cohort.parquet"
    _progress(f"writing cohort table: {cohort_path}")
    con.execute(
        f"""
        COPY (
            SELECT
                a.*,
                o.* EXCLUDE(stay_id),
                'real' AS data_source,
                false AS is_synthetic
            FROM anchors a
            LEFT JOIN outcomes_df o USING (stay_id)
        ) TO '{cohort_path.as_posix()}' (FORMAT PARQUET)
        """
    )
    cohort = pd.read_parquet(cohort_path)
    _progress(f"cohort written: rows={len(cohort)}, transfused={int(cohort['transfused'].sum())}")

    treatment = cohort[["stay_id", "transfused", "t0", "pseudo_t0", "data_source", "is_synthetic"]].copy()
    treatment.to_parquet(out_dir / "treatment.parquet", index=False)
    _progress("treatment table written")

    static_cols, temporal_cols, alias_map = _select_real_columns(columns_with_types, config)
    if not temporal_cols:
        raise RuntimeError("No configured temporal variables were found in the real timegrid schema.")
    _progress(f"selected static columns: {[alias_map.get(c, c) for c in static_cols]}")
    _progress(f"selected temporal columns: {[alias_map.get(c, c) for c in temporal_cols]}")

    select_parts = ["c.stay_id"]
    if subject_col:
        select_parts.append(f"any_value(pre.{_quote(subject_col)}) AS subject_id")
    for col in static_cols:
        alias = alias_map.get(col, col)
        select_parts.append(f"any_value(pre.{_quote(col)}) AS {_quote(alias)}")
    for col in temporal_cols:
        alias = alias_map.get(col, col)
        select_parts.append(f"avg(CAST(pre.{_quote(col)} AS DOUBLE)) AS {_quote(alias)}")

    raw_temporal_path = out_dir / "raw_temporal.parquet"
    _progress(
        f"writing pre-t0 temporal table: {raw_temporal_path} "
        f"(window={window_hours}h, step={step_minutes}min)"
    )
    con.execute(
        f"""
        COPY (
            WITH cohort AS (
                SELECT stay_id, t0
                FROM read_parquet('{cohort_path.as_posix()}')
            ), pre AS (
                SELECT
                    tg.*,
                    c.t0,
                    floor(({time_expr} - (c.t0 - {window_minutes})) / {step_minutes}) AS bin_idx
                FROM {timegrid_rel} tg
                JOIN cohort c ON tg.{_quote(stay_col)} = c.stay_id
                WHERE {time_expr} >= c.t0 - {window_minutes}
                  AND {time_expr} < c.t0
            )
            SELECT
                {", ".join(select_parts)},
                min(c.t0 - {window_minutes} + pre.bin_idx * {step_minutes}) AS time_min
            FROM pre
            JOIN cohort c ON pre.{_quote(stay_col)} = c.stay_id
            WHERE pre.bin_idx >= 0
            GROUP BY c.stay_id, pre.bin_idx
        ) TO '{raw_temporal_path.as_posix()}' (FORMAT PARQUET)
        """
    )
    raw_counts = con.execute(
        f"SELECT count(*) AS n_rows, count(DISTINCT stay_id) AS n_stays FROM read_parquet('{raw_temporal_path.as_posix()}')"
    ).fetchone()
    _progress(f"pre-t0 temporal table written: rows={int(raw_counts[0])}, stays={int(raw_counts[1])}")

    metadata = {
        "source": "real",
        "synthetic": False,
        "timegrid_path": str(timegrid_path),
        "outcomes_path": str(outcomes_path),
        "n_stays": int(len(cohort)),
        "n_transfused": int(cohort["transfused"].sum()),
        "raw_temporal_is_pre_t0_hourly": True,
        "window_hours": window_hours,
        "step_minutes": step_minutes,
        "selected_temporal_columns": [alias_map.get(c, c) for c in temporal_cols],
        "selected_static_columns": [alias_map.get(c, c) for c in static_cols],
    }
    write_metadata(out_dir / "data_source_metadata.json", metadata)
    _progress("metadata written; real-data cohort build complete")
    return cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build transfusion cohort and t0 anchors.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--source", choices=["synthetic", "real", "auto"], default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    cohort = build_cohort(config, source_override=args.source, output_dir=args.output_dir)
    print(f"cohort_rows={len(cohort)} transfused={int(cohort['transfused'].sum())}")


if __name__ == "__main__":
    main()
