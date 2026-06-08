from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _existing_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def infer_treatment_column(temporal: pd.DataFrame, config: Dict[str, Any]) -> Optional[str]:
    configured = config.get("columns", {}).get("treatment", "rbc_transfusion_flag")
    candidates = [
        configured,
        "rbc_transfusion_flag",
        "rbc_amount_ml_event",
        "transfusion_flag",
        "transfused",
        "prbc",
    ]
    return _existing_column(temporal, candidates)


def create_time_anchors(temporal: pd.DataFrame, outcomes: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if temporal.empty:
        raise RuntimeError("Temporal data are empty; cannot create time anchors.")
    if "stay_id" not in temporal.columns or "time_min" not in temporal.columns:
        raise RuntimeError("Temporal data must contain stay_id and time_min.")

    treatment_col = infer_treatment_column(temporal, config)
    grouped = temporal.groupby("stay_id", as_index=False).agg(min_time=("time_min", "min"), max_time=("time_min", "max"))

    if treatment_col:
        event_df = temporal[pd.to_numeric(temporal[treatment_col], errors="coerce").fillna(0) > 0]
        t0_events = event_df.groupby("stay_id", as_index=False).agg(t0_transf=("time_min", "min"))
        grouped = grouped.merge(t0_events, on="stay_id", how="left")
        grouped["transfused_timegrid"] = grouped["t0_transf"].notna().astype(int)
    else:
        grouped["t0_transf"] = np.nan
        grouped["transfused_timegrid"] = 0

    treatment_name = config.get("columns", {}).get("treatment_name", "transfused")
    if not outcomes.empty and treatment_name in outcomes.columns:
        outcome_treat = outcomes[["stay_id", treatment_name]].drop_duplicates("stay_id").rename(
            columns={treatment_name: "_outcome_treatment"}
        )
        grouped = grouped.merge(outcome_treat, on="stay_id", how="left")
        grouped["transfused"] = grouped["_outcome_treatment"].fillna(grouped["transfused_timegrid"]).astype(int)
        grouped["transfused_source"] = np.where(grouped["_outcome_treatment"].notna(), "outcomes", "timegrid")
        grouped = grouped.drop(columns=["_outcome_treatment"])
    else:
        grouped["transfused"] = grouped["transfused_timegrid"].astype(int)
        grouped["transfused_source"] = "timegrid"

    treated_offsets = grouped.loc[grouped["t0_transf"].notna(), "t0_transf"] - grouped.loc[
        grouped["t0_transf"].notna(), "min_time"
    ]
    if len(treated_offsets):
        pseudo_offset = float(np.nanmedian(treated_offsets))
    else:
        pseudo_offset = float(config.get("cohort", {}).get("min_observation_hours", 24) * 60)

    grouped["t0"] = np.where(
        grouped["transfused"].eq(1) & grouped["t0_transf"].notna(),
        grouped["t0_transf"],
        np.minimum(grouped["min_time"] + pseudo_offset, grouped["max_time"]),
    )
    grouped["pseudo_t0"] = grouped["transfused"].eq(0).astype(int)
    grouped["pseudo_t0_offset_minutes"] = np.where(grouped["pseudo_t0"].eq(1), pseudo_offset, np.nan)
    return grouped
