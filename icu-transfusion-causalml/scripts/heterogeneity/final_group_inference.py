from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import (  # noqa: E402
    ensure_dir,
    load_trial_config,
    output_root,
    read_json,
    summarize_effect,
    weighted_mean,
    write_json,
)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        stats[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    denom = float(np.sum(weights**2))
    if denom <= 0:
        return np.nan
    return float(np.sum(weights) ** 2 / denom)


def smd_for_feature(df: pd.DataFrame, feature: str, treatment_col: str, weight_col: str | None = None) -> float:
    if feature not in df.columns:
        return np.nan
    tmp = pd.DataFrame(
        {
            "x": pd.to_numeric(df[feature], errors="coerce"),
            "a": pd.to_numeric(df[treatment_col], errors="coerce"),
        }
    )
    if weight_col:
        tmp["w"] = pd.to_numeric(df[weight_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "a"])
    if tmp.empty or tmp["a"].nunique() < 2:
        return np.nan
    treated = tmp[tmp["a"] == 1]
    control = tmp[tmp["a"] == 0]
    if treated.empty or control.empty:
        return np.nan
    if weight_col:
        mean_t = weighted_mean(treated["x"].to_numpy(), treated["w"].to_numpy())
        mean_c = weighted_mean(control["x"].to_numpy(), control["w"].to_numpy())
    else:
        mean_t = float(treated["x"].mean())
        mean_c = float(control["x"].mean())
    pooled = np.sqrt((float(treated["x"].var(ddof=0)) + float(control["x"].var(ddof=0))) / 2.0)
    return float((mean_t - mean_c) / pooled) if pooled > 1e-12 else 0.0


def final_rules() -> list[dict[str, Any]]:
    return [
        {
            "group_id": "B1",
            "direction": "benefit",
            "name": "Stable HR + falling hemoglobin + non-high final SpO2",
            "role": "primary",
            "description_pt": "Frequencia cardiaca estavel/controlada, hemoglobina em queda e SpO2 final nao alta.",
            "criteria": [
                ("heart_rate_max", "<=", 103.833),
                ("heart_rate_delta", "<=", 3.905),
                ("hemoglobin_slope", "<=", -0.002),
                ("spo2_last", "<=", 97.5),
            ],
        },
        {
            "group_id": "B2",
            "direction": "benefit",
            "name": "High-specificity benefit phenotype",
            "role": "sensitivity",
            "description_pt": "Subgrupo B1 com pressao arterial media baixa/moderada.",
            "criteria": [
                ("heart_rate_max", "<=", 103.833),
                ("heart_rate_delta", "<=", 3.905),
                ("hemoglobin_slope", "<=", -0.002),
                ("mbp_mean", "<=", 78.706),
                ("spo2_last", "<=", 97.5),
            ],
        },
        {
            "group_id": "M1",
            "direction": "harm",
            "name": "Rising HR + high maximum MBP",
            "role": "primary",
            "description_pt": "Frequencia cardiaca em ascensao com pico de pressao arterial media elevado.",
            "criteria": [
                ("heart_rate_delta", ">", 3.905),
                ("mbp_max", ">", 93.0),
            ],
        },
        {
            "group_id": "M2",
            "direction": "harm",
            "name": "Cardiorenal high-risk harm phenotype",
            "role": "sensitivity",
            "description_pt": "Subgrupo M1 com variabilidade de creatinina elevada.",
            "criteria": [
                ("heart_rate_delta", ">", 3.905),
                ("mbp_max", ">", 93.0),
                ("creatinine_std", ">", 0.146),
            ],
        },
        {
            "group_id": "M3",
            "direction": "harm",
            "name": "Hemoglobin not falling",
            "role": "secondary",
            "description_pt": "Hemoglobina sem queda relevante no periodo pre-t0.",
            "criteria": [
                ("hemoglobin_slope", ">", -0.002),
            ],
        },
    ]


def rule_text(criteria: list[tuple[str, str, float]]) -> str:
    return " AND ".join(f"{feature} {op} {threshold:g}" for feature, op, threshold in criteria)


def apply_criteria(df: pd.DataFrame, criteria: list[tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feature, op, threshold in criteria:
        values = pd.to_numeric(df[feature], errors="coerce")
        if op == "<=":
            mask &= values <= threshold
        elif op == ">":
            mask &= values > threshold
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return mask.fillna(False)


def load_analysis_frame(config_dir: str) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    metadata = read_json(out_root / "processed" / "target_trial_metadata.json")
    trial = pd.read_parquet(out_root / "processed" / "trial_dataset.parquet")
    nuisance = pd.read_parquet(out_root / "causal" / "nuisance_predictions.parquet")
    primary = metadata.get("primary_outcome", "mortality_anytime")
    nuisance = nuisance.drop(columns=["analysis_split", "transfused", primary], errors="ignore")
    df = trial.merge(nuisance, on="stay_id", how="left")
    return df, metadata, out_root


def balance_summary(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    rows = []
    for feature in feature_cols:
        if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
            continue
        rows.append(
            {
                "feature": feature,
                "smd_unweighted": smd_for_feature(df, feature, "transfused"),
                "smd_overlap": smd_for_feature(df, feature, "transfused", "overlap_weight"),
                "smd_iptw": smd_for_feature(df, feature, "transfused", "iptw_weight"),
            }
        )
    if not rows:
        return {}
    table = pd.DataFrame(rows)
    top = table.assign(abs_overlap=table["smd_overlap"].abs()).sort_values("abs_overlap", ascending=False).head(5)
    return {
        "balance_feature_count": int(len(table)),
        "max_abs_smd_unweighted": float(table["smd_unweighted"].abs().max()),
        "max_abs_smd_overlap": float(table["smd_overlap"].abs().max()),
        "max_abs_smd_iptw": float(table["smd_iptw"].abs().max()),
        "median_abs_smd_overlap": float(table["smd_overlap"].abs().median()),
        "n_smd_overlap_gt_0_1": int((table["smd_overlap"].abs() > 0.1).sum()),
        "top_residual_smd_overlap": "; ".join(
            f"{row.feature}:{row.smd_overlap:.3f}" for row in top.itertuples() if np.isfinite(row.smd_overlap)
        ),
    }


def estimate_group(
    df: pd.DataFrame,
    mask: pd.Series,
    outcome_col: str,
    split: str,
    seed: int,
    n_boot: int,
) -> dict[str, Any] | None:
    if split == "discovery":
        mask = mask & (df["analysis_split"] == "discovery")
    elif split == "validation":
        mask = mask & (df["analysis_split"] == "validation")
    elif split == "complement":
        mask = ~mask
    sub = df[mask].copy()
    if sub.empty or sub["transfused"].nunique() < 2:
        return None
    summary = summarize_effect(sub, outcome_col, "transfused")
    ci_low, ci_high = bootstrap_ci(sub["aipw_score_ate"].to_numpy(dtype=float), seed=seed, n_boot=n_boot)
    treatment = sub["transfused"].astype(int).to_numpy()
    ps = pd.to_numeric(sub["propensity_score"], errors="coerce")
    ite = pd.to_numeric(sub["mu1"], errors="coerce") - pd.to_numeric(sub["mu0"], errors="coerce")
    treated = sub[sub["transfused"] == 1]
    control = sub[sub["transfused"] == 0]
    summary.update(
        {
            "split": split,
            "aipw_ate_ci95_low": ci_low,
            "aipw_ate_ci95_high": ci_high,
            "model_y0_mean": float(np.nanmean(sub["mu0"])),
            "model_y1_mean": float(np.nanmean(sub["mu1"])),
            "model_ite_mean": float(np.nanmean(ite)),
            "model_ite_median": float(np.nanmedian(ite)),
            "prop_model_benefit": float(np.nanmean(ite < 0)),
            "prop_model_harm": float(np.nanmean(ite > 0)),
            "ps_min": float(np.nanmin(ps)),
            "ps_median": float(np.nanmedian(ps)),
            "ps_mean": float(np.nanmean(ps)),
            "ps_max": float(np.nanmax(ps)),
            "ps_low_clip_fraction": float(np.nanmean(ps <= 0.050001)),
            "ps_high_clip_fraction": float(np.nanmean(ps >= 0.949999)),
            "ess_overlap_treated": effective_sample_size(treated["overlap_weight"].to_numpy()) if not treated.empty else np.nan,
            "ess_overlap_control": effective_sample_size(control["overlap_weight"].to_numpy()) if not control.empty else np.nan,
            "ess_iptw_treated": effective_sample_size(treated["iptw_weight"].to_numpy()) if not treated.empty else np.nan,
            "ess_iptw_control": effective_sample_size(control["iptw_weight"].to_numpy()) if not control.empty else np.nan,
            "treated_fraction": float(treatment.mean()),
        }
    )
    return summary


def profile_group(df: pd.DataFrame, mask: pd.Series, group_id: str, variables: list[str]) -> list[dict[str, Any]]:
    rows = []
    sub = df[mask].copy()
    for variable in variables:
        if variable not in sub.columns:
            continue
        values = pd.to_numeric(sub[variable], errors="coerce")
        rows.append(
            {
                "group_id": group_id,
                "feature": variable,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "missing_fraction": float(values.isna().mean()),
            }
        )
    return rows


def final_group_label(row: pd.Series) -> str:
    if bool(row.get("B1", False)) and bool(row.get("M1", False)):
        return "overlap_B1_M1"
    if bool(row.get("B1", False)):
        return "benefit_primary"
    if bool(row.get("M1", False)):
        return "harm_primary"
    return "neither_primary"


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    shown = df[columns].head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, effects: pd.DataFrame, profiles: pd.DataFrame, assignment: pd.DataFrame) -> None:
    all_effects = effects[effects["split"] == "all"].copy()
    split_effects = effects[effects["split"].isin(["discovery", "validation"])].copy()
    columns = [
        "group_id",
        "direction",
        "role",
        "n",
        "n_treated",
        "n_control",
        "treated_mean",
        "control_mean",
        "unadjusted_diff",
        "aipw_ate",
        "aipw_ate_ci95_low",
        "aipw_ate_ci95_high",
        "aipw_att",
        "overlap_weighted_diff",
        "iptw_diff",
        "model_ite_mean",
        "prop_model_benefit",
        "prop_model_harm",
    ]
    split_columns = [
        "group_id",
        "split",
        "n",
        "n_treated",
        "n_control",
        "aipw_ate",
        "aipw_ate_ci95_low",
        "aipw_ate_ci95_high",
        "model_ite_mean",
    ]
    overlap_counts = assignment["final_primary_label"].value_counts().rename_axis("label").reset_index(name="n")
    lines = [
        "# Final causal inference for scan-derived transfusion phenotypes",
        "",
        "## Final groups",
        "",
        "- B1 benefit primary: stable/controlled heart rate, falling hemoglobin, and non-high final SpO2.",
        "- B2 benefit sensitivity: B1 plus low-to-moderate mean arterial pressure.",
        "- M1 harm primary: rising heart rate plus high maximum mean arterial pressure.",
        "- M2 harm sensitivity: M1 plus high creatinine variability.",
        "- M3 harm secondary: hemoglobin not falling.",
        "",
        "## Main all-sample estimates",
        "",
        markdown_table(all_effects, columns),
        "",
        "## Discovery/validation estimates",
        "",
        markdown_table(split_effects, split_columns, max_rows=30),
        "",
        "## Primary group membership",
        "",
        markdown_table(overlap_counts, ["label", "n"]),
        "",
        "## Interpretation",
        "",
        "The final benefit phenotype is most consistent with dynamic anemia in a patient who remains physiologically compensated: heart rate is controlled/stable, hemoglobin is falling, and final SpO2 is not high.",
        "",
        "The final harm phenotype is most consistent with a stress/deterioration trajectory: heart rate is rising and maximum mean arterial pressure is high before the decision point. The cardiorenal sensitivity group strengthens this signal when creatinine variability is added.",
        "",
        "These estimates use observational target-trial emulation and doubly robust adjustment. They support heterogeneous-effect hypotheses, not clinical decision rules without external or temporal validation.",
        "",
        "## Output files",
        "",
        "- `final_group_effects.csv`",
        "- `final_group_balance.csv`",
        "- `final_group_profiles.csv`",
        "- `final_group_assignments.parquet`",
        "- `final_group_patient_counterfactuals.parquet`",
        "- `final_group_report.md`",
    ]
    (output_dir / "final_group_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_final_group_inference(config_dir: str, n_boot: int) -> None:
    df, metadata, out_root = load_analysis_frame(config_dir)
    output_dir = ensure_dir(out_root / "final_groups")
    outcome_col = metadata.get("primary_outcome", "mortality_anytime")
    feature_cols = metadata.get("feature_columns", [])

    assignments = df[["stay_id", "analysis_split", "transfused", outcome_col]].copy()
    rules = final_rules()
    masks: dict[str, pd.Series] = {}
    for rule in rules:
        mask = apply_criteria(df, rule["criteria"])
        masks[rule["group_id"]] = mask
        assignments[rule["group_id"]] = mask
    assignments["final_primary_label"] = assignments.apply(final_group_label, axis=1)

    effect_rows = []
    balance_rows = []
    profile_rows = []
    patient_rows = []
    profile_vars = [
        "age",
        "hemoglobin_slope",
        "hemoglobin_last",
        "heart_rate_delta",
        "heart_rate_max",
        "mbp_mean",
        "mbp_max",
        "spo2_last",
        "spo2_min",
        "sofa_mean",
        "creatinine_std",
        "lactate_std",
    ]

    for rule_index, rule in enumerate(rules):
        group_id = rule["group_id"]
        mask = masks[group_id]
        for split_index, split in enumerate(["all", "discovery", "validation", "complement"]):
            estimate = estimate_group(
                df,
                mask,
                outcome_col=outcome_col,
                split=split,
                seed=42 + 100 * rule_index + split_index,
                n_boot=n_boot,
            )
            if estimate is None:
                continue
            estimate.update(
                {
                    "group_id": group_id,
                    "direction": rule["direction"],
                    "role": rule["role"],
                    "name": rule["name"],
                    "description_pt": rule["description_pt"],
                    "rule_text": rule_text(rule["criteria"]),
                }
            )
            if split == "all":
                estimate.update(balance_summary(df[mask].copy(), feature_cols))
            effect_rows.append(estimate)
        if mask.any():
            balance = balance_summary(df[mask].copy(), feature_cols)
            balance.update({"group_id": group_id, "rule_text": rule_text(rule["criteria"])})
            balance_rows.append(balance)
            profile_rows.extend(profile_group(df, mask, group_id, profile_vars))
            patient = df.loc[
                mask,
                [
                    "stay_id",
                    "analysis_split",
                    "transfused",
                    outcome_col,
                    "propensity_score",
                    "mu0",
                    "mu1",
                    "aipw_score_ate",
                ],
            ].copy()
            patient.insert(0, "group_id", group_id)
            patient["model_ite"] = pd.to_numeric(patient["mu1"], errors="coerce") - pd.to_numeric(patient["mu0"], errors="coerce")
            patient_rows.append(patient)

    effects = pd.DataFrame(effect_rows)
    balances = pd.DataFrame(balance_rows)
    profiles = pd.DataFrame(profile_rows)
    patient_cf = pd.concat(patient_rows, ignore_index=True) if patient_rows else pd.DataFrame()

    effects.to_csv(output_dir / "final_group_effects.csv", index=False)
    balances.to_csv(output_dir / "final_group_balance.csv", index=False)
    profiles.to_csv(output_dir / "final_group_profiles.csv", index=False)
    assignments.to_parquet(output_dir / "final_group_assignments.parquet", index=False)
    patient_cf.to_parquet(output_dir / "final_group_patient_counterfactuals.parquet", index=False)
    write_json(
        output_dir / "final_group_metadata.json",
        {
            "outcome": outcome_col,
            "bootstrap": n_boot,
            "groups": [
                {
                    "group_id": rule["group_id"],
                    "direction": rule["direction"],
                    "role": rule["role"],
                    "name": rule["name"],
                    "description_pt": rule["description_pt"],
                    "rule_text": rule_text(rule["criteria"]),
                }
                for rule in rules
            ],
            "interpretation": "Final scan-derived groups for hypothesis-generating causal inference.",
        },
    )
    write_report(output_dir, effects, profiles, assignments)
    print(f"final_group_inference_complete output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final causal inference for selected benefit and harm scan groups.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    run_final_group_inference(args.config_dir, args.bootstrap)


if __name__ == "__main__":
    main()
