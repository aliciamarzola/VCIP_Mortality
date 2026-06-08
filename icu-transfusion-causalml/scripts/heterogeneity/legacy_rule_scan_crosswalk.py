from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import (  # noqa: E402
    bootstrap_ci,
    ensure_dir,
    load_trial_config,
    output_root,
    read_json,
    save_table,
    summarize_effect,
    write_json,
)


CURRENT_GROUPS = ("B1", "B2", "M1", "M2", "M3")


def legacy_rules() -> list[dict[str, Any]]:
    return [
        {
            "legacy_group_id": "old_B1",
            "direction": "benefit",
            "name": "Perfundido amplo com baixa taquicardia",
            "criteria": [
                ("glucose_lab_std", "<=", 0.0),
                ("heart_rate_median", "<=", 70.0),
                ("spo2_slope", ">", 0.001268),
            ],
        },
        {
            "legacy_group_id": "old_B2",
            "direction": "benefit",
            "name": "Anemia dinamica com oxigenacao estavel",
            "criteria": [
                ("glucose_lab_std", "<=", 0.0),
                ("hemoglobin_slope", "<=", -0.006063),
                ("spo2_std", "<=", 0.8245),
            ],
        },
        {
            "legacy_group_id": "old_B3",
            "direction": "benefit",
            "name": "Estresse metabolico com perfusao relativa preservada",
            "criteria": [
                ("lactate_std", ">", 0.3523),
                ("potassium_std", "<=", 0.04242),
                ("spo2_std", "<=", 0.8245),
            ],
        },
        {
            "legacy_group_id": "old_B4",
            "direction": "benefit",
            "name": "Pressao media elevada com inflamacao moderada",
            "criteria": [
                ("mbp_median", ">", 89.0),
                ("platelets_slope", "<=", -0.07924),
                ("urea_std", "<=", 0.6436),
            ],
        },
        {
            "legacy_group_id": "old_M1",
            "direction": "harm",
            "name": "Falencia hepatobiliar com estresse metabolico",
            "criteria": [
                ("alkaline_phosphatase_max", ">", 75.0),
                ("lactate_slope", ">", 0.0002952),
            ],
        },
        {
            "legacy_group_id": "old_M2",
            "direction": "harm",
            "name": "Doenca critica sustentada hepatobiliar-renal",
            "criteria": [
                ("alkaline_phosphatase_mean", ">", 75.0),
                ("sapsii_min", ">", 43.0),
            ],
        },
        {
            "legacy_group_id": "old_M3",
            "direction": "harm",
            "name": "Baixo fluxo cardiorrenal com hipotensao diastolica",
            "criteria": [
                ("creatinine_max", ">", 1.6),
                ("dbp_median", "<=", 52.0),
            ],
        },
        {
            "legacy_group_id": "old_M4",
            "direction": "harm",
            "name": "Falencia progressiva renal e hipoperfusao metabolica",
            "criteria": [
                ("creatinine_max", ">", 1.6),
                ("lactate_slope", ">", 0.0002952),
            ],
        },
    ]


def find_legacy_runs_root(user_path: str | None) -> Path:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates.extend(
        [
            REPO_ROOT.parent / "cluster-transfusion" / "outputs" / "runs",
            REPO_ROOT.parent / "medDreamer" / "medDreamer" / "cluster-transfusion" / "outputs" / "runs",
            REPO_ROOT.parent.parent / "medDreamer" / "medDreamer" / "cluster-transfusion" / "outputs" / "runs",
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find cluster-transfusion outputs/runs. Searched:\n{searched}")


def rule_text(criteria: list[tuple[str, str, float]]) -> str:
    return " AND ".join(f"{feature} {op} {threshold:g}" for feature, op, threshold in criteria)


def apply_criteria(df: pd.DataFrame, criteria: list[tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feature, op, threshold in criteria:
        if feature not in df.columns:
            raise RuntimeError(f"Legacy baseline table missing feature required by rule: {feature}")
        values = pd.to_numeric(df[feature], errors="coerce")
        if op == "<=":
            mask &= values <= threshold
        elif op == ">":
            mask &= values > threshold
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return mask.fillna(False)


def load_analysis(config_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    metadata = read_json(out_root / "processed" / "target_trial_metadata.json")
    primary = metadata.get("primary_outcome", "mortality_anytime")
    trial = pd.read_parquet(out_root / "processed" / "trial_dataset.parquet")
    nuisance = pd.read_parquet(out_root / "causal" / "nuisance_predictions.parquet")
    final_assign = pd.read_parquet(out_root / "final_groups" / "final_group_assignments.parquet")
    nuisance = nuisance.drop(columns=["analysis_split", "transfused", primary], errors="ignore")
    analysis = trial.merge(nuisance, on="stay_id", how="left").merge(
        final_assign[["stay_id", "B1", "B2", "M1", "M2", "M3", "final_primary_label"]],
        on="stay_id",
        how="left",
    )
    return analysis, final_assign, metadata, out_root


def load_legacy_baseline(runs_root: Path, run_id: str) -> pd.DataFrame:
    path = runs_root / run_id / "w48" / "features" / "baseline_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy baseline features: {path}")
    baseline = pd.read_parquet(path)
    baseline["stay_id"] = baseline["stay_id"].astype("int64")
    return baseline


def build_legacy_assignments(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = baseline[["stay_id"]].drop_duplicates("stay_id").copy()
    rows = []
    for rule in legacy_rules():
        mask = apply_criteria(baseline, rule["criteria"])
        group_id = rule["legacy_group_id"]
        matched = baseline.loc[mask, ["stay_id"]].drop_duplicates("stay_id")
        assignments[group_id] = assignments["stay_id"].isin(matched["stay_id"])
        rows.append(
            {
                "legacy_group_id": group_id,
                "direction": rule["direction"],
                "name": rule["name"],
                "rule_text": rule_text(rule["criteria"]),
                "n_in_full_legacy_baseline": int(mask.sum()),
            }
        )
    return assignments, pd.DataFrame(rows)


def effect_with_ci(df: pd.DataFrame, mask: pd.Series, outcome_col: str, seed: int, n_boot: int) -> dict[str, Any] | None:
    sub = df[mask].copy()
    if sub.empty or sub["transfused"].nunique() < 2:
        return None
    row = summarize_effect(sub, outcome_col, "transfused")
    ci_low, ci_high = bootstrap_ci(sub["aipw_score_ate"].to_numpy(dtype=float), seed=seed, n_boot=n_boot)
    row["aipw_ate_ci95_low"] = ci_low
    row["aipw_ate_ci95_high"] = ci_high
    row["model_ite_mean"] = float(np.nanmean(pd.to_numeric(sub["mu1"], errors="coerce") - pd.to_numeric(sub["mu0"], errors="coerce")))
    return row


def estimate_legacy_rule_effects(
    df: pd.DataFrame,
    rule_meta: pd.DataFrame,
    outcome_col: str,
    run_id: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for rule_index, rule in enumerate(rule_meta.itertuples(index=False)):
        mask_all = df[getattr(rule, "legacy_group_id")].fillna(False)
        for split_index, split in enumerate(["all", "discovery", "validation"]):
            mask = mask_all if split == "all" else mask_all & df["analysis_split"].eq(split)
            row = effect_with_ci(df, mask, outcome_col, seed + rule_index * 100 + split_index, n_boot)
            if row is None:
                continue
            row.update(
                {
                    "run_id": run_id,
                    "split": split,
                    "legacy_group_id": getattr(rule, "legacy_group_id"),
                    "direction": getattr(rule, "direction"),
                    "name": getattr(rule, "name"),
                    "rule_text": getattr(rule, "rule_text"),
                    "n_in_full_legacy_baseline": int(getattr(rule, "n_in_full_legacy_baseline")),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def current_direction(group_id: str) -> str:
    return "benefit" if group_id in {"B1", "B2"} else "harm"


def build_overlap(df: pd.DataFrame, rule_meta: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    for rule in rule_meta.itertuples(index=False):
        old_id = getattr(rule, "legacy_group_id")
        old_mask = df[old_id].fillna(False)
        old_n = int(old_mask.sum())
        for cur_id in CURRENT_GROUPS:
            cur_mask = df[cur_id].fillna(False)
            cur_n = int(cur_mask.sum())
            both = int((old_mask & cur_mask).sum())
            union = int((old_mask | cur_mask).sum())
            table = pd.crosstab(cur_mask.astype(bool), old_mask.astype(bool))
            odds_ratio = np.nan
            p_value = np.nan
            if all(v in table.index for v in [False, True]) and all(v in table.columns for v in [False, True]):
                odds_ratio, p_value = fisher_exact(
                    [
                        [int(table.loc[True, True]), int(table.loc[True, False])],
                        [int(table.loc[False, True]), int(table.loc[False, False])],
                    ]
                )
            rows.append(
                {
                    "run_id": run_id,
                    "legacy_group_id": old_id,
                    "legacy_direction": getattr(rule, "direction"),
                    "current_group_id": cur_id,
                    "current_direction": current_direction(cur_id),
                    "same_direction": getattr(rule, "direction") == current_direction(cur_id),
                    "n_total": n_total,
                    "n_legacy": old_n,
                    "n_current": cur_n,
                    "n_overlap": both,
                    "pct_legacy_captured_by_current": float(both / old_n) if old_n else np.nan,
                    "pct_current_in_legacy": float(both / cur_n) if cur_n else np.nan,
                    "jaccard": float(both / union) if union else np.nan,
                    "fisher_odds_ratio": float(odds_ratio),
                    "fisher_p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def estimate_intersection_effects(
    df: pd.DataFrame,
    rule_meta: pd.DataFrame,
    outcome_col: str,
    run_id: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for r_index, rule in enumerate(rule_meta.itertuples(index=False)):
        old_id = getattr(rule, "legacy_group_id")
        old_mask = df[old_id].fillna(False)
        for c_index, cur_id in enumerate(CURRENT_GROUPS):
            mask = old_mask & df[cur_id].fillna(False)
            row = effect_with_ci(df, mask, outcome_col, seed + r_index * 1000 + c_index * 20, n_boot)
            if row is None:
                continue
            row.update(
                {
                    "run_id": run_id,
                    "legacy_group_id": old_id,
                    "legacy_direction": getattr(rule, "direction"),
                    "current_group_id": cur_id,
                    "current_direction": current_direction(cur_id),
                    "same_direction": getattr(rule, "direction") == current_direction(cur_id),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_Nao disponivel._"
    shown = df[columns].head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, effects: pd.DataFrame, overlap: pd.DataFrame, intersections: pd.DataFrame) -> None:
    all_effects = effects[effects["split"].eq("all")].copy()
    all_effects["expected_sign"] = np.where(all_effects["direction"].eq("benefit"), "negative", "positive")
    all_effects["validated_direction"] = np.where(
        (all_effects["direction"].eq("benefit") & (all_effects["aipw_ate_ci95_high"] < 0))
        | (all_effects["direction"].eq("harm") & (all_effects["aipw_ate_ci95_low"] > 0)),
        "yes",
        "no",
    )
    concordant_overlap = overlap[overlap["same_direction"]].sort_values(["legacy_group_id", "pct_legacy_captured_by_current"], ascending=[True, False])
    discordant_overlap = overlap[~overlap["same_direction"]].sort_values(["legacy_group_id", "pct_legacy_captured_by_current"], ascending=[True, False])
    effect_cols = [
        "legacy_group_id",
        "direction",
        "n",
        "n_treated",
        "n_control",
        "unadjusted_diff",
        "aipw_ate",
        "aipw_ate_ci95_low",
        "aipw_ate_ci95_high",
        "validated_direction",
        "name",
    ]
    overlap_cols = [
        "legacy_group_id",
        "current_group_id",
        "n_legacy",
        "n_current",
        "n_overlap",
        "pct_legacy_captured_by_current",
        "pct_current_in_legacy",
        "jaccard",
        "fisher_odds_ratio",
        "fisher_p_value",
    ]
    int_cols = [
        "legacy_group_id",
        "current_group_id",
        "same_direction",
        "n",
        "n_treated",
        "n_control",
        "unadjusted_diff",
        "aipw_ate",
        "aipw_ate_ci95_low",
        "aipw_ate_ci95_high",
    ]
    lines = [
        "# Crosswalk between legacy clinical scan rules and final causal groups",
        "",
        "## Objective",
        "",
        "This analysis applies the representative legacy scan rules B1-B4/M1-M4 from the cluster-transfusion project to the same target-trial cohort used in the current causal analysis. It then estimates doubly robust AIPW effects in each old rule group and measures overlap with the final groups B1/B2/M1/M2/M3.",
        "",
        "## AIPW effects in legacy scan rules",
        "",
        markdown_table(all_effects, effect_cols, max_rows=80),
        "",
        "## Concordant overlap",
        "",
        "Rows below compare old benefit rules with current benefit rules and old harm rules with current harm rules.",
        "",
        markdown_table(concordant_overlap, overlap_cols, max_rows=80),
        "",
        "## Discordant overlap",
        "",
        "Rows below show old benefit rules overlapping current harm rules, or old harm rules overlapping current benefit rules.",
        "",
        markdown_table(discordant_overlap, overlap_cols, max_rows=80),
        "",
        "## AIPW effects in intersections",
        "",
        markdown_table(intersections.sort_values(["legacy_group_id", "current_group_id"]), int_cols, max_rows=120),
        "",
        "## Interpretation",
        "",
        "If an old rule has the same sign under AIPW and overlaps strongly with a current same-direction group, it is reproduced by the current causal scan. If the old rule does not keep its sign or overlaps weakly, it should be treated as an earlier associative phenotype rather than a confirmed causal subgroup.",
    ]
    (output_dir / "legacy_rule_scan_crosswalk_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_legacy_rule_crosswalk(config_dir: str, legacy_root: str | None, run_id: str, n_boot: int, seed: int) -> None:
    analysis, _final_assign, metadata, out_root = load_analysis(config_dir)
    primary = metadata.get("primary_outcome", "mortality_anytime")
    runs_root = find_legacy_runs_root(legacy_root)
    baseline = load_legacy_baseline(runs_root, run_id)
    legacy_assign, rule_meta = build_legacy_assignments(baseline)
    df = analysis.merge(legacy_assign, on="stay_id", how="left")
    for rule_id in rule_meta["legacy_group_id"]:
        df[rule_id] = df[rule_id].fillna(False).astype(bool)

    output_dir = ensure_dir(out_root / "legacy_rule_crosswalk" / run_id)
    effects = estimate_legacy_rule_effects(df, rule_meta, primary, run_id, n_boot=n_boot, seed=seed)
    overlap = build_overlap(df, rule_meta, run_id)
    intersections = estimate_intersection_effects(df, rule_meta, primary, run_id, n_boot=n_boot, seed=seed)

    save_table(df[["stay_id", "analysis_split", "transfused", primary, *rule_meta["legacy_group_id"].tolist(), *CURRENT_GROUPS, "final_primary_label"]], output_dir / "legacy_rule_assignments.parquet")
    save_table(rule_meta, output_dir / "legacy_rule_definitions.csv")
    save_table(effects, output_dir / "legacy_rule_aipw_effects.csv")
    save_table(overlap, output_dir / "legacy_rule_current_group_overlap.csv")
    save_table(intersections, output_dir / "legacy_rule_current_group_intersection_effects.csv")
    write_json(
        output_dir / "legacy_rule_crosswalk_metadata.json",
        {
            "legacy_runs_root": str(runs_root),
            "legacy_run_id": run_id,
            "baseline_features": str(runs_root / run_id / "w48" / "features" / "baseline_features.parquet"),
            "outcome": primary,
            "bootstrap": n_boot,
            "seed": seed,
            "note": "Representative legacy rules B1-B4/M1-M4 are applied exactly using legacy baseline_features and evaluated with current target-trial AIPW nuisance estimates.",
        },
    )
    write_report(output_dir, effects, overlap, intersections)
    print(f"legacy_rule_scan_crosswalk_complete output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old representative scan rules with current final causal scan groups.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--legacy-root", default=None, help="Path to cluster-transfusion/outputs/runs.")
    parser.add_argument("--run-id", default="run_cal03_replace_full_w48")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_legacy_rule_crosswalk(args.config_dir, args.legacy_root, args.run_id, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
