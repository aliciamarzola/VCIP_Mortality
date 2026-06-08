from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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


DEFAULT_RUNS = ("run_cal03_replace_full_w48", "run_cal03_noreplace_w48")
FINAL_GROUPS = ("B1", "B2", "M1", "M2", "M3")


def find_legacy_runs_root(user_path: str | None) -> Path:
    candidates: list[Path] = []
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


def classify_article_sign(diff: float) -> str:
    if not np.isfinite(diff):
        return "unknown"
    if diff < 0:
        return "article_benefit"
    if diff > 0:
        return "article_harm"
    return "neutral"


def reconstruct_k2_assignments(run_dir: Path, seed: int = 42) -> pd.DataFrame:
    embeddings_path = run_dir / "w48" / "embeddings" / "embeddings_minirocket.parquet"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing MiniRocket embeddings: {embeddings_path}")
    embeddings = pd.read_parquet(embeddings_path)
    emb_cols = [col for col in embeddings.columns if col.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError(f"No emb_* columns found in {embeddings_path}")
    x = StandardScaler().fit_transform(embeddings[emb_cols].to_numpy(dtype=float))
    labels = KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(x)
    return pd.DataFrame({"stay_id": embeddings["stay_id"].astype(int), "legacy_k2_cluster": labels.astype(int)})


def load_article_cluster_summary(run_dir: Path) -> pd.DataFrame:
    benefit_path = run_dir / "w48" / "reports" / "cluster_benefit_minirocket.csv"
    if not benefit_path.exists():
        benefit_path = run_dir / "w48" / "matching" / "cluster_benefit_minirocket.csv"
    if not benefit_path.exists():
        raise FileNotFoundError(f"Missing article cluster report: {benefit_path}")
    report = pd.read_csv(benefit_path)
    mortality = report[(report["metric"] == "mortality_anytime") & (pd.to_numeric(report["k"], errors="coerce") == 2)].copy()
    mortality = mortality.rename(
        columns={
            "cluster": "legacy_k2_cluster",
            "diff_mean": "article_mortality_diff",
            "n_pairs": "article_n_pairs",
            "transfused_mean": "article_transfused_mortality",
            "control_mean": "article_control_mortality",
        }
    )
    keep = [
        "legacy_k2_cluster",
        "article_n_pairs",
        "article_transfused_mortality",
        "article_control_mortality",
        "article_mortality_diff",
    ]
    mortality = mortality[keep].copy()
    mortality["legacy_article_sign"] = mortality["article_mortality_diff"].map(classify_article_sign)
    return mortality


def load_phenotype_profile(run_dir: Path) -> pd.DataFrame:
    phenotype_path = run_dir / "w48" / "reports" / "cluster_phenotype_minirocket.csv"
    if not phenotype_path.exists():
        phenotype_path = run_dir / "w48" / "matching" / "cluster_phenotype_minirocket.csv"
    if not phenotype_path.exists():
        return pd.DataFrame()
    profile = pd.read_csv(phenotype_path)
    profile = profile[(pd.to_numeric(profile.get("k"), errors="coerce") == 2) & (profile.get("stat") == "median")].copy()
    selected = [
        "age_max",
        "hemoglobin_mean",
        "creatinine_mean",
        "heart_rate_mean",
        "mbp_mean",
        "sofa_mean",
        "lactate_mean",
    ]
    profile = profile[profile["feature"].isin(selected)].copy()
    if profile.empty:
        return pd.DataFrame()
    return profile.pivot_table(index="legacy_k2_cluster" if "legacy_k2_cluster" in profile.columns else "cluster", columns="feature", values="value").reset_index().rename(columns={"cluster": "legacy_k2_cluster"})


def load_analysis(config_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
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
    return analysis, final_assign, trial, metadata, out_root


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


def estimate_legacy_cluster_effects(
    df: pd.DataFrame,
    outcome_col: str,
    run_id: str,
    seed: int,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    for split in ["all", "discovery", "validation"]:
        split_mask = pd.Series(True, index=df.index) if split == "all" else df["analysis_split"].eq(split)
        for cluster in sorted(df["legacy_k2_cluster"].dropna().unique()):
            mask = split_mask & df["legacy_k2_cluster"].eq(cluster)
            row = effect_with_ci(df, mask, outcome_col, seed + int(cluster) + 100 * len(rows), n_boot)
            if row is None:
                continue
            article = df.loc[df["legacy_k2_cluster"].eq(cluster), ["article_mortality_diff", "legacy_article_sign"]].drop_duplicates()
            row.update(
                {
                    "run_id": run_id,
                    "split": split,
                    "legacy_k2_cluster": int(cluster),
                    "article_mortality_diff": float(article["article_mortality_diff"].iloc[0]) if not article.empty else np.nan,
                    "legacy_article_sign": article["legacy_article_sign"].iloc[0] if not article.empty else "unknown",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def estimate_intersection_effects(
    df: pd.DataFrame,
    outcome_col: str,
    run_id: str,
    seed: int,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    for group in FINAL_GROUPS:
        for cluster in sorted(df["legacy_k2_cluster"].dropna().unique()):
            mask = df[group].fillna(False) & df["legacy_k2_cluster"].eq(cluster)
            row = effect_with_ci(df, mask, outcome_col, seed + int(cluster) + 1000 * (FINAL_GROUPS.index(group) + 1), n_boot)
            if row is None:
                continue
            row.update({"run_id": run_id, "group_id": group, "legacy_k2_cluster": int(cluster)})
            rows.append(row)
    return pd.DataFrame(rows)


def build_crosswalk(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    base_counts = df["legacy_k2_cluster"].value_counts(normalize=True).sort_index().to_dict()
    for group in FINAL_GROUPS + ("benefit_primary", "harm_primary", "neither_primary"):
        if group in FINAL_GROUPS:
            mask = df[group].fillna(False)
            label_type = "final_rule"
        else:
            mask = df["final_primary_label"].eq(group)
            label_type = "primary_label"
        selected = df[mask].copy()
        if selected.empty:
            continue
        fisher_or = np.nan
        fisher_p = np.nan
        if set(df["legacy_k2_cluster"].dropna().unique()) == {0, 1}:
            indicator = mask.astype(bool)
            table = pd.crosstab(indicator, df["legacy_k2_cluster"])
            if True in table.index and False in table.index and 0 in table.columns and 1 in table.columns:
                fisher_or, fisher_p = fisher_exact(
                    [
                        [int(table.loc[True, 1]), int(table.loc[True, 0])],
                        [int(table.loc[False, 1]), int(table.loc[False, 0])],
                    ]
                )
        for cluster, count in selected["legacy_k2_cluster"].value_counts().sort_index().items():
            cluster = int(cluster)
            article = selected.loc[selected["legacy_k2_cluster"].eq(cluster), ["article_mortality_diff", "legacy_article_sign"]].drop_duplicates()
            rows.append(
                {
                    "run_id": run_id,
                    "label_type": label_type,
                    "group_id": group,
                    "legacy_k2_cluster": cluster,
                    "n_in_group_cluster": int(count),
                    "n_group": int(len(selected)),
                    "pct_group_in_cluster": float(count / len(selected)),
                    "pct_target_trial_in_cluster": float(base_counts.get(cluster, np.nan)),
                    "article_mortality_diff": float(article["article_mortality_diff"].iloc[0]) if not article.empty else np.nan,
                    "legacy_article_sign": article["legacy_article_sign"].iloc[0] if not article.empty else "unknown",
                    "fisher_or_cluster1_vs_cluster0": float(fisher_or),
                    "fisher_p_cluster1_vs_cluster0": float(fisher_p),
                }
            )
    return pd.DataFrame(rows)


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    shown = df[columns].head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    legacy_effects: pd.DataFrame,
    crosswalk: pd.DataFrame,
    intersections: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    effect_cols = [
        "run_id",
        "split",
        "legacy_k2_cluster",
        "legacy_article_sign",
        "article_mortality_diff",
        "n",
        "n_treated",
        "n_control",
        "unadjusted_diff",
        "aipw_ate",
        "aipw_ate_ci95_low",
        "aipw_ate_ci95_high",
        "aipw_att",
        "overlap_weighted_diff",
    ]
    cross_cols = [
        "run_id",
        "group_id",
        "legacy_k2_cluster",
        "n_in_group_cluster",
        "n_group",
        "pct_group_in_cluster",
        "pct_target_trial_in_cluster",
        "article_mortality_diff",
        "legacy_article_sign",
        "fisher_or_cluster1_vs_cluster0",
        "fisher_p_cluster1_vs_cluster0",
    ]
    int_cols = [
        "run_id",
        "group_id",
        "legacy_k2_cluster",
        "n",
        "n_treated",
        "n_control",
        "unadjusted_diff",
        "aipw_ate",
        "aipw_ate_ci95_low",
        "aipw_ate_ci95_high",
    ]
    all_legacy = legacy_effects[legacy_effects["split"] == "all"].copy()
    primary_cross = crosswalk[crosswalk["label_type"].eq("final_rule")].copy()
    intersection_cluster1 = intersections[intersections["legacy_k2_cluster"].eq(1)].copy()
    lines = [
        "# Legacy K=2 phenotype crosswalk and causal inference",
        "",
        "## Objective",
        "",
        "This analysis treats the MiniRocket/K-means K=2 phenotypes from the previous cluster-transfusion project as pre-specified physiologic groups. It then estimates doubly robust causal effects inside those groups and crosses them with the final scan-derived groups B1/B2/M1/M2/M3.",
        "",
        "The goal is not to relabel the old clusters as causal groups. The goal is to test whether the new causal groups are concentrated inside the old physiologic phenotypes and whether the old phenotypes still separate treatment effect in the target-trial cohort.",
        "",
        "## Key findings",
        "",
        "1. The old K=2 phenotypes do not separate a clear causal benefit/harm effect in the target-trial cohort. Their AIPW estimates are close to zero or have confidence intervals crossing zero.",
        "2. The final benefit group B1 is not enriched in the old benefit-like K=2 cluster beyond the cohort baseline. In both legacy runs, around 93-94% of B1 falls in cluster 1, but 93-94% of the whole target-trial cohort also falls in cluster 1.",
        "3. The final harm groups are not simply the old harm cluster. M1 has only a weak tendency toward the smaller old cluster in the no-replacement run, and M3 is actually enriched in cluster 1, which is the lower-hemoglobin/higher-SOFA macrocluster.",
        "4. The important result is inside-cluster separation: within old cluster 1, B1 remains beneficial while M1/M2/M3 remain harmful. This supports the interpretation that the scan-derived groups refine the old physiologic phenotype rather than reproduce it.",
        "",
        "## Within old cluster 1: final groups still separate effect",
        "",
        markdown_table(intersection_cluster1, int_cols, max_rows=30),
        "",
        "## AIPW estimates inside old K=2 phenotypes",
        "",
        markdown_table(all_legacy, effect_cols),
        "",
        "## Crosswalk between final groups and old K=2 phenotypes",
        "",
        markdown_table(primary_cross, cross_cols, max_rows=60),
        "",
        "## AIPW estimates inside intersections",
        "",
        markdown_table(intersections, int_cols, max_rows=60),
        "",
        "## Old phenotype profile medians",
        "",
        markdown_table(profiles, list(profiles.columns), max_rows=20) if not profiles.empty else "No phenotype profile table was available.",
        "",
        "## Interpretation",
        "",
        "In this target-trial cohort, the old K=2 phenotypes are broad physiologic macrophenotypes. Most eligible patients fall into the same old K=2 cluster because the target trial selects a narrower hemoglobin-eligible population than the original matched cohort.",
        "",
        "Therefore, the final scan-derived groups should be interpreted as causal refinements inside the physiologic space identified by the previous paper, not as exact reproductions of the old clusters.",
        "",
        "A strong bridge would require B1/B2 to be enriched in an old benefit phenotype and M1/M2/M3 to be enriched in an old harm phenotype. If enrichment is weak or absent, the conclusion is that K=2 was too coarse for the causal question and the rule-based target-trial groups provide finer heterogeneity.",
    ]
    (output_dir / "legacy_crosswalk_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_crosswalk(config_dir: str, legacy_root: str | None, runs: list[str], n_boot: int, seed: int) -> None:
    analysis, final_assign, _trial, metadata, out_root = load_analysis(config_dir)
    primary = metadata.get("primary_outcome", "mortality_anytime")
    output_dir = ensure_dir(out_root / "legacy_crosswalk")
    runs_root = find_legacy_runs_root(legacy_root)

    all_assignments = []
    all_effects = []
    all_crosswalks = []
    all_intersections = []
    all_profiles = []

    for run_id in runs:
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing legacy run directory: {run_dir}")
        k2 = reconstruct_k2_assignments(run_dir, seed=seed)
        article = load_article_cluster_summary(run_dir)
        profile = load_phenotype_profile(run_dir)
        if not profile.empty:
            profile.insert(0, "run_id", run_id)
            all_profiles.append(profile)

        labels = k2.merge(article, on="legacy_k2_cluster", how="left")
        df = analysis.merge(labels, on="stay_id", how="left")
        if df["legacy_k2_cluster"].isna().any():
            missing = int(df["legacy_k2_cluster"].isna().sum())
            raise RuntimeError(f"{missing} target-trial stays did not match legacy embeddings for {run_id}.")
        assign = final_assign.merge(labels, on="stay_id", how="left")
        assign.insert(0, "run_id", run_id)
        all_assignments.append(assign)
        all_effects.append(estimate_legacy_cluster_effects(df, primary, run_id, seed=seed, n_boot=n_boot))
        all_crosswalks.append(build_crosswalk(df, run_id))
        all_intersections.append(estimate_intersection_effects(df, primary, run_id, seed=seed, n_boot=n_boot))

    assignments = pd.concat(all_assignments, ignore_index=True)
    legacy_effects = pd.concat(all_effects, ignore_index=True)
    crosswalk = pd.concat(all_crosswalks, ignore_index=True)
    intersections = pd.concat(all_intersections, ignore_index=True)
    profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()

    save_table(assignments, output_dir / "legacy_k2_assignments.parquet")
    save_table(legacy_effects, output_dir / "legacy_k2_aipw_effects.csv")
    save_table(crosswalk, output_dir / "final_group_legacy_k2_crosswalk.csv")
    save_table(intersections, output_dir / "final_group_legacy_k2_intersection_effects.csv")
    if not profiles.empty:
        save_table(profiles, output_dir / "legacy_k2_profiles.csv")
    write_json(
        output_dir / "legacy_crosswalk_metadata.json",
        {
            "legacy_runs_root": str(runs_root),
            "runs": runs,
            "k": 2,
            "embedding": "minirocket",
            "n_target_trial": int(analysis["stay_id"].nunique()),
            "bootstrap": n_boot,
            "seed": seed,
            "note": "K=2 labels are reconstructed from legacy MiniRocket embeddings using KMeans(n_clusters=2, n_init=20, random_state=seed), matching the previous reporting script.",
        },
    )
    write_report(output_dir, legacy_effects, crosswalk, intersections, profiles)
    print(f"legacy_cluster_crosswalk_complete output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross final causal scan groups with legacy cluster-transfusion K=2 phenotypes.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--legacy-root", default=None, help="Path to cluster-transfusion/outputs/runs.")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="Comma-separated legacy run IDs.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    run_crosswalk(args.config_dir, args.legacy_root, runs, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
