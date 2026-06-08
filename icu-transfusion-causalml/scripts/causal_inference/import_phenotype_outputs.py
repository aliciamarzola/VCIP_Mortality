from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.pipeline_common import (
    causal_dir,
    counterfactual_dir,
    evaluation_dir,
    load_config,
    phenotypes_dir,
    processed_dir,
    resolve_path,
    save_table,
    write_json,
)  # noqa: E402


def _clean_stale_outputs(config: Dict[str, Any]) -> None:
    proc = processed_dir(config)
    phen = phenotypes_dir(config)
    caus = causal_dir(config)
    counter = counterfactual_dir(config)
    evals = evaluation_dir(config)
    report_path = resolve_path(config, "report_file", "outputs/causal_inference/reports/research_report.md")
    scan_report_path = resolve_path(config, "scan_group_report_file", "outputs/causal_inference/reports/scan_group_inference_report.md")

    paths = [
        proc / "raw_temporal.parquet",
        proc / "temporal_tensor.npy",
        proc / "temporal_tensor_ids.parquet",
        proc / "temporal_missing_mask.npy",
        phen / "clusters.parquet",
        phen / "cluster_metrics.csv",
        phen / "cluster_metrics_by_seed.csv",
        phen / "cluster_summary.csv",
        phen / "temporal_embeddings.npy",
        phen / "temporal_embedding_ids.parquet",
        phen / "temporal_representation_metadata.json",
        caus / "cate_estimates.csv",
        caus / "ite_estimates.parquet",
        caus / "causal_forest_ite.parquet",
        caus / "effect_by_phenotype.csv",
        caus / "patient_benefit_ranking.csv",
        caus / "propensity_scores.parquet",
        caus / "propensity_covariates.csv",
        caus / "weighted_population.parquet",
        caus / "effect_by_scan_group.csv",
        caus / "ite_by_scan_group.csv",
        caus / "scan_group_summary.csv",
        counter / "predicted_y0_y1.parquet",
        counter / "individual_treatment_effects.parquet",
        counter / "model_metrics.csv",
        counter / "counterfactual_evaluation.csv",
        evals / "predictive_metrics.csv",
        evals / "causal_metrics.csv",
        evals / "sensitivity_results.csv",
        report_path,
        scan_report_path,
    ]
    for path in paths:
        if path.exists():
            path.unlink()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required phenotype-construction artifact: {path}")
    return path


def _phenotype_run_dir(phenotype_root: Path, run_id: str, window: int) -> tuple[Path, Path]:
    candidates = [
        phenotype_root / "runs" / run_id,
        phenotype_root / "outputs" / "runs" / run_id,
        phenotype_root / "outputs" / "phenotype_construction" / "runs" / run_id,
    ]
    run_dir = next((path for path in candidates if path.exists()), candidates[0])
    shared = run_dir / "shared"
    window_dir = run_dir / f"w{window}"
    return shared, window_dir


def _requested_outcomes(config: Dict[str, Any], available: list[str]) -> list[str]:
    outcomes_cfg = config.get("variables", {}).get("outcomes", {})
    requested = list(outcomes_cfg.get("primary", [])) + list(outcomes_cfg.get("secondary", []))
    cols = [col for col in requested if col in available]
    if cols:
        return cols
    excluded = {"stay_id", "transfused", "t0_transf"}
    return [col for col in available if col not in excluded and pd.api.types.is_numeric_dtype(pd.Series(dtype="float64"))]


def _drop_leaky_columns(df: pd.DataFrame, keep_post_t0_features: bool) -> pd.DataFrame:
    if keep_post_t0_features:
        return df
    drop_patterns = ("_post_mean", "_delta")
    drop_cols = [col for col in df.columns if col.endswith(drop_patterns)]
    return df.drop(columns=drop_cols, errors="ignore")


def _build_clusters_from_embeddings(embeddings: pd.DataFrame, metrics: pd.DataFrame | None, selected_k: int | None, seed: int) -> pd.DataFrame:
    emb_cols = [col for col in embeddings.columns if col.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("Legacy embeddings do not contain emb_* columns.")
    if selected_k is None:
        if metrics is not None and not metrics.empty and "silhouette" in metrics.columns and "k" in metrics.columns:
            usable = metrics.dropna(subset=["silhouette"]).copy()
            selected_k = int(usable.sort_values("silhouette", ascending=False).iloc[0]["k"]) if not usable.empty else 3
        else:
            selected_k = 3
    x = StandardScaler().fit_transform(embeddings[emb_cols].to_numpy(dtype=float))
    selected_k = max(2, min(int(selected_k), x.shape[0] - 1))
    labels = KMeans(n_clusters=selected_k, random_state=seed, n_init=20).fit_predict(x)
    return pd.DataFrame({"stay_id": embeddings["stay_id"].to_numpy(), "phenotype": labels.astype(int), "selected_k": selected_k})


def _matched_effects(matched: pd.DataFrame, cohort: pd.DataFrame, outcome_col: str, clusters: pd.DataFrame | None) -> pd.DataFrame:
    if matched.empty or outcome_col not in cohort.columns:
        return pd.DataFrame()
    outcomes = cohort[["stay_id", outcome_col]].drop_duplicates("stay_id")
    pairs = matched.merge(outcomes, left_on="stay_id_transf", right_on="stay_id", how="left").rename(columns={outcome_col: "y_t"})
    pairs = pairs.drop(columns=["stay_id"])
    pairs = pairs.merge(outcomes, left_on="stay_id_ctrl", right_on="stay_id", how="left").rename(columns={outcome_col: "y_c"})
    pairs = pairs.drop(columns=["stay_id"])
    rows = []
    diff = pairs["y_t"] - pairs["y_c"]
    rows.append(
        {
            "method": "article_nearest_neighbor_psm",
            "estimand": "ATT",
            "n_pairs": int(len(pairs)),
            "treated_mean": float(pairs["y_t"].mean()),
            "control_mean": float(pairs["y_c"].mean()),
            "effect": float(diff.mean()),
        }
    )
    if clusters is not None and not clusters.empty:
        pairs = pairs.merge(clusters[["stay_id", "phenotype"]], left_on="stay_id_transf", right_on="stay_id", how="left")
        for phenotype, sub in pairs.dropna(subset=["phenotype"]).groupby("phenotype"):
            d = sub["y_t"] - sub["y_c"]
            rows.append(
                {
                    "method": "article_nearest_neighbor_psm",
                    "estimand": "ATT_by_phenotype",
                    "phenotype": int(phenotype),
                    "n_pairs": int(len(sub)),
                    "treated_mean": float(sub["y_t"].mean()),
                    "control_mean": float(sub["y_c"].mean()),
                    "effect": float(d.mean()),
                }
            )
    return pd.DataFrame(rows)


def import_phenotype_outputs(
    config: Dict[str, Any],
    phenotype_root: Path,
    run_id: str,
    window: int,
    embedding: str,
    selected_k: int | None,
    keep_post_t0_features: bool,
) -> None:
    _clean_stale_outputs(config)
    if not phenotype_root.is_absolute():
        phenotype_root = (ROOT / phenotype_root).resolve()
    shared, window_dir = _phenotype_run_dir(phenotype_root, run_id, window)
    proc = processed_dir(config)
    phen = phenotypes_dir(config)
    caus = causal_dir(config)

    t0_path = _require(shared / "t0_table.parquet")
    cohort_path = _require(shared / "outcomes_cohort.parquet")
    baseline_path = _require(window_dir / "features" / "baseline_features.parquet")
    matched_path = window_dir / "matching" / "matched_pairs.parquet"
    balance_path = window_dir / "matching" / "balance_diagnostics.csv"
    embeddings_path = window_dir / "embeddings" / f"embeddings_{embedding}.parquet"
    metrics_path = window_dir / "matching" / f"cluster_metrics_{embedding}.csv"

    print(f"[import] phenotype root: {phenotype_root}", flush=True)
    print(f"[import] run: {run_id}, window={window}, embedding={embedding}", flush=True)
    print(f"[import] reading {t0_path}", flush=True)
    t0 = pd.read_parquet(t0_path)
    print(f"[import] reading {cohort_path}", flush=True)
    outcomes_cohort = pd.read_parquet(cohort_path)
    print(f"[import] reading {baseline_path}", flush=True)
    baseline = pd.read_parquet(baseline_path)

    drop_from_outcomes = [col for col in ["transfused", "t0"] if col in outcomes_cohort.columns]
    outcomes_only = outcomes_cohort.drop(columns=drop_from_outcomes, errors="ignore")
    cohort = t0.merge(outcomes_only, on="stay_id", how="left", suffixes=("_t0", "_outcomes"))
    cohort["data_source"] = "phenotype_construction"
    cohort["is_synthetic"] = False
    cohort.to_parquet(proc / "cohort.parquet", index=False)

    treatment = cohort[["stay_id", "transfused", "t0"]].copy()
    treatment["pseudo_t0"] = (treatment["transfused"] == 0).astype(int)
    treatment["data_source"] = "phenotype_construction"
    treatment["is_synthetic"] = False
    treatment.to_parquet(proc / "treatment.parquet", index=False)

    outcome_cols = _requested_outcomes(config, list(cohort.columns))
    outcomes = cohort[["stay_id"] + outcome_cols].copy()
    outcomes.to_parquet(proc / "outcomes.parquet", index=False)

    baseline = baseline.drop(columns=[c for c in ["transfused", "t0"] if c in baseline.columns], errors="ignore")
    baseline = _drop_leaky_columns(baseline, keep_post_t0_features=keep_post_t0_features)
    baseline.to_parquet(proc / "static_features.parquet", index=False)
    analysis = baseline.merge(treatment, on="stay_id", how="left").merge(outcomes, on="stay_id", how="left")
    analysis.to_parquet(proc / "analysis_features.parquet", index=False)

    clusters = None
    if embeddings_path.exists():
        print(f"[import] reading {embeddings_path}", flush=True)
        embeddings = pd.read_parquet(embeddings_path)
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
        clusters = _build_clusters_from_embeddings(
            embeddings,
            metrics=metrics,
            selected_k=selected_k,
            seed=int(config.get("run", {}).get("seed", 42)),
        )
        emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
        clusters.to_parquet(phen / "clusters.parquet", index=False)
        np.save(phen / "temporal_embeddings.npy", embeddings[emb_cols].to_numpy(dtype=float))
        embeddings[["stay_id"]].to_parquet(phen / "temporal_embedding_ids.parquet", index=False)
        write_json(
            phen / "temporal_representation_metadata.json",
            {
                "method": f"phenotype_construction_{embedding}",
                "shape": [int(len(embeddings)), int(len(emb_cols))],
            "source": "phenotype_construction",
                "phenotype_run_id": run_id,
                "window_hours": window,
                "note": "Imported from the previous cluster-transfusion pipeline; not recomputed in this repository.",
            },
        )
        if metrics is not None:
            metrics.to_csv(phen / "cluster_metrics.csv", index=False)
    else:
        print(f"[warn] embeddings not found: {embeddings_path}", flush=True)

    if matched_path.exists():
        print(f"[import] copying {matched_path}", flush=True)
        matched = pd.read_parquet(matched_path)
        matched.to_parquet(caus / "matched_population.parquet", index=False)
        primary_outcome = config.get("columns", {}).get("primary_outcome", "mortality_anytime")
        effects = _matched_effects(matched, cohort, primary_outcome, clusters)
        if not effects.empty:
            effects.to_csv(caus / "effect_estimates_psm.csv", index=False)
    if balance_path.exists():
        print(f"[import] copying {balance_path}", flush=True)
        balance = pd.read_csv(balance_path)
        balance.to_csv(caus / "balance_table.csv", index=False)

    metadata = {
        "source": "phenotype_construction",
        "synthetic": False,
            "phenotype_root": str(phenotype_root),
        "phenotype_run_id": run_id,
        "window_hours": window,
        "embedding": embedding,
        "n_stays": int(len(cohort)),
        "n_transfused": int(cohort["transfused"].sum()) if "transfused" in cohort.columns else None,
        "n_controls": int((cohort["transfused"] == 0).sum()) if "transfused" in cohort.columns else None,
        "keep_post_t0_features": bool(keep_post_t0_features),
        "note": "Imported from the previous cluster-transfusion pipeline to avoid recomputing timegrid, t0, baseline features, embeddings and matching.",
    }
    write_json(proc / "data_source_metadata.json", metadata)
    write_json(
        proc / "feature_metadata.json",
        {
            "n_stays": int(len(cohort)),
            "static_feature_columns": [col for col in baseline.columns if col != "stay_id"],
            "outcome_columns": outcome_cols,
            "primary_outcome": config.get("columns", {}).get("primary_outcome", "mortality_anytime"),
            "treatment_column": "transfused",
            "imported_from_phenotype_construction": True,
            "no_post_t0_features": not keep_post_t0_features,
        },
    )
    print("[import] complete", flush=True)
    print(f"[import] cohort rows={len(cohort)} treated={metadata['n_transfused']} controls={metadata['n_controls']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import phenotype-construction artifacts into the causal pipeline format.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--phenotype-root", default="outputs/phenotype_construction")
    parser.add_argument("--run-id", default="run_cal03_replace_full_w48")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--embedding", choices=["minirocket", "ts2vec"], default="minirocket")
    parser.add_argument("--selected-k", type=int, default=None)
    parser.add_argument("--keep-post-t0-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    import_phenotype_outputs(
        config=config,
        phenotype_root=Path(args.phenotype_root),
        run_id=args.run_id,
        window=args.window,
        embedding=args.embedding,
        selected_k=args.selected_k,
        keep_post_t0_features=args.keep_post_t0_features,
    )


if __name__ == "__main__":
    main()
