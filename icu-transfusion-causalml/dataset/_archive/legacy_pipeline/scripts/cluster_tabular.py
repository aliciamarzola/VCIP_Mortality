#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clustering_utils import (
    BIN_MINUTES,
    find_repo_root,
    get_dataset,
    iter_window_batches,
    scan_t0_map,
    select_feature_cols,
    update_metrics_json,
)


ID_COL = "stay_id"
TBIN_COL = "tbin"
FLAG_COL = "rbc_transfusion_flag"

WINDOW_HOURS_LIST = [24, 48, 72]

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200000"))
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "50"))

K_LIST = [2, 3, 4, 5, 6]
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
EXPORT_K_LIST = [2, 3, 4]

# Scenarios: transfused only, and transfused + non-transfused
SCENARIOS = [
    ("with_non_transfused", True),
]
NON_TRANSFUSED_MAX = 0

CLUSTER_USE_PRE_STD = False
CLUSTER_USE_PRE_OBS_FRAC = False
EXCLUDE_PREFIXES = [
    "fluid_balance_total_ml",
]




def safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    mask = labels >= 0
    if mask.sum() == 0:
        return float("nan")
    uniq = np.unique(labels[mask])
    if uniq.size < 2:
        return float("nan")
    return float(silhouette_score(x[mask], labels[mask], metric="euclidean"))


def build_tabular_features(
    dataset,
    t0_map: Dict[int, int],
    stay_ids_cluster: List[int],
    transfused_set: set[int],
    feature_cols: List[str],
    pre_bins: int,
    post_bins: int,
) -> pd.DataFrame:
    stay_to_idx = {sid: i for i, sid in enumerate(stay_ids_cluster)}
    n_stays = len(stay_ids_cluster)
    n_feat = len(feature_cols)
    seq_len = pre_bins + post_bins

    sum_all = np.zeros((n_stays, n_feat), dtype=np.float64)
    sumsq_all = np.zeros((n_stays, n_feat), dtype=np.float64)
    count_all = np.zeros((n_stays, n_feat), dtype=np.float64)
    sum_pre = np.zeros((n_stays, n_feat), dtype=np.float64)
    count_pre = np.zeros((n_stays, n_feat), dtype=np.float64)
    sumsq_pre = np.zeros((n_stays, n_feat), dtype=np.float64)
    sum_post = np.zeros((n_stays, n_feat), dtype=np.float64)
    count_post = np.zeros((n_stays, n_feat), dtype=np.float64)

    for stay_idx, rel_idx, values, is_pre in iter_window_batches(
        dataset=dataset,
        t0_map=t0_map,
        stay_to_idx=stay_to_idx,
        feature_cols=feature_cols,
        id_col=ID_COL,
        tbin_col=TBIN_COL,
        pre_bins=pre_bins,
        post_bins=post_bins,
        batch_size=BATCH_SIZE,
        progress_every=PROGRESS_EVERY,
        progress_label="agg_tabular",
    ):
        if values.size == 0:
            continue
        vals = values.astype(np.float64, copy=False)
        nan_mask = np.isnan(vals)
        safe_vals = np.nan_to_num(vals, nan=0.0)

        np.add.at(sum_all, stay_idx, safe_vals)
        np.add.at(sumsq_all, stay_idx, safe_vals * safe_vals)
        np.add.at(count_all, stay_idx, (~nan_mask).astype(np.float64))

        if is_pre.any():
            idx_pre = stay_idx[is_pre]
            vals_pre = safe_vals[is_pre]
            nan_pre = nan_mask[is_pre]
            np.add.at(sum_pre, idx_pre, vals_pre)
            np.add.at(count_pre, idx_pre, (~nan_pre).astype(np.float64))
            np.add.at(sumsq_pre, idx_pre, vals_pre * vals_pre)

        if (~is_pre).any():
            idx_post = stay_idx[~is_pre]
            vals_post = safe_vals[~is_pre]
            nan_post = nan_mask[~is_pre]
            np.add.at(sum_post, idx_post, vals_post)
            np.add.at(count_post, idx_post, (~nan_post).astype(np.float64))

    def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        out = np.full_like(num, np.nan, dtype=np.float64)
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out

    mean_all = safe_div(sum_all, count_all)
    var_all = safe_div(sumsq_all, count_all) - mean_all * mean_all
    std_all = np.sqrt(np.maximum(var_all, 0.0))

    pre_mean = safe_div(sum_pre, count_pre)
    pre_var = safe_div(sumsq_pre, count_pre) - pre_mean * pre_mean
    pre_std = np.sqrt(np.maximum(pre_var, 0.0))
    post_mean = safe_div(sum_post, count_post)
    delta_mean = post_mean - pre_mean

    obs_frac = count_all / float(seq_len)
    pre_obs_frac = count_pre / float(pre_bins)

    data = {ID_COL: stay_ids_cluster}
    for j, col in enumerate(feature_cols):
        data[f"{col}_mean"] = mean_all[:, j]
        data[f"{col}_std"] = std_all[:, j]
        data[f"{col}_pre_mean"] = pre_mean[:, j]
        data[f"{col}_pre_std"] = pre_std[:, j]
        data[f"{col}_post_mean"] = post_mean[:, j]
        data[f"{col}_delta"] = delta_mean[:, j]
        data[f"{col}_obs_frac"] = obs_frac[:, j]
        data[f"{col}_pre_obs_frac"] = pre_obs_frac[:, j]

    data["has_transfusion"] = [1 if sid in transfused_set else 0 for sid in stay_ids_cluster]
    data["t0_tbin"] = [t0_map[sid] for sid in stay_ids_cluster]

    return pd.DataFrame(data)


def run_scenario(
    *,
    outdir: Path,
    dataset,
    stay_ids_all: List[int],
    base_t0_map: Dict[int, int],
    include_non_transfused: bool,
    suffix: str,
    window_hours: int,
    root: Path,
) -> None:
    transfused_set = set(base_t0_map.keys())
    stay_ids_transfused = sorted(transfused_set)

    pre_bins = window_hours * 60 // BIN_MINUTES
    post_bins = window_hours * 60 // BIN_MINUTES
    t0_non_transfused = window_hours * 60

    t0_map = dict(base_t0_map)
    stay_ids_cluster = stay_ids_transfused
    non_transfused: List[int] = []

    if include_non_transfused:
        non_transfused = [sid for sid in stay_ids_all if sid not in transfused_set]
        if NON_TRANSFUSED_MAX > 0 and len(non_transfused) > NON_TRANSFUSED_MAX:
            rng = np.random.RandomState(RANDOM_STATE)
            non_transfused = rng.choice(non_transfused, size=NON_TRANSFUSED_MAX, replace=False).tolist()
        for sid in non_transfused:
            t0_map[sid] = t0_non_transfused
        stay_ids_cluster = sorted(transfused_set.union(non_transfused))
        print(
            f"[cohort] include_non_transfused={len(non_transfused)} "
            f"t0_non_transfused={t0_non_transfused}min"
        )

    exclude_cols = {
        ID_COL,
        TBIN_COL,
        "subject_id",
        "hadm_id",
        "stay_bucket",
        "charttime_bin",
        "charttime",
        "intime",
        "outtime",
    }
    feature_cols = select_feature_cols(dataset, exclude_cols, drop_imputed=False)
    if EXCLUDE_PREFIXES:
        before = len(feature_cols)
        feature_cols = [
            c for c in feature_cols if not any(c.startswith(prefix) for prefix in EXCLUDE_PREFIXES)
        ]
        dropped = before - len(feature_cols)
        if dropped:
            print(f"[features] dropped {dropped} columns by prefix: {EXCLUDE_PREFIXES}")
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for clustering.")

    print(f"[features] base columns: {len(feature_cols)}")
    print(f"[features] transfused stays: {len(stay_ids_transfused)}")
    print(f"[features] clustering stays: {len(stay_ids_cluster)}")

    df_feat = build_tabular_features(
        dataset,
        t0_map,
        stay_ids_cluster,
        transfused_set,
        feature_cols,
        pre_bins,
        post_bins,
    )
    features_path = outdir / f"tabular_features_by_stay_{suffix}_w{window_hours}.parquet"
    df_feat.to_parquet(features_path, index=False)
    print(f"[save] {features_path}")

    # Prepare for clustering
    pre_mean_cols = [c for c in df_feat.columns if c.endswith("_pre_mean")]
    pre_std_cols = [c for c in df_feat.columns if c.endswith("_pre_std")] if CLUSTER_USE_PRE_STD else []
    pre_obs_cols = [c for c in df_feat.columns if c.endswith("_pre_obs_frac")] if CLUSTER_USE_PRE_OBS_FRAC else []
    feature_cols_agg = pre_mean_cols + pre_std_cols + pre_obs_cols
    x = df_feat[feature_cols_agg].to_numpy(dtype=np.float64, copy=True)
    n_samples = int(x.shape[0])
    col_has_obs = ~np.isnan(x).all(axis=0)
    if not col_has_obs.any():
        raise RuntimeError("No usable features: all columns are NaN.")
    if (~col_has_obs).any():
        info_path = outdir / f"tabular_all_nan_features_{suffix}.json"
        all_nan_cols = [c for c, ok in zip(feature_cols_agg, col_has_obs) if not ok]
        info_path.write_text(
            json.dumps({"all_nan_filled_zero": all_nan_cols}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[prep] filled {len(all_nan_cols)} all-NaN features with 0 -> {info_path}")

    t0 = time.time()
    imputer = SimpleImputer(strategy="median")
    if col_has_obs.all():
        x = imputer.fit_transform(x)
    else:
        x_valid = x[:, col_has_obs]
        x[:, col_has_obs] = imputer.fit_transform(x_valid)
        x[:, ~col_has_obs] = 0.0
    print(f"[prep] impute done in {time.time() - t0:.1f}s")

    t0 = time.time()
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    print(f"[prep] scale done in {time.time() - t0:.1f}s")

    metrics = {
        "n_samples": n_samples,
        "n_features": int(x.shape[1]),
        "window_hours": {"pre": window_hours, "post": window_hours},
        "include_non_transfused": bool(include_non_transfused),
        "non_transfused_count": int(len(non_transfused)),
        "kmeans": {"scores": []},
    }

    if n_samples < 2:
        print(f"[skip] only {n_samples} sample(s); skipping clustering for {suffix}")
        df_labels = pd.DataFrame({ID_COL: stay_ids_all})
        df_labels["has_transfusion"] = df_labels[ID_COL].isin(stay_ids_transfused).astype(int)
        df_labels["cluster_kmeans"] = np.nan
        labels_path = outdir / f"tabular_clusters_{suffix}_w{window_hours}.csv"
        df_labels.to_csv(labels_path, index=False)
        print(f"[save] {labels_path}")
        metrics["skipped"] = "n_samples<2"
        metrics_path = outdir / "cluster_metrics.json"
        update_metrics_json(metrics_path, f"tabular_{suffix}_w{window_hours}", metrics)
        print(f"[save] {metrics_path}")
        return

    # KMeans
    k_list = [k for k in K_LIST if k <= n_samples]
    if not k_list:
        print(f"[skip] no valid k in K_LIST for n_samples={n_samples} ({suffix})")
        metrics["skipped"] = "no_valid_k"
        labels_path = outdir / f"tabular_clusters_{suffix}_w{window_hours}.csv"
        df_labels = pd.DataFrame({ID_COL: stay_ids_all})
        df_labels["has_transfusion"] = df_labels[ID_COL].isin(stay_ids_transfused).astype(int)
        df_labels["cluster_kmeans"] = np.nan
        df_labels.to_csv(labels_path, index=False)
        print(f"[save] {labels_path}")
        metrics_path = outdir / "cluster_metrics.json"
        update_metrics_json(metrics_path, f"tabular_{suffix}_w{window_hours}", metrics)
        print(f"[save] {metrics_path}")
        return

    best_k = None
    best_sil = -float("inf")
    export_labels: Dict[int, np.ndarray] = {}
    for k in k_list:
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
        labels = km.fit_predict(x)
        sil = safe_silhouette(x, labels)
        metrics["kmeans"]["scores"].append(
            {"k": int(k), "silhouette": float(sil), "inertia": float(km.inertia_)}
        )
        print(f"[kmeans] k={k} silhouette={sil:.4f} inertia={km.inertia_:.1f} time={time.time() - t0:.1f}s")
        if k in EXPORT_K_LIST:
            export_labels[k] = labels
        if not math.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    if best_k is None:
        best_k = k_list[0]
    metrics["kmeans"]["best_k"] = int(best_k)

    # Fit best KMeans labels
    km = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE)
    kmeans_labels = km.fit_predict(x)

    # Build labels output
    df_labels = pd.DataFrame({ID_COL: stay_ids_all})
    df_labels["has_transfusion"] = df_labels[ID_COL].isin(stay_ids_transfused).astype(int)

    df_labels["cluster_kmeans"] = df_labels[ID_COL].map(dict(zip(stay_ids_cluster, kmeans_labels)))

    labels_path = outdir / f"tabular_clusters_{suffix}_w{window_hours}.csv"
    df_labels.to_csv(labels_path, index=False)
    print(f"[save] {labels_path}")

    for k, labels in export_labels.items():
        df_k = pd.DataFrame({ID_COL: stay_ids_all})
        df_k["has_transfusion"] = df_k[ID_COL].isin(stay_ids_transfused).astype(int)
        df_k[f"cluster_kmeans_k{k}"] = df_k[ID_COL].map(dict(zip(stay_ids_cluster, labels)))
        k_path = outdir / f"tabular_clusters_{suffix}_w{window_hours}_k{k}.csv"
        df_k.to_csv(k_path, index=False)
        print(f"[save] {k_path}")

    metrics_path = outdir / "cluster_metrics.json"
    update_metrics_json(metrics_path, f"tabular_{suffix}_w{window_hours}", metrics)
    print(f"[save] {metrics_path}")

    df_feat["cluster_kmeans"] = df_feat[ID_COL].map(dict(zip(stay_ids_cluster, kmeans_labels)))
    df_feat_cluster = df_feat[df_feat["cluster_kmeans"].notna()].copy()
    if not df_feat_cluster.empty:
        df_feat_cluster["cluster_kmeans"] = df_feat_cluster["cluster_kmeans"].astype(int)
        pre_cols = [c for c in df_feat_cluster.columns if c.endswith("_pre_mean")]
        post_cols = [c for c in df_feat_cluster.columns if c.endswith("_post_mean")]
        delta_cols = [c for c in df_feat_cluster.columns if c.endswith("_delta")]
        cluster_means = df_feat_cluster.groupby("cluster_kmeans")[pre_cols + post_cols + delta_cols].mean()
        feature_rows = []
        for base_col in feature_cols:
            pre_col = f"{base_col}_pre_mean"
            post_col = f"{base_col}_post_mean"
            delta_col = f"{base_col}_delta"
            if pre_col not in cluster_means.columns:
                continue
            for cluster_id, row in cluster_means.iterrows():
                feature_rows.append(
                    {
                        "cluster_kmeans": int(cluster_id),
                        "feature": base_col,
                        "pre_mean": row.get(pre_col),
                        "post_mean": row.get(post_col),
                        "delta_mean": row.get(delta_col),
                    }
                )
        if feature_rows:
            feat_summary = pd.DataFrame(feature_rows)
            feat_summary_path = outdir / f"tabular_cluster_feature_summary_{suffix}_w{window_hours}.parquet"
            feat_summary.to_parquet(feat_summary_path, index=False)
            print(f"[save] {feat_summary_path}")

    outcomes_path = Path(os.getenv("OUTCOMES_CSV", root / "outputs_outcomes" / "outcomes_by_stay.csv"))
    if outcomes_path.exists():
        df_out = pd.read_csv(outcomes_path)
        if "stay_id" in df_out.columns:
            df_merge = df_labels.merge(df_out, on="stay_id", how="left")
            df_merge["mortality_anytime"] = df_merge.get("mortality_anytime", 0).fillna(0).astype(int)
            df_merge["vm_time_hours"] = pd.to_numeric(df_merge.get("vm_time_hours"), errors="coerce")

            summary = (
                df_merge.groupby(["cluster_kmeans", "has_transfusion"], dropna=False)
                .agg(
                    n=("stay_id", "count"),
                    mortality_rate=("mortality_anytime", "mean"),
                    vm_mean=("vm_time_hours", "mean"),
                    vm_median=("vm_time_hours", "median"),
                )
                .reset_index()
            )
            summary_path = outdir / f"tabular_cluster_outcomes_{suffix}_w{window_hours}.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[save] {summary_path}")

            if include_non_transfused:
                pivot = summary.pivot_table(
                    index="cluster_kmeans",
                    columns="has_transfusion",
                    values=["mortality_rate", "vm_mean"],
                )
                benefit = pd.DataFrame(
                    {
                        "cluster_kmeans": pivot.index,
                        "mortality_rate_diff": (pivot["mortality_rate"][1] - pivot["mortality_rate"][0]).values,
                        "vm_mean_diff": (pivot["vm_mean"][1] - pivot["vm_mean"][0]).values,
                    }
                )
                benefit_path = outdir / f"tabular_cluster_benefit_{suffix}_w{window_hours}.csv"
                benefit.to_csv(benefit_path, index=False)
                print(f"[save] {benefit_path}")

    print(f"[done] tabular clustering complete ({suffix}, w{window_hours})")


def main() -> None:
    root = find_repo_root(Path(__file__).resolve())
    outdir = Path(os.getenv("CLUSTER_OUTDIR", root / "outputs" / "reports" / "clustering"))
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(root)

    print("[scan] locating transfusion t0...")
    t0_map, stay_ids_all = scan_t0_map(
        dataset,
        ID_COL,
        TBIN_COL,
        FLAG_COL,
        BATCH_SIZE,
        progress_every=PROGRESS_EVERY,
        progress_label="scan_t0_tabular",
    )
    if not t0_map:
        raise RuntimeError("No transfused stays found (rbc_transfusion_flag == 1).")

    for suffix, include_non_transfused in SCENARIOS:
        for window_hours in WINDOW_HOURS_LIST:
            run_scenario(
                outdir=outdir,
                dataset=dataset,
                stay_ids_all=stay_ids_all,
                base_t0_map=t0_map,
                include_non_transfused=include_non_transfused,
                suffix=suffix,
                window_hours=window_hours,
                root=root,
            )


if __name__ == "__main__":
    main()
