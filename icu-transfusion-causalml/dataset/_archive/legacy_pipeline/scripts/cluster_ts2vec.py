#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

try:
    import hdbscan  # type: ignore
    from hdbscan import validity_index as hdb_validity_index  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False
    hdb_validity_index = None

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

PRE_HOURS = int(os.getenv("PRE_HOURS", "48"))
POST_HOURS = int(os.getenv("POST_HOURS", "48"))
PRE_BINS = PRE_HOURS * 60 // BIN_MINUTES
POST_BINS = POST_HOURS * 60 // BIN_MINUTES
SEQ_LEN = PRE_BINS + POST_BINS

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200000"))
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "50"))

# TS2Vec params
OUTPUT_DIMS = int(os.getenv("TS2VEC_OUTPUT_DIMS", "320"))
HIDDEN_DIMS = int(os.getenv("TS2VEC_HIDDEN_DIMS", "64"))
DEPTH = int(os.getenv("TS2VEC_DEPTH", "10"))
TS2VEC_LR = float(os.getenv("TS2VEC_LR", "0.001"))
TS2VEC_BATCH = int(os.getenv("TS2VEC_BATCH", "16"))
TS2VEC_MAX_TRAIN_LENGTH = os.getenv("TS2VEC_MAX_TRAIN_LENGTH", "")
TS2VEC_MAX_TRAIN_LENGTH = int(TS2VEC_MAX_TRAIN_LENGTH) if TS2VEC_MAX_TRAIN_LENGTH else None
TS2VEC_ITERS = os.getenv("TS2VEC_ITERS", "")
TS2VEC_ITERS = int(TS2VEC_ITERS) if TS2VEC_ITERS else None
TS2VEC_EPOCHS = os.getenv("TS2VEC_EPOCHS", "")
TS2VEC_EPOCHS = int(TS2VEC_EPOCHS) if TS2VEC_EPOCHS else None

USE_PCA = os.getenv("TS2VEC_USE_PCA", "0") == "1"
PCA_NCOMP = int(os.getenv("TS2VEC_PCA_NCOMP", "50"))

K_LIST = [int(x) for x in os.getenv("K_LIST", "2,3,4").split(",")]
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

RUN_HDBSCAN = os.getenv("RUN_HDBSCAN", "1") == "1"
HDB_MIN_CLUSTER_SIZE = int(os.getenv("HDB_MIN_CLUSTER_SIZE", "100"))
HDB_MIN_SAMPLES = int(os.getenv("HDB_MIN_SAMPLES", "20"))

RUN_GAP = os.getenv("RUN_GAP", "1") == "1"
GAP_B = int(os.getenv("GAP_B", "10"))

# Scenarios: transfused only, and transfused + non-transfused
SCENARIOS = [
    ("with_non_transfused", True),
]
NON_TRANSFUSED_T0_MINUTES = PRE_HOURS * 60
NON_TRANSFUSED_MAX = 0

GMM_MAX_ITER = int(os.getenv("GMM_MAX_ITER", "500"))
GMM_TOL = float(os.getenv("GMM_TOL", "1e-3"))
GMM_REG_COVAR = float(os.getenv("GMM_REG_COVAR", "1e-6"))
GMM_N_INIT = int(os.getenv("GMM_N_INIT", "5"))


def safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    mask = labels >= 0
    if mask.sum() == 0:
        return float("nan")
    uniq = np.unique(labels[mask])
    if uniq.size < 2:
        return float("nan")
    return float(silhouette_score(x[mask], labels[mask], metric="euclidean"))


def gap_statistic(
    x: np.ndarray,
    ks: List[int],
    b: int,
    random_state: int,
    label: str = "gap",
) -> List[Dict[str, float]]:
    rng = np.random.RandomState(random_state)
    mins = x.min(axis=0)
    maxs = x.max(axis=0)

    rows = []
    for k in ks:
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        km.fit(x)
        log_wk = np.log(km.inertia_)
        ref_logs = []
        for _ in range(b):
            xref = rng.uniform(low=mins, high=maxs, size=x.shape)
            km_ref = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km_ref.fit(xref)
            ref_logs.append(np.log(km_ref.inertia_))
        gap = float(np.mean(ref_logs) - log_wk)
        sdk = float(np.std(ref_logs, ddof=1) * np.sqrt(1 + 1 / b)) if b > 1 else float("nan")
        rows.append({"k": int(k), "gap": gap, "sdk": sdk})
        print(f"[{label}] k={k} gap={gap:.4f} sdk={sdk:.4f} time={time.time() - t0:.1f}s")
    return rows


def build_sequences(
    dataset,
    t0_map: Dict[int, int],
    stay_ids_cluster: List[int],
    feature_cols: List[str],
    seq_path: Path,
) -> tuple[np.memmap, np.ndarray]:
    stay_to_idx = {sid: i for i, sid in enumerate(stay_ids_cluster)}
    n_stays = len(stay_ids_cluster)
    n_feat = len(feature_cols)

    seq = np.memmap(seq_path, dtype="float32", mode="w+", shape=(n_stays, SEQ_LEN, n_feat))
    seq[:] = np.nan
    feat_obs = np.zeros(n_feat, dtype=np.int64)

    for stay_idx, rel_idx, values, _ in iter_window_batches(
        dataset=dataset,
        t0_map=t0_map,
        stay_to_idx=stay_to_idx,
        feature_cols=feature_cols,
        id_col=ID_COL,
        tbin_col=TBIN_COL,
        pre_bins=PRE_BINS,
        post_bins=POST_BINS,
        batch_size=BATCH_SIZE,
        progress_every=PROGRESS_EVERY,
        progress_label="build_seq",
    ):
        if values.size == 0:
            continue
        nan_mask = np.isnan(values)
        feat_obs += (~nan_mask).sum(axis=0)
        seq[stay_idx, rel_idx, :] = values

    col_has_obs = feat_obs > 0
    return seq, col_has_obs


def run_scenario(
    *,
    root: Path,
    outdir: Path,
    dataset,
    stay_ids_all: List[int],
    base_t0_map: Dict[int, int],
    include_non_transfused: bool,
    suffix: str,
) -> None:
    sys.path.insert(0, str(root / "ts2vec"))
    from ts2vec import TS2Vec  # type: ignore

    transfused_set = set(base_t0_map.keys())
    stay_ids_transfused = sorted(transfused_set)

    t0_map = dict(base_t0_map)
    stay_ids_cluster = stay_ids_transfused
    non_transfused: List[int] = []

    if include_non_transfused:
        non_transfused = [sid for sid in stay_ids_all if sid not in transfused_set]
        if NON_TRANSFUSED_MAX > 0 and len(non_transfused) > NON_TRANSFUSED_MAX:
            rng = np.random.RandomState(RANDOM_STATE)
            non_transfused = rng.choice(non_transfused, size=NON_TRANSFUSED_MAX, replace=False).tolist()
        for sid in non_transfused:
            t0_map[sid] = NON_TRANSFUSED_T0_MINUTES
        stay_ids_cluster = sorted(transfused_set.union(non_transfused))
        print(
            f"[cohort] include_non_transfused={len(non_transfused)} "
            f"t0_non_transfused={NON_TRANSFUSED_T0_MINUTES}min"
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
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for TS2Vec.")

    print(f"[features] base columns: {len(feature_cols)}")
    print(f"[features] transfused stays: {len(stay_ids_transfused)}")
    print(f"[features] clustering stays: {len(stay_ids_cluster)}")

    seq_path = Path(os.getenv("TS2VEC_SEQ_PATH", outdir / f"ts2vec_sequences_{suffix}.dat"))
    seq, col_has_obs = build_sequences(dataset, t0_map, stay_ids_cluster, feature_cols, seq_path)
    if not col_has_obs.any():
        raise RuntimeError("No usable features: all columns are NaN.")
    if (~col_has_obs).any():
        info_path = outdir / f"ts2vec_all_nan_features_{suffix}.json"
        all_nan_cols = [c for c, ok in zip(feature_cols, col_has_obs) if not ok]
        info_path.write_text(
            json.dumps({"all_nan_filled_zero": all_nan_cols}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        seq[:, :, ~col_has_obs] = 0.0
        print(f"[prep] filled {len(all_nan_cols)} all-NaN features with 0 -> {info_path}")
    print(f"[save] {seq_path}")

    device = os.getenv("TS2VEC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model = TS2Vec(
        input_dims=len(feature_cols),
        output_dims=OUTPUT_DIMS,
        hidden_dims=HIDDEN_DIMS,
        depth=DEPTH,
        device=device,
        lr=TS2VEC_LR,
        batch_size=TS2VEC_BATCH,
        max_train_length=TS2VEC_MAX_TRAIN_LENGTH,
    )

    print(f"[train] device={device} seq_shape={seq.shape}")
    model.fit(seq, n_iters=TS2VEC_ITERS, n_epochs=TS2VEC_EPOCHS, verbose=True)

    print("[encode] building embeddings...")
    t_enc = time.time()
    emb = model.encode(seq, mask="all_true", encoding_window="full_series")
    print(f"[encode] done in {time.time() - t_enc:.1f}s")
    emb_path = outdir / f"ts2vec_embeddings_{suffix}.npy"
    np.save(emb_path, emb)
    print(f"[save] {emb_path}")

    idx_path = outdir / f"ts2vec_embeddings_index_{suffix}.csv"
    pd.DataFrame({ID_COL: stay_ids_cluster}).to_csv(idx_path, index=False)
    print(f"[save] {idx_path}")

    # Clustering on embeddings
    x = emb.astype(np.float64, copy=False)
    n_samples = int(x.shape[0])
    t0 = time.time()
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    print(f"[prep] scale done in {time.time() - t0:.1f}s")

    pca = None
    if USE_PCA and PCA_NCOMP > 0 and PCA_NCOMP < x.shape[1]:
        pca = PCA(n_components=PCA_NCOMP, random_state=RANDOM_STATE)
        x = pca.fit_transform(x)
        print(f"[pca] ncomp={PCA_NCOMP} explained_var={pca.explained_variance_ratio_.sum():.4f}")

    metrics = {
        "n_samples": n_samples,
        "n_features": int(x.shape[1]),
        "window_hours": {"pre": PRE_HOURS, "post": POST_HOURS},
        "include_non_transfused": bool(include_non_transfused),
        "non_transfused_count": int(len(non_transfused)),
        "kmeans": {"scores": []},
        "gmm": {"scores": []},
        "hdbscan": {},
    }

    if n_samples < 2:
        print(f"[skip] only {n_samples} sample(s); skipping clustering for {suffix}")
        df_labels = pd.DataFrame({ID_COL: stay_ids_all})
        df_labels["has_transfusion"] = df_labels[ID_COL].isin(stay_ids_transfused).astype(int)
        df_labels["cluster_kmeans"] = np.nan
        df_labels["cluster_gmm"] = np.nan
        if RUN_HDBSCAN:
            df_labels["cluster_hdbscan"] = np.nan
        labels_path = outdir / f"ts2vec_clusters_{suffix}.csv"
        df_labels.to_csv(labels_path, index=False)
        print(f"[save] {labels_path}")
        metrics["skipped"] = "n_samples<2"
        metrics_path = outdir / "cluster_metrics.json"
        update_metrics_json(metrics_path, f"ts2vec_{suffix}", metrics)
        print(f"[save] {metrics_path}")
        return

    # KMeans
    k_list = [k for k in K_LIST if k <= n_samples]
    if not k_list:
        print(f"[skip] no valid k in K_LIST for n_samples={n_samples} ({suffix})")
        metrics["skipped"] = "no_valid_k"
        df_labels = pd.DataFrame({ID_COL: stay_ids_all})
        df_labels["has_transfusion"] = df_labels[ID_COL].isin(stay_ids_transfused).astype(int)
        df_labels["cluster_kmeans"] = np.nan
        df_labels["cluster_gmm"] = np.nan
        if RUN_HDBSCAN:
            df_labels["cluster_hdbscan"] = np.nan
        labels_path = outdir / f"ts2vec_clusters_{suffix}.csv"
        df_labels.to_csv(labels_path, index=False)
        print(f"[save] {labels_path}")
        metrics_path = outdir / "cluster_metrics.json"
        update_metrics_json(metrics_path, f"ts2vec_{suffix}", metrics)
        print(f"[save] {metrics_path}")
        return

    best_k = None
    best_sil = -float("inf")
    for k in k_list:
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
        labels = km.fit_predict(x)
        sil = safe_silhouette(x, labels)
        metrics["kmeans"]["scores"].append({"k": int(k), "silhouette": float(sil), "inertia": float(km.inertia_)})
        print(f"[kmeans] k={k} silhouette={sil:.4f} inertia={km.inertia_:.1f} time={time.time() - t0:.1f}s")
        if not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    if best_k is None:
        best_k = k_list[0]
    metrics["kmeans"]["best_k"] = int(best_k)
    if RUN_GAP:
        metrics["kmeans"]["gap"] = gap_statistic(x, k_list, GAP_B, RANDOM_STATE, label=f"gap_ts2vec_{suffix}")

    km = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE)
    kmeans_labels = km.fit_predict(x)

    # GMM
    best_gmm_k = None
    best_bic = float("inf")
    for k in k_list:
        t0 = time.time()
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            max_iter=GMM_MAX_ITER,
            tol=GMM_TOL,
            reg_covar=GMM_REG_COVAR,
            n_init=GMM_N_INIT,
        )
        labels = gmm.fit_predict(x)
        sil = safe_silhouette(x, labels)
        metrics["gmm"]["scores"].append(
            {"k": int(k), "silhouette": float(sil), "bic": float(gmm.bic(x)), "aic": float(gmm.aic(x))}
        )
        print(
            f"[gmm] k={k} silhouette={sil:.4f} bic={gmm.bic(x):.1f} "
            f"aic={gmm.aic(x):.1f} time={time.time() - t0:.1f}s"
        )
        if gmm.bic(x) < best_bic:
            best_bic = float(gmm.bic(x))
            best_gmm_k = k

    if best_gmm_k is None:
        best_gmm_k = k_list[0]
    metrics["gmm"]["best_k"] = int(best_gmm_k)

    gmm = GaussianMixture(
        n_components=best_gmm_k,
        covariance_type="full",
        random_state=RANDOM_STATE,
        max_iter=GMM_MAX_ITER,
        tol=GMM_TOL,
        reg_covar=GMM_REG_COVAR,
        n_init=GMM_N_INIT,
    )
    gmm_labels = gmm.fit_predict(x)

    # HDBSCAN
    hdb_labels = None
    if RUN_HDBSCAN and n_samples >= max(2, HDB_MIN_CLUSTER_SIZE):
        if not HAS_HDBSCAN:
            raise RuntimeError("hdbscan is not installed. Disable RUN_HDBSCAN or install it.")
        t0 = time.time()
        model_hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDB_MIN_CLUSTER_SIZE,
            min_samples=HDB_MIN_SAMPLES,
            metric="euclidean",
        )
        hdb_labels = model_hdb.fit_predict(x)
        print(f"[hdbscan] done in {time.time() - t0:.1f}s")
        n_noise = int((hdb_labels < 0).sum())
        n_clusters = int(np.unique(hdb_labels[hdb_labels >= 0]).size)
        metrics["hdbscan"] = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_frac": float(n_noise / len(hdb_labels)) if len(hdb_labels) else float("nan"),
            "silhouette": float(safe_silhouette(x, hdb_labels)),
        }
        if hdb_validity_index is not None and n_clusters >= 2:
            try:
                metrics["hdbscan"]["dbcv"] = float(hdb_validity_index(x, hdb_labels))
            except Exception:
                metrics["hdbscan"]["dbcv"] = float("nan")

    df_labels = pd.DataFrame({ID_COL: stay_ids_all})
    df_labels["has_transfusion"] = df_labels[ID_COL].isin(stay_ids_transfused).astype(int)
    df_labels["cluster_kmeans"] = df_labels[ID_COL].map(dict(zip(stay_ids_cluster, kmeans_labels)))
    df_labels["cluster_gmm"] = df_labels[ID_COL].map(dict(zip(stay_ids_cluster, gmm_labels)))
    if hdb_labels is not None:
        df_labels["cluster_hdbscan"] = df_labels[ID_COL].map(dict(zip(stay_ids_cluster, hdb_labels)))
    elif RUN_HDBSCAN:
        df_labels["cluster_hdbscan"] = np.nan

    labels_path = outdir / f"ts2vec_clusters_{suffix}.csv"
    df_labels.to_csv(labels_path, index=False)
    print(f"[save] {labels_path}")

    metrics_path = outdir / "cluster_metrics.json"
    update_metrics_json(metrics_path, f"ts2vec_{suffix}", metrics)
    print(f"[save] {metrics_path}")

    print(f"[done] ts2vec clustering complete ({suffix})")


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
        progress_label="scan_t0_ts2vec",
    )
    if not t0_map:
        raise RuntimeError("No transfused stays found (rbc_transfusion_flag == 1).")

    for suffix, include_non_transfused in SCENARIOS:
        run_scenario(
            root=root,
            outdir=outdir,
            dataset=dataset,
            stay_ids_all=stay_ids_all,
            base_t0_map=t0_map,
            include_non_transfused=include_non_transfused,
            suffix=suffix,
        )


if __name__ == "__main__":
    main()
