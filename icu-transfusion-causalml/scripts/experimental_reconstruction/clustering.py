from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.common.pipeline_common import figures_dir, load_config, phenotypes_dir, save_table


def _safe_metric(func, x, labels):
    try:
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(func(x, labels))
    except Exception:
        return np.nan


def run_clustering(config) -> None:
    out_dir = phenotypes_dir(config)
    fig_dir = figures_dir(config)
    x = np.load(out_dir / "temporal_embeddings.npy")
    ids = pd.read_parquet(out_dir / "temporal_embedding_ids.parquet")
    x = StandardScaler().fit_transform(x)

    cluster_cfg = config.get("clustering", {})
    k_min = int(cluster_cfg.get("k_min", 2))
    k_max = int(cluster_cfg.get("k_max", 8))
    seeds = cluster_cfg.get("seeds", [42])
    threshold = int(cluster_cfg.get("minibatch_threshold", 5000))
    k_values = [k for k in range(k_min, k_max + 1) if 1 < k < x.shape[0]]

    rows = []
    labels_by_key = {}
    for k in k_values:
        for seed in seeds:
            model_cls = MiniBatchKMeans if x.shape[0] >= threshold else KMeans
            model = model_cls(n_clusters=k, random_state=int(seed), n_init=20)
            labels = model.fit_predict(x)
            labels_by_key[(k, int(seed))] = labels
            counts = pd.Series(labels).value_counts().sort_index()
            rows.append(
                {
                    "k": k,
                    "seed": int(seed),
                    "n_samples": int(x.shape[0]),
                    "inertia": float(model.inertia_),
                    "silhouette": _safe_metric(silhouette_score, x, labels),
                    "davies_bouldin": _safe_metric(davies_bouldin_score, x, labels),
                    "calinski_harabasz": _safe_metric(calinski_harabasz_score, x, labels),
                    "min_cluster_size": int(counts.min()),
                    "max_cluster_size": int(counts.max()),
                }
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No valid k values for clustering.")

    summary = metrics.groupby("k", as_index=False).agg(
        silhouette=("silhouette", "mean"),
        silhouette_sd=("silhouette", "std"),
        davies_bouldin=("davies_bouldin", "mean"),
        calinski_harabasz=("calinski_harabasz", "mean"),
        inertia=("inertia", "mean"),
        min_cluster_size=("min_cluster_size", "min"),
        max_cluster_size=("max_cluster_size", "max"),
    )
    selected_k_cfg = cluster_cfg.get("selected_k", "auto")
    if selected_k_cfg == "auto":
        selected_k = int(summary.sort_values(["silhouette", "min_cluster_size"], ascending=False).iloc[0]["k"])
    else:
        selected_k = int(selected_k_cfg)
    seed = int(config.get("run", {}).get("seed", 42))
    if (selected_k, seed) not in labels_by_key:
        seed = int(seeds[0])
    labels = labels_by_key[(selected_k, seed)]

    clusters = ids[["stay_id"]].copy()
    clusters["phenotype"] = labels.astype(int)
    clusters["selected_k"] = selected_k
    save_table(clusters, out_dir / "clusters.parquet")
    save_table(metrics, out_dir / "cluster_metrics_by_seed.csv")
    save_table(summary, out_dir / "cluster_metrics.csv")

    plt.figure(figsize=(7, 4))
    plt.plot(summary["k"], summary["silhouette"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.title("Silhouette by K")
    plt.tight_layout()
    plt.savefig(fig_dir / "silhouette_by_k.png", dpi=160)
    plt.close()

    sizes = clusters["phenotype"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(sizes.index.astype(str), sizes.values)
    plt.xlabel("Phenotype")
    plt.ylabel("N")
    plt.title("Cluster Sizes")
    plt.tight_layout()
    plt.savefig(fig_dir / "cluster_sizes.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster temporal representations.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    run_clustering(config)
    print("clustering_complete")


if __name__ == "__main__":
    main()

