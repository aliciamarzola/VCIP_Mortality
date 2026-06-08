from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.common.pipeline_common import load_config, phenotypes_dir, processed_dir, save_table, write_json


def tensor_summary(tensor: np.ndarray) -> np.ndarray:
    mean = tensor.mean(axis=1)
    std = tensor.std(axis=1)
    min_v = tensor.min(axis=1)
    max_v = tensor.max(axis=1)
    last = tensor[:, -1, :]
    diff = tensor[:, -1, :] - tensor[:, 0, :]
    return np.concatenate([mean, std, min_v, max_v, last, diff], axis=1)


def build_embeddings(config) -> None:
    proc_dir = processed_dir(config)
    out_dir = phenotypes_dir(config)
    tensor_path = proc_dir / "temporal_tensor.npy"
    ids_path = proc_dir / "temporal_tensor_ids.parquet"
    if not tensor_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Missing temporal tensor outputs. Run preprocessing first.")

    tensor = np.load(tensor_path)
    ids = pd.read_parquet(ids_path)
    x = tensor_summary(tensor)
    x = StandardScaler().fit_transform(x)

    pca_cfg = config.get("temporal_representation", {}).get("pca", {})
    used_pca = bool(pca_cfg.get("enabled", True))
    if used_pca:
        n_components = min(int(pca_cfg.get("n_components", 12)), x.shape[0] - 1, x.shape[1])
        if n_components >= 2:
            pca = PCA(n_components=n_components, random_state=int(config.get("run", {}).get("seed", 42)))
            x = pca.fit_transform(x)
            explained = pca.explained_variance_ratio_.tolist()
        else:
            used_pca = False
            explained = []
    else:
        explained = []

    np.save(out_dir / "temporal_embeddings.npy", x)
    save_table(ids[["stay_id", "train_split"]], out_dir / "temporal_embedding_ids.parquet")
    write_json(
        out_dir / "temporal_representation_metadata.json",
        {
            "method": "temporal_summary_pca" if used_pca else "temporal_summary",
            "shape": list(x.shape),
            "pca_explained_variance_ratio": explained,
            "note": "Fallback representation based only on pre-t0 temporal windows.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal representations.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    build_embeddings(config)
    print("temporal_representation_complete")


if __name__ == "__main__":
    main()

