#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal


ID_COL = "stay_id"
MORT_COL = "mortality_anytime"
VM_COL = "vm_time_hours"


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "src").exists() and (parent / "outputs").exists():
            return parent
    return start


def read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path.suffix}")


def summarize_clusters(df: pd.DataFrame, cluster_col: str, source: str) -> tuple[pd.DataFrame, dict]:
    dfc = df[df[cluster_col].notna()].copy()
    dfc[cluster_col] = dfc[cluster_col].astype(int)

    if dfc.empty:
        return pd.DataFrame(), {"source": source, "cluster_col": cluster_col, "chi2_pvalue": np.nan, "kruskal_pvalue": np.nan}

    by = dfc.groupby(cluster_col, as_index=False)
    summary = by.agg(
        n=(ID_COL, "count"),
        deaths=(MORT_COL, "sum"),
        mortality_rate=(MORT_COL, "mean"),
        vm_mean=(VM_COL, "mean"),
        vm_median=(VM_COL, "median"),
    )
    summary.insert(0, "source", source)
    summary.insert(1, "cluster_col", cluster_col)
    summary = summary.rename(columns={cluster_col: "cluster"})

    # Tests (exclude noise -1 for HDBSCAN)
    df_test = dfc[dfc[cluster_col] >= 0]
    chi2_pvalue = np.nan
    kruskal_pvalue = np.nan

    if df_test[cluster_col].nunique() >= 2:
        cont = pd.crosstab(df_test[cluster_col], df_test[MORT_COL])
        if cont.shape[0] >= 2 and cont.shape[1] >= 2:
            _, p, _, _ = chi2_contingency(cont)
            chi2_pvalue = float(p)

        groups = [
            df_test.loc[df_test[cluster_col] == cl, VM_COL].dropna().values
            for cl in sorted(df_test[cluster_col].unique())
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            _, pkw = kruskal(*groups)
            kruskal_pvalue = float(pkw)

    tests = {
        "source": source,
        "cluster_col": cluster_col,
        "chi2_pvalue": chi2_pvalue,
        "kruskal_pvalue": kruskal_pvalue,
    }
    return summary, tests


def main() -> None:
    root = find_repo_root(Path(__file__).resolve())
    outdir = Path(os.getenv("CLUSTER_OUTDIR", root / "outputs" / "reports" / "clustering"))
    outdir.mkdir(parents=True, exist_ok=True)

    outcomes_path = Path(os.getenv("OUTCOMES_CSV", root / "outputs_outcomes" / "outcomes_by_stay.csv"))
    df_out = read_any(outcomes_path)
    if ID_COL not in df_out.columns:
        raise ValueError(f"Missing {ID_COL} in outcomes")
    if MORT_COL not in df_out.columns:
        raise ValueError(f"Missing {MORT_COL} in outcomes")
    if VM_COL not in df_out.columns:
        raise ValueError(f"Missing {VM_COL} in outcomes")

    df_out[MORT_COL] = df_out[MORT_COL].fillna(0).astype(int)

    cluster_files: List[Path] = []
    labels_env = os.getenv("CLUSTER_LABELS", "")
    if labels_env:
        cluster_files = [Path(p.strip()) for p in labels_env.split(",") if p.strip()]
    else:
        cluster_files = sorted(outdir.glob("*clusters*.csv"))

    if not cluster_files:
        raise FileNotFoundError("No cluster label files found. Set CLUSTER_LABELS or run clustering first.")

    all_summaries = []
    all_tests = []

    for labels_path in cluster_files:
        df_labels = read_any(labels_path)
        if ID_COL not in df_labels.columns:
            raise ValueError(f"Missing {ID_COL} in {labels_path}")

        df = df_labels.merge(df_out, on=ID_COL, how="left")
        df[MORT_COL] = df[MORT_COL].fillna(0).astype(int)
        df[VM_COL] = pd.to_numeric(df[VM_COL], errors="coerce")
        cluster_cols = [c for c in df_labels.columns if c.startswith("cluster_")]
        if not cluster_cols:
            continue

        source = labels_path.stem
        for cluster_col in cluster_cols:
            summary, tests = summarize_clusters(df, cluster_col, source)
            if not summary.empty:
                all_summaries.append(summary)
            all_tests.append(tests)

    if all_summaries:
        summary_df = pd.concat(all_summaries, ignore_index=True)
        summary_path = outdir / "cluster_outcome_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[save] {summary_path}")

    tests_df = pd.DataFrame(all_tests)
    tests_path = outdir / "cluster_outcome_tests.csv"
    tests_df.to_csv(tests_path, index=False)
    print(f"[save] {tests_path}")

    print("[done] cluster outcome analysis complete")


if __name__ == "__main__":
    main()
