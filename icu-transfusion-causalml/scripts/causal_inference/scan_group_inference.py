from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.pipeline_common import causal_dir, figures_dir, load_config, phenotypes_dir, resolve_path, write_json  # noqa: E402


DEFAULT_OUTCOMES = [
    "mortality_anytime",
    "vm_time_hours",
    "ventilation_hours",
    "rrt_on",
    "any_vasopressor",
    "nee_mcgkgmin_max",
    "icu_los_hours",
    "sofa_delta",
]


def _resolve_phenotype_run(phenotype_root: Path, run_id: str, window: int) -> tuple[Path, Path, Path]:
    if not phenotype_root.is_absolute():
        phenotype_root = (ROOT / phenotype_root).resolve()

    candidates = [
        phenotype_root / "runs" / run_id,
        phenotype_root / "outputs" / "runs" / run_id,
        phenotype_root / "outputs" / "phenotype_construction" / "runs" / run_id,
        ROOT / "outputs" / "runs" / run_id,
        ROOT / "outputs" / "phenotype_construction" / "runs" / run_id,
    ]
    run_dir = next((p for p in candidates if p.exists()), None)
    if run_dir is None:
        searched = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find phenotype-construction run {run_id}. Searched:\n{searched}")

    shared_dir = run_dir / "shared"
    window_dir = run_dir / f"w{window}"
    if not shared_dir.exists() or not window_dir.exists():
        raise FileNotFoundError(f"Invalid phenotype-construction run layout: {run_dir}")
    return run_dir, shared_dir, window_dir


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _parse_k_list(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted({k for k in out if k >= 2})


def _classify_effect(diff: float, n_pairs: int, benefit_threshold: float, harm_threshold: float, min_pairs: int) -> str:
    if n_pairs < min_pairs:
        return "unstable_small_n"
    if diff <= -abs(benefit_threshold):
        return "benefit_apparent"
    if diff >= abs(harm_threshold):
        return "harm_apparent"
    return "neutral_apparent"


def _reconstruct_assignments(embeddings: pd.DataFrame, k_values: list[int], seed: int) -> pd.DataFrame:
    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("Embeddings table has no emb_* columns.")

    x = StandardScaler().fit_transform(embeddings[emb_cols].to_numpy(dtype=float))
    assignments = embeddings[["stay_id"]].copy()
    assignments["stay_id"] = assignments["stay_id"].astype("int64")

    for k in k_values:
        if x.shape[0] <= k:
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        assignments[f"cluster_k{k}"] = model.fit_predict(x).astype(int)
    return assignments


def _build_scan_summary(
    benefit_table: pd.DataFrame,
    k_values: list[int],
    primary_outcome: str,
    benefit_threshold: float,
    harm_threshold: float,
    min_pairs: int,
) -> pd.DataFrame:
    primary = benefit_table[(benefit_table["metric"] == primary_outcome) & (benefit_table["k"].isin(k_values))].copy()
    if primary.empty:
        raise RuntimeError(f"No rows for primary outcome {primary_outcome} in cluster benefit table.")

    primary["scan_group"] = [
        _classify_effect(float(row.diff_mean), int(row.n_pairs), benefit_threshold, harm_threshold, min_pairs)
        for row in primary.itertuples(index=False)
    ]

    rows = []
    for row in primary.itertuples(index=False):
        subset = benefit_table[(benefit_table["k"] == row.k) & (benefit_table["cluster"] == row.cluster)].copy()
        support = subset[subset["metric"].isin(["vm_time_hours", "ventilation_hours", "rrt_on", "nee_mcgkgmin_max", "icu_los_hours"])]
        support_harm_count = int((support["diff_mean"] > 0).sum())
        rows.append(
            {
                "k": int(row.k),
                "cluster": int(row.cluster),
                "n_pairs": int(row.n_pairs),
                "primary_outcome": primary_outcome,
                "treated_mean": float(row.transfused_mean),
                "control_mean": float(row.control_mean),
                "effect": float(row.diff_mean),
                "scan_group": row.scan_group,
                "support_harm_count": support_harm_count,
            }
        )
    summary = pd.DataFrame(rows)
    return _add_article_oriented_labels(summary, benefit_table)


def _add_article_oriented_labels(summary: pd.DataFrame, benefit_table: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["article_oriented_label"] = summary["scan_group"]

    k2 = summary[summary["k"] == 2]
    if len(k2) == 2:
        burden_rows = []
        for row in k2.itertuples(index=False):
            sub = benefit_table[(benefit_table["k"] == 2) & (benefit_table["cluster"] == row.cluster)]
            vm = sub.loc[sub["metric"] == "ventilation_hours", "diff_mean"].mean()
            los = sub.loc[sub["metric"] == "icu_los_hours", "diff_mean"].mean()
            nee = sub.loc[sub["metric"] == "nee_mcgkgmin_max", "diff_mean"].mean()
            burden_rows.append({"cluster": int(row.cluster), "support_burden": float(np.nan_to_num(vm) + np.nan_to_num(los) / 24.0 + 24.0 * np.nan_to_num(nee))})
        burden = pd.DataFrame(burden_rows)
        high_burden = int(burden.sort_values("support_burden", ascending=False).iloc[0]["cluster"])
        low_burden = int(burden.sort_values("support_burden", ascending=True).iloc[0]["cluster"])
        summary.loc[(summary["k"] == 2) & (summary["cluster"] == low_burden), "article_oriented_label"] = "macro_benefit_relative"
        summary.loc[(summary["k"] == 2) & (summary["cluster"] == high_burden), "article_oriented_label"] = "macro_risk_high_support"

    return summary


def _attach_scan_labels(assignments: pd.DataFrame, scan_summary: pd.DataFrame) -> pd.DataFrame:
    out = assignments.copy()
    for k in sorted(scan_summary["k"].unique()):
        cluster_col = f"cluster_k{k}"
        if cluster_col not in out.columns:
            continue
        mapping = scan_summary[scan_summary["k"] == k].set_index("cluster")
        out[f"scan_group_k{k}"] = out[cluster_col].map(mapping["scan_group"])
        out[f"article_label_k{k}"] = out[cluster_col].map(mapping["article_oriented_label"])
    return out


def _paired_effect_frame(pairs: pd.DataFrame, cohort: pd.DataFrame, outcome: str) -> pd.DataFrame:
    outcomes = cohort[["stay_id", outcome]].drop_duplicates("stay_id")
    merged = pairs.merge(outcomes, left_on="stay_id_transf", right_on="stay_id", how="left").rename(columns={outcome: "y_t"})
    merged = merged.drop(columns=["stay_id"])
    merged = merged.merge(outcomes, left_on="stay_id_ctrl", right_on="stay_id", how="left").rename(columns={outcome: "y_c"})
    merged = merged.drop(columns=["stay_id"])
    return merged.dropna(subset=["y_t", "y_c"]).copy()


def _bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan, np.nan
    if len(diff) == 1 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _summarize_pairs(
    pairs: pd.DataFrame,
    cohort: pd.DataFrame,
    outcomes: list[str],
    level: str,
    group_name: str,
    k: int | None,
    cluster: int | None,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for outcome in outcomes:
        if outcome not in cohort.columns:
            continue
        merged = _paired_effect_frame(pairs, cohort, outcome)
        if merged.empty:
            continue
        diff = merged["y_t"].to_numpy(dtype=float) - merged["y_c"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(diff, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "level": level,
                "group": group_name,
                "k": k,
                "cluster": cluster,
                "outcome": outcome,
                "n_pairs": int(len(merged)),
                "n_treated": int(merged["stay_id_transf"].nunique()),
                "n_control": int(merged["stay_id_ctrl"].nunique()),
                "treated_mean": float(merged["y_t"].mean()),
                "control_mean": float(merged["y_c"].mean()),
                "effect": float(np.nanmean(diff)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return rows


def _scan_group_effects(
    assignments: pd.DataFrame,
    scan_summary: pd.DataFrame,
    pairs: pd.DataFrame,
    cohort: pd.DataFrame,
    outcomes: list[str],
    k_values: list[int],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = pairs.copy()
    pairs["stay_id_transf"] = pairs["stay_id_transf"].astype("int64")
    pairs["stay_id_ctrl"] = pairs["stay_id_ctrl"].astype("int64")

    rows.extend(_summarize_pairs(pairs, cohort, outcomes, "all_matched", "all", None, None, n_boot, seed))

    for k in k_values:
        cluster_col = f"cluster_k{k}"
        group_col = f"scan_group_k{k}"
        article_col = f"article_label_k{k}"
        if cluster_col not in assignments.columns:
            continue

        for cluster, sub_assign in assignments.groupby(cluster_col):
            sub_pairs = pairs[pairs["stay_id_transf"].isin(sub_assign["stay_id"])]
            if sub_pairs.empty:
                continue
            label = scan_summary[(scan_summary["k"] == k) & (scan_summary["cluster"] == int(cluster))]
            group = str(label["scan_group"].iloc[0]) if not label.empty else f"cluster_{cluster}"
            rows.extend(_summarize_pairs(sub_pairs, cohort, outcomes, "cluster", group, k, int(cluster), n_boot, seed))

        if group_col in assignments.columns:
            for group, sub_assign in assignments.dropna(subset=[group_col]).groupby(group_col):
                sub_pairs = pairs[pairs["stay_id_transf"].isin(sub_assign["stay_id"])]
                rows.extend(_summarize_pairs(sub_pairs, cohort, outcomes, "scan_group", str(group), k, None, n_boot, seed))

        if article_col in assignments.columns:
            for group, sub_assign in assignments.dropna(subset=[article_col]).groupby(article_col):
                sub_pairs = pairs[pairs["stay_id_transf"].isin(sub_assign["stay_id"])]
                rows.extend(_summarize_pairs(sub_pairs, cohort, outcomes, "article_oriented_group", str(group), k, None, n_boot, seed))

    return pd.DataFrame(rows)


def _ite_summary(assignments: pd.DataFrame, k_values: list[int], c_dir: Path) -> pd.DataFrame:
    ite_path = c_dir / "ite_estimates.parquet"
    if not ite_path.exists():
        ite_path = c_dir / "causal_forest_ite.parquet"
    if not ite_path.exists():
        return pd.DataFrame()
    ite = pd.read_parquet(ite_path)
    ite_cols = [c for c in ite.columns if c.startswith("ite_")]
    if not ite_cols:
        return pd.DataFrame()
    preferred = "ite_causal_forest" if "ite_causal_forest" in ite_cols else ite_cols[0]
    merged = ite.merge(assignments, on="stay_id", how="inner")
    rows = []
    for k in k_values:
        for col in [f"cluster_k{k}", f"scan_group_k{k}", f"article_label_k{k}"]:
            if col not in merged.columns:
                continue
            for group, sub in merged.dropna(subset=[col]).groupby(col):
                rows.append(
                    {
                        "level": col,
                        "group": group,
                        "k": k,
                        "n": int(len(sub)),
                        "ite_column": preferred,
                        "ite_mean": float(sub[preferred].mean()),
                        "ite_median": float(sub[preferred].median()),
                        "ite_sd": float(sub[preferred].std()),
                        "ite_p10": float(sub[preferred].quantile(0.10)),
                        "ite_p90": float(sub[preferred].quantile(0.90)),
                    }
                )
    return pd.DataFrame(rows)


def _plot_primary_effects(effects: pd.DataFrame, primary_outcome: str, fig_path: Path) -> None:
    view = effects[(effects["outcome"] == primary_outcome) & (effects["level"].isin(["cluster", "article_oriented_group"]))].copy()
    view = view.dropna(subset=["effect"])
    if view.empty:
        return
    view["label"] = view.apply(lambda r: f"{r['level']} k={int(r['k']) if pd.notna(r['k']) else 'all'} {r['group']}" + (f" c={int(r['cluster'])}" if pd.notna(r["cluster"]) else ""), axis=1)
    view = view.sort_values(["k", "level", "effect"], na_position="first").tail(35)
    y = np.arange(len(view))
    xerr = np.vstack([(view["effect"] - view["ci95_low"]).to_numpy(), (view["ci95_high"] - view["effect"]).to_numpy()])
    xerr = np.where(np.isfinite(xerr), xerr, 0)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, max(5, len(view) * 0.28)))
    plt.errorbar(view["effect"], y, xerr=xerr, fmt="o", color="#2f5d8c", ecolor="#7b8da0", capsize=2)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, view["label"])
    plt.xlabel("Risk difference, treated minus matched control")
    plt.title("Scan groups: paired effect on mortality")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()


def _write_report(
    path: Path,
    run_id: str,
    window: int,
    embedding: str,
    primary_outcome: str,
    scan_summary: pd.DataFrame,
    effects: pd.DataFrame,
    ite_summary: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary_effects = effects[(effects["outcome"] == primary_outcome) & (effects["level"].isin(["article_oriented_group", "scan_group", "cluster"]))].copy()
    primary_effects = primary_effects.sort_values(["level", "k", "effect"], na_position="last")

    def table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
        if df.empty:
            return "_Nao disponivel._"
        view = df[cols].head(max_rows).copy()
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Scan Groups e Inferencia Causal por Fenotipo",
        "",
        f"Run legado: `{run_id}`.",
        f"Janela: {window}h.",
        f"Embedding: `{embedding}`.",
        "",
        "## Racional",
        "",
        "Esta analise usa os grupos descobertos no scan do artigo anterior como candidatos de heterogeneidade de efeito. O grupo e definido pelo cluster do paciente transfundido; o controle entra pelo par pareado.",
        "",
        "O `K=2` e tratado como macrofenotipo orientado pelo artigo. Valores maiores de K sao usados como analise secundaria/exploratoria para separar beneficio, neutralidade e risco aparente.",
        "",
        "## Grupos do Scan",
        "",
        table(scan_summary, ["k", "cluster", "n_pairs", "effect", "scan_group", "article_oriented_label", "support_harm_count"], max_rows=40),
        "",
        "## Efeito Pareado por Grupo",
        "",
        table(primary_effects, ["level", "group", "k", "cluster", "n_pairs", "treated_mean", "control_mean", "effect", "ci95_low", "ci95_high"], max_rows=60),
        "",
        "## ITE por Grupo",
        "",
        table(ite_summary, ["level", "group", "k", "n", "ite_mean", "ite_median", "ite_p10", "ite_p90"], max_rows=60),
        "",
        "## Interpretacao",
        "",
        "- `benefit_apparent`: mortalidade menor nos transfundidos pareados.",
        "- `harm_apparent`: mortalidade maior nos transfundidos pareados.",
        "- `neutral_apparent`: diferenca pequena, dentro do limiar configurado.",
        "- `macro_risk_high_support`: rotulo orientado pelo artigo para o macrogrupo com maior carga de suporte organico, mesmo que o sinal de mortalidade isolado nao seja francamente positivo.",
        "",
        "Esses grupos devem ser tratados como hipoteses geradas pelo estudo anterior. A inferencia causal mais forte deve estimar efeitos nesses grupos com features estritamente pre-t0, diagnostico de overlap, balanceamento e bootstrap.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scan_group_inference(args: argparse.Namespace) -> None:
    config = load_config(args.config_dir)
    k_values = _parse_k_list(args.k_list)
    run_dir, shared_dir, window_dir = _resolve_phenotype_run(Path(args.phenotype_root), args.run_id, args.window)

    embeddings_path = _required(window_dir / "embeddings" / f"embeddings_{args.embedding}.parquet")
    benefit_path = _required(window_dir / "reports" / f"cluster_benefit_{args.embedding}.csv")
    matched_path = _required(window_dir / "matching" / "matched_pairs.parquet")
    cohort_path = _required(shared_dir / "outcomes_cohort.parquet")

    print(f"[scan] run_dir={run_dir}", flush=True)
    print(f"[scan] reading embeddings: {embeddings_path}", flush=True)
    embeddings = pd.read_parquet(embeddings_path)
    print(f"[scan] reading benefit table: {benefit_path}", flush=True)
    benefit_table = pd.read_csv(benefit_path)
    print(f"[scan] reading matched pairs: {matched_path}", flush=True)
    pairs = pd.read_parquet(matched_path)
    print(f"[scan] reading outcomes cohort: {cohort_path}", flush=True)
    cohort = pd.read_parquet(cohort_path)

    if "window_hours" in pairs.columns:
        pairs = pairs[pairs["window_hours"] == args.window].copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")

    outcomes = [o for o in DEFAULT_OUTCOMES if o in cohort.columns or o in benefit_table["metric"].unique()]
    if args.outcomes:
        requested = [x.strip() for x in args.outcomes.split(",") if x.strip()]
        outcomes = [o for o in requested if o in cohort.columns]
    if args.primary_outcome not in outcomes:
        outcomes = [args.primary_outcome] + outcomes

    assignments = _reconstruct_assignments(embeddings, k_values=k_values, seed=args.seed)
    scan_summary = _build_scan_summary(
        benefit_table,
        k_values=k_values,
        primary_outcome=args.primary_outcome,
        benefit_threshold=args.benefit_threshold,
        harm_threshold=args.harm_threshold,
        min_pairs=args.min_pairs,
    )
    assignments = _attach_scan_labels(assignments, scan_summary)

    p_dir = phenotypes_dir(config)
    c_dir = causal_dir(config)
    f_dir = figures_dir(config)
    report_path = resolve_path(config, "scan_group_report_file", "outputs/causal_inference/reports/scan_group_inference_report.md")

    assignments.to_parquet(p_dir / "scan_group_assignments.parquet", index=False)
    scan_summary.to_csv(p_dir / "scan_group_summary.csv", index=False)
    scan_summary.to_csv(c_dir / "scan_group_summary.csv", index=False)

    effects = _scan_group_effects(
        assignments=assignments,
        scan_summary=scan_summary,
        pairs=pairs,
        cohort=cohort,
        outcomes=outcomes,
        k_values=k_values,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    effects.to_csv(c_dir / "effect_by_scan_group.csv", index=False)

    ite = _ite_summary(assignments, k_values=k_values, c_dir=c_dir)
    if not ite.empty:
        ite.to_csv(c_dir / "ite_by_scan_group.csv", index=False)

    _plot_primary_effects(effects, args.primary_outcome, f_dir / "scan_group_mortality_effects.png")
    _write_report(
        report_path,
        run_id=args.run_id,
        window=args.window,
        embedding=args.embedding,
        primary_outcome=args.primary_outcome,
        scan_summary=scan_summary,
        effects=effects,
        ite_summary=ite,
    )
    write_json(
        p_dir / "scan_group_metadata.json",
        {
            "phenotype_run_dir": str(run_dir),
            "run_id": args.run_id,
            "window": args.window,
            "embedding": args.embedding,
            "k_values": k_values,
            "primary_outcome": args.primary_outcome,
            "benefit_threshold": args.benefit_threshold,
            "harm_threshold": args.harm_threshold,
            "min_pairs": args.min_pairs,
            "bootstrap": args.bootstrap,
            "note": "Groups are reconstructed from phenotype-construction embeddings using the same KMeans procedure used in the phenotype report script.",
        },
    )

    print("[scan] outputs written:", flush=True)
    print(f"[scan] {p_dir / 'scan_group_assignments.parquet'}", flush=True)
    print(f"[scan] {p_dir / 'scan_group_summary.csv'}", flush=True)
    print(f"[scan] {c_dir / 'effect_by_scan_group.csv'}", flush=True)
    print(f"[scan] {report_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct article scan groups and estimate paired effects by group.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--phenotype-root", default="outputs/phenotype_construction")
    parser.add_argument("--run-id", default="run_cal03_replace_full_w48")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--embedding", choices=["minirocket", "ts2vec"], default="minirocket")
    parser.add_argument("--k-list", default="2,3,5,6")
    parser.add_argument("--primary-outcome", default="mortality_anytime")
    parser.add_argument("--outcomes", default=None, help="Optional comma-separated outcome list.")
    parser.add_argument("--benefit-threshold", type=float, default=0.02)
    parser.add_argument("--harm-threshold", type=float, default=0.02)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scan_group_inference(args)


if __name__ == "__main__":
    main()
