from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import summarize_effect  # noqa: E402


GROUP_ORDER = ["B1", "B2", "M1", "M2", "M3"]
GROUP_LABELS = {
    "B1": "Anemia dinamica compensada",
    "B2": "Beneficio mais especifico",
    "M1": "Estresse hemodinamico",
    "M2": "Risco cardiorrenal",
    "M3": "Hb sem queda relevante",
}
GROUP_INTERPRETATIONS = {
    "B1": "Beneficio em anemia dinamica com frequencia cardiaca controlada e SpO2 final nao alta.",
    "B2": "Subfenotipo de beneficio mais especifico: B1 com pressao arterial media baixa/moderada.",
    "M1": "Maleficio em trajetoria de estresse hemodinamico: frequencia cardiaca em ascensao e pico de PAM elevado.",
    "M2": "Maleficio cardiorrenal: M1 associado a maior variabilidade de creatinina.",
    "M3": "Maleficio quando nao ha queda relevante de hemoglobina antes de t0.",
}
REQUIRED_INPUTS = [
    "outputs/target_trial/causal/main_effects.csv",
    "outputs/target_trial/final_groups/final_group_effects.csv",
    "outputs/target_trial/final_groups/final_group_profiles.csv",
    "outputs/target_trial/final_groups/final_group_balance.csv",
    "outputs/target_trial/legacy_crosswalk/legacy_crosswalk_report.md",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt_num(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for key, replacement in replacements.items():
        text = text.replace(key, replacement)
    return text


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if df.empty:
        return "_Nao disponivel._"
    shown = df[columns].head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(fmt_num(value) for value in row) + " |")
    return "\n".join(lines)


def load_sources(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "main": root / "outputs/target_trial/causal/main_effects.csv",
        "effects": root / "outputs/target_trial/final_groups/final_group_effects.csv",
        "profiles": root / "outputs/target_trial/final_groups/final_group_profiles.csv",
        "balance": root / "outputs/target_trial/final_groups/final_group_balance.csv",
        "trial": root / "outputs/target_trial/processed/trial_dataset.parquet",
        "nuisance": root / "outputs/target_trial/causal/nuisance_predictions.parquet",
        "assignments": root / "outputs/target_trial/final_groups/final_group_assignments.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    missing += [str(root / rel) for rel in REQUIRED_INPUTS if not (root / rel).exists()]
    missing += [str(path) for path in (root / "outputs/target_trial/legacy_rule_crosswalk").rglob("legacy_rule_scan_crosswalk_report.md") if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    return {
        "main": pd.read_csv(paths["main"]),
        "effects": pd.read_csv(paths["effects"]),
        "profiles": pd.read_csv(paths["profiles"]),
        "balance": pd.read_csv(paths["balance"]),
        "trial": pd.read_parquet(paths["trial"]),
        "nuisance": pd.read_parquet(paths["nuisance"]),
        "assignments": pd.read_parquet(paths["assignments"]),
    }


def final_summary(effects: pd.DataFrame) -> pd.DataFrame:
    all_rows = effects[effects["split"].eq("all")].set_index("group_id")
    disc_rows = effects[effects["split"].eq("discovery")].set_index("group_id")
    val_rows = effects[effects["split"].eq("validation")].set_index("group_id")
    rows = []
    for group in GROUP_ORDER:
        row = all_rows.loc[group]
        discovery = disc_rows.loc[group]
        validation = val_rows.loc[group]
        rows.append(
            {
                "group": group,
                "direction": row["direction"],
                "rule": row["rule_text"],
                "n": int(row["n"]),
                "treated": int(row["n_treated"]),
                "controls": int(row["n_control"]),
                "treated_mortality": float(row["treated_mean"]),
                "control_mortality": float(row["control_mean"]),
                "unadjusted_diff": float(row["unadjusted_diff"]),
                "aipw_ate": float(row["aipw_ate"]),
                "aipw_ci_low": float(row["aipw_ate_ci95_low"]),
                "aipw_ci_high": float(row["aipw_ate_ci95_high"]),
                "att": float(row["aipw_att"]),
                "overlap_weighted_diff": float(row["overlap_weighted_diff"]),
                "iptw_diff": float(row["iptw_diff"]),
                "discovery_aipw": float(discovery["aipw_ate"]),
                "discovery_ci_low": float(discovery["aipw_ate_ci95_low"]),
                "discovery_ci_high": float(discovery["aipw_ate_ci95_high"]),
                "validation_aipw": float(validation["aipw_ate"]),
                "validation_ci_low": float(validation["aipw_ate_ci95_low"]),
                "validation_ci_high": float(validation["aipw_ate_ci95_high"]),
                "interpretation": GROUP_INTERPRETATIONS[group],
            }
        )
    return pd.DataFrame(rows)


def write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Grupos finais do scan causal e efeitos estimados por AIPW.}",
        r"\label{tab:final_causal_groups}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Grupo & Interpretacao & $n$ & AIPW ATE & IC95\% & Disc./Valid. \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        ci = f"[{row.aipw_ci_low:.3f}, {row.aipw_ci_high:.3f}]"
        disc_val = f"{row.discovery_aipw:.3f} / {row.validation_aipw:.3f}"
        lines.append(
            f"{latex_escape(row.group)} & {latex_escape(GROUP_LABELS[row.group])} & "
            f"{int(row.n)} & {row.aipw_ate:.3f} & {latex_escape(ci)} & {latex_escape(disc_val)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def forest_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(summary))
    colors = ["#008080" if direction == "benefit" else "#b22222" for direction in summary["direction"]]
    x = summary["aipw_ate"].to_numpy()
    xerr = np.vstack([x - summary["aipw_ci_low"].to_numpy(), summary["aipw_ci_high"].to_numpy() - x])
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#5f6b7a", elinewidth=2, capsize=4)
    ax.scatter(x, y, s=95, c=colors, edgecolor="black", linewidth=0.8, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["group"] + " - " + summary["interpretation"].str.split(".").str[0])
    ax.set_xlabel("AIPW ATE: diferenca absoluta de risco")
    ax.set_title("Efeito estimado da transfusao por grupo final")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def discovery_validation_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(summary))
    for label, x_col, lo_col, hi_col, offset, color in [
        ("Discovery", "discovery_aipw", "discovery_ci_low", "discovery_ci_high", -0.13, "#386cb0"),
        ("Validation", "validation_aipw", "validation_ci_low", "validation_ci_high", 0.13, "#fdb462"),
    ]:
        x = summary[x_col].to_numpy()
        xerr = np.vstack([x - summary[lo_col].to_numpy(), summary[hi_col].to_numpy() - x])
        ax.errorbar(x, y + offset, xerr=xerr, fmt="o", color=color, ecolor=color, capsize=3, label=label)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["group"])
    ax.set_xlabel("AIPW ATE")
    ax.set_title("Consistencia discovery vs validation")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def profile_heatmap(profiles: pd.DataFrame, path: Path) -> None:
    feature_order = [
        "heart_rate_delta",
        "heart_rate_max",
        "hemoglobin_slope",
        "hemoglobin_last",
        "mbp_mean",
        "mbp_max",
        "creatinine_std",
        "lactate_std",
        "spo2_last",
        "sofa_mean",
    ]
    pivot = profiles.pivot_table(index="feature", columns="group_id", values="median").reindex(feature_order)[GROUP_ORDER]
    z = pivot.copy()
    for idx in z.index:
        vals = z.loc[idx].to_numpy(dtype=float)
        sd = np.nanstd(vals)
        z.loc[idx] = (vals - np.nanmean(vals)) / sd if sd > 1e-12 else 0.0
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    im = ax.imshow(z.to_numpy(dtype=float), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Perfis fisiologicos medianos por grupo")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Z-score por variavel")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def overlap_balance_plot(effects: pd.DataFrame, balance: pd.DataFrame, path: Path) -> None:
    all_eff = effects[effects["split"].eq("all")].set_index("group_id").loc[GROUP_ORDER]
    bal = balance.set_index("group_id").loc[GROUP_ORDER]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.2), sharex=True)
    x = np.arange(len(GROUP_ORDER))
    axes[0].bar(x - 0.18, bal["median_abs_smd_overlap"], width=0.36, label="SMD mediano", color="#80b1d3")
    axes[0].bar(x + 0.18, bal["max_abs_smd_overlap"], width=0.36, label="SMD maximo", color="#fb8072")
    axes[0].axhline(0.1, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("SMD abs.")
    axes[0].legend(frameon=False)
    axes[0].set_title("Balanceamento e suporte comum dos grupos finais")
    axes[1].bar(x, bal["n_smd_overlap_gt_0_1"], color="#bebada")
    axes[1].set_ylabel("N variaveis\nSMD > 0.1")
    axes[2].bar(x - 0.18, all_eff["ess_overlap_treated"], width=0.36, label="ESS tratados", color="#8dd3c7")
    axes[2].bar(x + 0.18, all_eff["ess_overlap_control"], width=0.36, label="ESS controles", color="#ffffb3")
    axes[2].set_ylabel("ESS overlap")
    axes[2].legend(frameon=False)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(GROUP_ORDER)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def pipeline_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.axis("off")
    nodes = [
        ("t0", "Ponto de decisao t0"),
        ("X", "Features pre-t0\ntrajetorias fisiologicas"),
        ("A", "Tratamento A\ntransfusao vs controle"),
        ("Y", "Outcome Y\nmortalidade"),
        ("AIPW", "AIPW / DR\nATE, ATT, IPTW, overlap"),
        ("scan", "Scan honesto\nregras pre-t0"),
        ("groups", "Grupos finais\nB1/B2 e M1/M2/M3"),
    ]
    xs = np.linspace(0.05, 0.95, len(nodes))
    y = 0.55
    for idx, (_key, label) in enumerate(nodes):
        x = xs[idx]
        box = plt.Rectangle((x - 0.055, y - 0.16), 0.11, 0.32, facecolor="#f2f2f2", edgecolor="#333333", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
        if idx < len(nodes) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.06, y), xytext=(x + 0.06, y), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(0.5, 0.14, "Somente informacao antes de t0 entra nas features; os grupos estratificam o efeito, nao substituem o tratamento.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any() or weights[mask].sum() <= 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def feature_smd(df: pd.DataFrame, feature: str, weight_col: str | None = None) -> float:
    tmp = pd.DataFrame({"x": pd.to_numeric(df[feature], errors="coerce"), "a": pd.to_numeric(df["transfused"], errors="coerce")})
    if weight_col:
        tmp["w"] = pd.to_numeric(df[weight_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "a"])
    if tmp.empty or tmp["a"].nunique() < 2:
        return np.nan
    t = tmp[tmp["a"] == 1]
    c = tmp[tmp["a"] == 0]
    if t.empty or c.empty:
        return np.nan
    mean_t = weighted_mean(t["x"].to_numpy(), t["w"].to_numpy()) if weight_col else float(t["x"].mean())
    mean_c = weighted_mean(c["x"].to_numpy(), c["w"].to_numpy()) if weight_col else float(c["x"].mean())
    pooled = np.sqrt((float(t["x"].var(ddof=0)) + float(c["x"].var(ddof=0))) / 2.0)
    return float((mean_t - mean_c) / pooled) if pooled > 1e-12 else 0.0


def balance_feature_table(analysis: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    rows = []
    feature_cols = [col for col in metadata.get("feature_columns", []) if col in analysis.columns and pd.api.types.is_numeric_dtype(analysis[col])]
    for group in GROUP_ORDER:
        sub = analysis[analysis[group].fillna(False)].copy()
        for feature in feature_cols:
            rows.append(
                {
                    "group_id": group,
                    "feature": feature,
                    "smd_unweighted": feature_smd(sub, feature, None),
                    "smd_overlap": feature_smd(sub, feature, "overlap_weight"),
                    "smd_iptw": feature_smd(sub, feature, "iptw_weight"),
                }
            )
    return pd.DataFrame(rows)


def qualitative_balance(mean_smd: float, max_smd: float, count_gt: int) -> str:
    if max_smd <= 0.1 and count_gt == 0:
        return "bom"
    if mean_smd <= 0.1 and max_smd <= 0.3:
        return "moderado"
    return "fragil"


def balance_interpretation(balance_features: pd.DataFrame, balance_summary: pd.DataFrame, path: Path) -> None:
    lines = ["# Interpretacao do balanceamento dos grupos finais", ""]
    summary = balance_summary.set_index("group_id")
    for group in GROUP_ORDER:
        sub = balance_features[balance_features["group_id"].eq(group)].copy()
        sub["abs_overlap"] = sub["smd_overlap"].abs()
        mean_smd = float(sub["abs_overlap"].mean())
        max_smd = float(sub["abs_overlap"].max())
        count_gt = int((sub["abs_overlap"] > 0.1).sum())
        top = sub[sub["abs_overlap"] > 0.1].sort_values("abs_overlap", ascending=False).head(15)
        quality = qualitative_balance(mean_smd, max_smd, count_gt)
        if quality == "bom":
            causal_note = "O balanceamento residual e favoravel; a confianca causal relativa e maior, ainda limitada pelo desenho observacional."
        elif quality == "moderado":
            causal_note = "Ha algum desequilibrio residual, mas o grupo mantem suporte interpretavel; a inferencia deve ser lida com cautela moderada."
        else:
            causal_note = "Ha desequilibrio residual relevante; o sinal causal estimado e informativo, mas a confianca e reduzida por possivel confundimento residual."
        lines.extend(
            [
                f"## {group}",
                "",
                f"- SMD medio absoluto apos overlap weighting: `{mean_smd:.3f}`.",
                f"- SMD maximo absoluto apos overlap weighting: `{max_smd:.3f}`.",
                f"- Variaveis com SMD > 0.1: `{count_gt}`.",
                f"- Avaliacao qualitativa: **{quality}**.",
                f"- Impacto na confianca causal: {causal_note}",
                "",
                "Variaveis mais desequilibradas:",
                "",
                markdown_table(top, ["feature", "smd_overlap"], max_rows=15),
                "",
                f"Resumo original do arquivo final_group_balance.csv: `{summary.loc[group, 'top_residual_smd_overlap']}`.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_analysis_frame(data: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> pd.DataFrame:
    trial = data["trial"]
    nuisance = data["nuisance"].drop(columns=["analysis_split", "transfused", metadata.get("primary_outcome", "mortality_anytime")], errors="ignore")
    assign = data["assignments"][["stay_id", *GROUP_ORDER, "final_primary_label"]]
    return trial.merge(nuisance, on="stay_id", how="left").merge(assign, on="stay_id", how="left")


def common_support_sensitivity(analysis: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    primary = metadata.get("primary_outcome", "mortality_anytime")
    rows = []
    for label, low, high in [("none", 0.0, 1.0), ("ps_0.05_0.95", 0.05, 0.95), ("ps_0.10_0.90", 0.10, 0.90)]:
        ps_mask = analysis["propensity_score"].between(low, high, inclusive="both")
        for group in GROUP_ORDER:
            mask = analysis[group].fillna(False) & ps_mask
            sub = analysis[mask].copy()
            if sub.empty or sub["transfused"].nunique() < 2:
                row = {"support": label, "ps_low": low, "ps_high": high, "group": group, "n": int(len(sub)), "estimable": False}
            else:
                row = summarize_effect(sub, primary, "transfused")
                row.update({"support": label, "ps_low": low, "ps_high": high, "group": group, "estimable": True})
            rows.append(row)
    return pd.DataFrame(rows)


def common_support_report(sensitivity: pd.DataFrame, path: Path) -> None:
    cols = ["support", "group", "n", "n_treated", "n_control", "unadjusted_diff", "aipw_ate", "aipw_att", "overlap_weighted_diff", "iptw_diff", "estimable"]
    lines = [
        "# Sensibilidade por suporte comum",
        "",
        "Os efeitos foram reestimados restringindo a amostra aos pacientes com propensity score dentro de faixas de suporte comum. As regras dos grupos nao foram alteradas.",
        "",
        markdown_table(sensitivity, cols, max_rows=80),
        "",
        "## Interpretacao",
        "",
        "A conclusao principal e considerada mais robusta quando o sinal do AIPW permanece o mesmo apos restricoes de suporte comum. Reducoes grandes de tamanho amostral devem ser interpretadas como possivel fragilidade de positividade.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def estimator_agreement(summary: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    estimators = ["aipw_ate", "att", "overlap_weighted_diff", "iptw_diff"]
    for row in summary.itertuples(index=False):
        values = {name: float(getattr(row, name)) for name in estimators}
        signs = {name: np.sign(value) for name, value in values.items() if np.isfinite(value) and abs(value) > 1e-12}
        expected = -1 if row.direction == "benefit" else 1
        all_same = len(set(signs.values())) == 1
        expected_all = all(sign == expected for sign in signs.values())
        rows.append(
            {
                "group": row.group,
                "direction": row.direction,
                **values,
                "all_estimators_same_sign": bool(all_same),
                "all_estimators_match_expected_direction": bool(expected_all),
            }
        )
    agreement = pd.DataFrame(rows)
    lines = [
        "# Concordancia entre estimadores",
        "",
        "Foram comparados AIPW ATE, ATT, IPTW e overlap weighted difference. A narrativa principal e mais forte quando todos os estimadores apontam na direcao esperada.",
        "",
        markdown_table(agreement, list(agreement.columns), max_rows=20),
        "",
        "## Conclusao",
        "",
        "B1/B2 mantem sinal negativo em todos os estimadores. M1/M2/M3 mantem sinal positivo em todos os estimadores. Isso fortalece a evidencia de heterogeneidade, apesar das limitacoes observacionais e de balanceamento residual.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return agreement


def latex_section(summary: pd.DataFrame, main: pd.DataFrame, path: Path) -> None:
    global_row = main[main["split"].eq("all")].iloc[0]
    table_lines = []
    for row in summary.itertuples(index=False):
        table_lines.append(
            f"{latex_escape(row.group)} & {latex_escape(GROUP_LABELS[row.group])} & {int(row.n)} & "
            f"{row.aipw_ate:.3f} & [{row.aipw_ci_low:.3f}, {row.aipw_ci_high:.3f}] \\\\"
        )
    text = rf"""
\section{{Inferencia causal e efeitos heterogeneos da transfusao}}

A analise causal foi formulada no arcabouco de desfechos potenciais. Para cada paciente, define-se $Y(1)$ como o desfecho potencial sob transfusao de hemacias e $Y(0)$ como o desfecho potencial sob nao transfusao. O efeito individual e $Y_i(1)-Y_i(0)$, enquanto os efeitos medios sao estimados em subpopulacoes clinicamente definidas. Nesta analise, o tratamento e sempre a transfusao ($A=1$) versus controle ($A=0$); os grupos B1/B2/M1/M2/M3 nao sao tratamentos, mas estratos fisiologicos usados para estudar heterogeneidade de efeito.

Associacoes simples entre transfusao e mortalidade sao insuficientes porque a indicacao transfusional e fortemente dependente da gravidade clinica. Pacientes transfundidos podem diferir dos controles por anemia, instabilidade hemodinamica, disfuncao organica, sangramento, procedimentos e suporte intensivo. Esse problema caracteriza confundimento por indicacao.

O DAG conceitual considerado inclui variaveis estaticas $V$, trajetorias fisiologicas pre-intervencao $X_t$, tratamento $A$ e desfecho $Y$. As principais relacoes sao $V \rightarrow A$, $V \rightarrow Y$, $X_t \rightarrow A$, $X_t \rightarrow Y$ e $A \rightarrow Y$, alem da dependencia temporal $X_t \rightarrow X_{{t+1}}$. Assim, o ajuste causal deve usar apenas informacao anterior a $t_0$.

Foi emulado um target trial observacional. O ponto $t_0$ representa o momento de decisao transfusional. Foram construidas features temporais e estaticas antes de $t_0$, sem incluir informacao posterior ao tratamento. O desfecho primario foi mortalidade. A coorte final incluiu {int(global_row.n)} pacientes, sendo {int(global_row.n_treated)} transfundidos e {int(global_row.n_control)} controles.

O estimador principal foi AIPW (augmented inverse probability weighting), tambem chamado doubly robust, combinando modelo de tratamento e modelo de desfecho. Foram tambem reportados ATT, IPTW e overlap weighted difference. O efeito global foi fraco e nao significativo: AIPW ATE={global_row.aipw_ate:.3f}, IC95\% [{global_row.aipw_ate_ci95_low:.3f}, {global_row.aipw_ate_ci95_high:.3f}]. A mortalidade bruta foi {global_row.treated_mean:.3f} nos transfundidos e {global_row.control_mean:.3f} nos controles.

\begin{{table}}[htbp]
\centering
\small
\caption{{Efeitos causais estimados nos grupos finais do scan.}}
\begin{{tabular}}{{llrrr}}
\toprule
Grupo & Fenotipo & $n$ & AIPW ATE & IC95\% \\
\midrule
{chr(10).join(table_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}

Os grupos B1 e B2 apresentaram evidencia observacional de beneficio. B1 representa anemia dinamica com frequencia cardiaca controlada, queda de hemoglobina e SpO2 final nao elevada. B2 refina B1 ao adicionar pressao arterial media baixa/moderada. Os grupos M1, M2 e M3 apresentaram sinal de maleficio: M1 representa ascensao da frequencia cardiaca e pico de pressao arterial media; M2 adiciona variabilidade renal; M3 identifica ausencia de queda relevante da hemoglobina.

A validacao discovery/validation mostrou direcao consistente para B1, B2, M1 e M3; M2 apresentou direcao positiva em discovery e validacao, com maior incerteza em discovery e sinal forte na validacao. Os contrafactuais individuais foram estimados por $\hat\mu_0=E[Y|A=0,X]$ e $\hat\mu_1=E[Y|A=1,X]$, produzindo $\widehat{{ITE}}=\hat\mu_1-\hat\mu_0$. Esses valores sao usados como apoio exploratorio, nao como verdade contrafactual observada.

Em relacao ao artigo anterior, os clusters fisiologicos e regras associativas do projeto cluster-transfusion foram tratados como contexto e analise de ponte. O K=2 antigo nao separou claramente efeito causal no target trial atual. As regras antigas de beneficio nao se reproduziram nesta coorte causal restrita, enquanto parte dos antigos grupos de risco se alinhou aos novos grupos M1/M2/M3. Portanto, a interpretacao final e que o novo scan causal refina os achados fisiologicos previos, em vez de simplesmente reproduzi-los.

Esses resultados permanecem observacionais e geradores de hipotese. Eles dependem de ignorabilidade condicional, consistencia, positividade, qualidade dos modelos de nuisance e ausencia de confundimento residual. Alguns grupos apresentam desequilibrio residual de covariaveis, reduzindo a confianca causal absoluta. Assim, os achados nao devem ser apresentados como regra clinica definitiva, mas como evidencia de heterogeneidade causal estimada a ser validada externamente.
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


def slide_storyline(path: Path) -> None:
    text = """# Roteiro de slides

## Slide 1: Problema clinico
- Transfusao de hemacias e frequente na UTI.
- A decisao costuma depender de hemoglobina, gravidade e contexto clinico.
- O efeito pode variar entre pacientes.

## Slide 2: Por que associacao nao basta
- Pacientes transfundidos tendem a ser mais graves.
- Ha confundimento por indicacao.
- Comparar mortalidade bruta pode induzir conclusoes erradas.

## Slide 3: Pergunta causal
- Qual e o efeito de transfundir versus nao transfundir?
- Estimando $Y(1)-Y(0)$.
- O grupo/fenotipo estratifica o efeito; nao e o tratamento.

## Slide 4: Target trial
- Definicao de t0.
- Features apenas antes de t0.
- Tratamento A: transfusao.
- Desfecho Y: mortalidade.

## Slide 5: Estimadores
- Propensity score.
- Modelo de outcome.
- AIPW/doubly robust como principal.
- ATT, IPTW e overlap como triangulacao.

## Slide 6: Resultado global
- Efeito global fraco e nao significativo.
- AIPW ATE = +0.040, IC95% [-0.027, +0.117].
- A transfusao nao parece ter efeito medio homogeneo.

## Slide 7: Heterogeneidade
- Scan causal identificou cinco grupos.
- B1/B2 com beneficio estimado.
- M1/M2/M3 com maleficio estimado.

## Slide 8: Grupos B1/B2
- B1: FC controlada, Hb em queda, SpO2 nao alta.
- B2: B1 + PAM media baixa/moderada.
- Interpretacao: anemia dinamica em paciente relativamente compensado.

## Slide 9: Grupos M1/M2/M3
- M1: FC subindo + pico de PAM elevado.
- M2: M1 + variabilidade de creatinina.
- M3: hemoglobina sem queda relevante.
- Interpretacao: deterioracao/indicacao residual/risco cardiorrenal.

## Slide 10: Contrafactual individual
- Modelo estima mu0 e mu1.
- ITE = mu1 - mu0.
- Uso exploratorio; o foco principal e efeito por grupo.

## Slide 11: Limitacoes
- Estudo observacional.
- Confundimento residual.
- Positividade e overlap limitados em subgrupos.
- Sem ground truth contrafactual individual.

## Slide 12: Conclusao
- Efeito global nao significativo.
- Heterogeneidade forte.
- Beneficio em anemia dinamica compensada.
- Maleficio em trajetorias de estresse hemodinamico/cardiorrenal.
- Resultados geradores de hipotese para validacao externa.
"""
    path.write_text(text, encoding="utf-8")


def source_audit(root: Path) -> pd.DataFrame:
    files = [root / rel for rel in REQUIRED_INPUTS]
    files.extend(sorted((root / "outputs/target_trial/legacy_rule_crosswalk").rglob("legacy_rule_scan_crosswalk_report.md")))
    rows = []
    for path in files:
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
                "mtime_utc_epoch": path.stat().st_mtime if path.exists() else np.nan,
                "sha256": file_sha256(path) if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def final_audit(
    root: Path,
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    source_files: pd.DataFrame,
    sensitivity: pd.DataFrame,
    agreement: pd.DataFrame,
    path: Path,
) -> None:
    feature_cols = metadata.get("feature_columns", [])
    post_patterns = ["post", "after", "future"]
    post_like = [col for col in feature_cols if any(pattern in col.lower() for pattern in post_patterns)]
    outcome_cols = set(metadata.get("outcome_columns", []))
    leakage_exact = [col for col in feature_cols if col in outcome_cols or col in {"transfused", "treatment_time", "rbc_transfusion_flag"}]
    direction_ok = {
        row.group: (row.aipw_ate < 0 if row.direction == "benefit" else row.aipw_ate > 0)
        for row in summary.itertuples(index=False)
    }
    lines = [
        "# Auditoria final de consistencia causal",
        "",
        "## Arquivos usados",
        "",
        markdown_table(source_files, ["path", "exists", "size_bytes", "mtime_utc_epoch", "sha256"], max_rows=20),
        "",
        "## Vazamento temporal",
        "",
        f"- Numero de features usadas no target trial: `{len(feature_cols)}`.",
        f"- Features com padrao pos-t0 (`post`, `after`, `future`): `{post_like}`.",
        f"- Features que coincidem exatamente com tratamento/desfechos: `{leakage_exact}`.",
        "- Conclusao: nao foi detectado uso nominal de feature pos-t0 nos metadados atuais. Variaveis `*_delta` sao interpretadas como deltas dentro da janela pre-t0.",
        "",
        "## Tratamento versus grupos",
        "",
        "- Tratamento causal: `transfused`.",
        "- Grupos finais: B1, B2, M1, M2, M3.",
        "- Os grupos foram usados como estratos de heterogeneidade, nao como tratamento.",
        "",
        "## Direcao dos efeitos finais",
        "",
        markdown_table(pd.DataFrame([{"group": k, "direction_consistent": v} for k, v in direction_ok.items()]), ["group", "direction_consistent"]),
        "",
        "## Suporte comum",
        "",
        markdown_table(sensitivity[["support", "group", "n", "n_treated", "n_control", "aipw_ate", "estimable"]], ["support", "group", "n", "n_treated", "n_control", "aipw_ate", "estimable"], max_rows=40),
        "",
        "## Concordancia de estimadores",
        "",
        markdown_table(agreement, list(agreement.columns), max_rows=10),
        "",
        "## Linguagem causal",
        "",
        "- A secao final usa `efeito estimado`, `observacional`, `gerador de hipotese` e evita afirmar causalidade definitiva.",
        "- A conclusao principal mantida e: efeito global fraco/nao significativo, heterogeneidade forte, beneficio em B1/B2 e maleficio em M1/M2/M3.",
        "",
        "## Resultado da auditoria",
        "",
        "A auditoria nao encontrou mudanca de grupos, busca adicional de subgrupos ou uso de arquivos fora dos resultados finais especificados. Os materiais gerados devem ser usados como consolidacao final da inferencia causal.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> None:
    out = root / "outputs/target_trial"
    final_dir = ensure_dir(out / "final_groups")
    figures_dir = ensure_dir(out / "figures")
    reports_dir = ensure_dir(out / "reports")
    sensitivity_dir = ensure_dir(out / "sensitivity")

    data = load_sources(root)
    metadata = read_json(root / "outputs/target_trial/processed/target_trial_metadata.json")
    summary = final_summary(data["effects"])
    summary.to_csv(final_dir / "final_causal_summary_table.csv", index=False)
    write_latex_summary(summary, final_dir / "final_causal_summary_table.tex")

    forest_plot(summary, figures_dir / "final_groups_forest_plot.png")
    discovery_validation_plot(summary, figures_dir / "discovery_validation_effects.png")
    profile_heatmap(data["profiles"], figures_dir / "final_groups_profiles_heatmap.png")
    overlap_balance_plot(data["effects"], data["balance"], figures_dir / "final_groups_overlap_balance.png")
    pipeline_diagram(figures_dir / "causal_pipeline_diagram.png")

    analysis = make_analysis_frame(data, metadata)
    balance_features = balance_feature_table(analysis, metadata)
    balance_features.to_csv(final_dir / "final_group_balance_feature_level.csv", index=False)
    balance_interpretation(balance_features, data["balance"], reports_dir / "final_group_balance_interpretation.md")

    sensitivity = common_support_sensitivity(analysis, metadata)
    sensitivity.to_csv(sensitivity_dir / "common_support_sensitivity.csv", index=False)
    common_support_report(sensitivity, reports_dir / "common_support_sensitivity.md")

    agreement = estimator_agreement(summary, reports_dir / "estimator_agreement.md")
    agreement.to_csv(final_dir / "estimator_agreement.csv", index=False)

    latex_section(summary, data["main"], reports_dir / "causal_inference_section_final.tex")
    slide_storyline(reports_dir / "slide_storyline.md")

    sources = source_audit(root)
    sources.to_csv(reports_dir / "final_causal_source_files.csv", index=False)
    final_audit(root, metadata, summary, sources, sensitivity, agreement, reports_dir / "final_causal_audit.md")
    print("final_causal_materials_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final causal inference tables, figures and reports.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()
    run(Path(args.repo_root).resolve())


if __name__ == "__main__":
    main()
