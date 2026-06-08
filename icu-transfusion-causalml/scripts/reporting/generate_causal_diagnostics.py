from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.target_trial_utils import (  # noqa: E402
    bootstrap_ci,
    crossfit_nuisance,
    ensure_dir,
    load_trial_config,
    output_root,
    read_json,
    rule_mask,
    save_table,
    summarize_effect,
    weighted_mean,
    write_json,
)


GROUPS = ["Global", "B1", "B2", "M1", "M2", "M3"]
FINAL_GROUPS = ["B1", "B2", "M1", "M2", "M3"]
LEAK_PATTERNS = ("post", "after", "future", "outcome", "mortality", "death", "survival")
RULES: dict[str, list[dict[str, Any]]] = {
    "B1": [
        {"feature": "heart_rate_max", "op": "<=", "threshold": 103.833},
        {"feature": "heart_rate_delta", "op": "<=", "threshold": 3.905},
        {"feature": "hemoglobin_slope", "op": "<=", "threshold": -0.002},
        {"feature": "spo2_last", "op": "<=", "threshold": 97.5},
    ],
    "B2": [
        {"feature": "heart_rate_max", "op": "<=", "threshold": 103.833},
        {"feature": "heart_rate_delta", "op": "<=", "threshold": 3.905},
        {"feature": "hemoglobin_slope", "op": "<=", "threshold": -0.002},
        {"feature": "mbp_mean", "op": "<=", "threshold": 78.706},
        {"feature": "spo2_last", "op": "<=", "threshold": 97.5},
    ],
    "M1": [
        {"feature": "heart_rate_delta", "op": ">", "threshold": 3.905},
        {"feature": "mbp_max", "op": ">", "threshold": 93.0},
    ],
    "M2": [
        {"feature": "heart_rate_delta", "op": ">", "threshold": 3.905},
        {"feature": "mbp_max", "op": ">", "threshold": 93.0},
        {"feature": "creatinine_std", "op": ">", "threshold": 0.146},
    ],
    "M3": [
        {"feature": "hemoglobin_slope", "op": ">", "threshold": -0.002},
    ],
}


def smd_for_feature(df: pd.DataFrame, feature: str, treatment_col: str = "transfused", weight_col: str | None = None) -> float:
    tmp = pd.DataFrame({"x": pd.to_numeric(df[feature], errors="coerce"), "a": pd.to_numeric(df[treatment_col], errors="coerce")})
    if weight_col is not None:
        tmp["w"] = pd.to_numeric(df[weight_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "a"])
    if tmp.empty or tmp["a"].nunique() < 2:
        return np.nan
    treated = tmp[tmp["a"] == 1]
    control = tmp[tmp["a"] == 0]
    if treated.empty or control.empty:
        return np.nan
    if weight_col is None:
        m1 = float(treated["x"].mean())
        m0 = float(control["x"].mean())
    else:
        m1 = weighted_mean(treated["x"].to_numpy(), treated["w"].to_numpy())
        m0 = weighted_mean(control["x"].to_numpy(), control["w"].to_numpy())
    pooled = np.sqrt((float(treated["x"].var(ddof=0)) + float(control["x"].var(ddof=0))) / 2.0)
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0
    return float((m1 - m0) / pooled)


def group_mask(assignments: pd.DataFrame, group: str) -> pd.Series:
    if group == "Global":
        return pd.Series(True, index=assignments.index)
    return assignments[group].fillna(False).astype(bool)


def fmt(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def latex_table(df: pd.DataFrame, caption: str, label: str, columns: list[str], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(columns) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in df[columns].itertuples(index=False):
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(pred)
    y = y[mask]
    pred = np.clip(pred[mask], 1e-6, 1 - 1e-6)
    brier = brier_score_loss(y, pred)
    if len(np.unique(y)) < 2:
        return np.nan, brier, np.nan
    return roc_auc_score(y, pred), brier, average_precision_score(y, pred)


def adjusted_risks(df: pd.DataFrame, outcome_col: str = "mortality_anytime") -> tuple[float, float]:
    a = df["transfused"].astype(int).to_numpy()
    y = df[outcome_col].astype(float).to_numpy()
    e = np.clip(df["propensity_score"].astype(float).to_numpy(), 1e-4, 1 - 1e-4)
    mu0 = df["mu0"].astype(float).to_numpy()
    mu1 = df["mu1"].astype(float).to_numpy()
    risk1 = np.nanmean(mu1 + a * (y - mu1) / e)
    risk0 = np.nanmean(mu0 + (1 - a) * (y - mu0) / (1 - e))
    return float(np.clip(risk1, 1e-6, 1 - 1e-6)), float(np.clip(risk0, 1e-6, 1 - 1e-6))


def evalue_from_rr(rr: float) -> float:
    if not np.isfinite(rr) or rr <= 0:
        return np.nan
    rr = rr if rr >= 1 else 1 / rr
    if rr < 1:
        return np.nan
    return float(rr + np.sqrt(rr * (rr - 1)))


def perturb_rule(rule: list[dict[str, Any]], df: pd.DataFrame) -> list[list[dict[str, Any]]]:
    options = []
    for item in rule:
        feature = item["feature"]
        threshold = float(item["threshold"])
        series = pd.to_numeric(df[feature], errors="coerce")
        sd = float(series.std(skipna=True)) if series.notna().any() else 0.0
        step = max(abs(threshold) * 0.05, sd * 0.05, 1e-4)
        opts = []
        for delta in [-step, 0.0, step]:
            new = dict(item)
            new["threshold"] = threshold + delta
            opts.append(new)
        options.append(opts)
    return [list(combo) for combo in itertools.product(*options)]


def generate_ps_plots(df: pd.DataFrame, assignments: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    for group in GROUPS:
        mask = group_mask(assignments, group)
        sub = df.loc[mask].copy()
        plt.figure(figsize=(7.2, 4.6))
        for a, label, color in [(0, "Controles", "#4C78A8"), (1, "Transfundidos", "#F58518")]:
            vals = sub.loc[sub["transfused"].eq(a), "propensity_score"].dropna()
            plt.hist(vals, bins=np.linspace(0, 1, 26), alpha=0.55, density=True, label=label, color=color)
        plt.axvline(0.05, color="#555555", linestyle="--", lw=1)
        plt.axvline(0.95, color="#555555", linestyle="--", lw=1)
        plt.title(f"Overlap do propensity score - {group}")
        plt.xlabel("Propensity score")
        plt.ylabel("Densidade")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / f"propensity_overlap_{group}.png", dpi=220)
        plt.close()


def main(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"
    final = out_root / "final_groups"
    diag = ensure_dir(out_root / "diagnostics")
    ensure_dir(diag / "figures")

    metadata = read_json(proc / "target_trial_metadata.json")
    feature_cols = metadata.get("feature_columns", [])
    outcome_col = metadata.get("primary_outcome", "mortality_anytime")
    trial = pd.read_parquet(proc / "trial_dataset.parquet")
    nuisance = pd.read_parquet(caus / "nuisance_predictions.parquet")
    assignments = pd.read_parquet(final / "final_group_assignments.parquet")
    effects = pd.read_csv(final / "final_group_effects.csv")
    group_balance = pd.read_csv(final / "final_group_balance.csv")

    df = trial.merge(nuisance.drop(columns=["analysis_split", "transfused", outcome_col], errors="ignore"), on="stay_id", how="left")
    df = df.merge(assignments[["stay_id"] + FINAL_GROUPS + ["final_primary_label"]], on="stay_id", how="left")
    for group in FINAL_GROUPS:
        df[group] = df[group].fillna(False).astype(bool)
    assignments = df[["stay_id"] + FINAL_GROUPS].copy()

    numeric_features = [col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    possible_leak = [
        col
        for col in feature_cols
        if any(pattern in col.lower() for pattern in LEAK_PATTERNS) or col in {"treatment_time", "transfused", outcome_col}
    ]
    leakage = pd.DataFrame(
        [
            {
                "covariaveis_avaliadas": len(feature_cols),
                "covariaveis_exclusivamente_pre_t0": len(feature_cols) - len(possible_leak),
                "covariaveis_com_possivel_vazamento_temporal": len(possible_leak),
                "features_suspeitas": "; ".join(possible_leak),
                "resultado_da_checagem": "Sem vazamento identificado" if not possible_leak else f"{len(possible_leak)} variaveis suspeitas revisar/remover",
            }
        ]
    )
    save_table(leakage, diag / "temporal_leakage_check.csv")

    balance_rows = []
    feature_level_rows = []
    for group in GROUPS:
        mask = group_mask(assignments, group)
        sub = df.loc[mask].copy()
        smds = []
        for feature in numeric_features:
            before = smd_for_feature(sub, feature, "transfused")
            after = smd_for_feature(sub, feature, "transfused", "overlap_weight")
            smds.append((feature, before, after))
            feature_level_rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "smd_before": before,
                    "smd_after_overlap": after,
                    "abs_smd_before": abs(before) if np.isfinite(before) else np.nan,
                    "abs_smd_after_overlap": abs(after) if np.isfinite(after) else np.nan,
                }
            )
        tab = pd.DataFrame(smds, columns=["feature", "smd_before", "smd_after_overlap"])
        balance_rows.append(
            {
                "Grupo": group,
                "SMD medio antes": float(tab["smd_before"].abs().mean()),
                "SMD medio depois": float(tab["smd_after_overlap"].abs().mean()),
                "SMD maximo depois": float(tab["smd_after_overlap"].abs().max()),
                "Variaveis com SMD > 0,1": int((tab["smd_after_overlap"].abs() > 0.1).sum()),
                "n_covariaveis_numericas": int(tab["smd_after_overlap"].notna().sum()),
            }
        )
    balance_summary = pd.DataFrame(balance_rows)
    save_table(balance_summary, diag / "smd_balance_summary.csv")
    save_table(pd.DataFrame(feature_level_rows), diag / "smd_balance_feature_level.csv")

    positivity_rows = []
    for group in GROUPS:
        mask = group_mask(assignments, group)
        sub = df.loc[mask].copy()
        ps = pd.to_numeric(sub["propensity_score"], errors="coerce")
        treated_ps = ps[sub["transfused"].eq(1)]
        control_ps = ps[sub["transfused"].eq(0)]
        positivity_rows.append(
            {
                "Grupo": group,
                "n": int(len(sub)),
                "PS medio tratados": float(treated_ps.mean()),
                "PS medio controles": float(control_ps.mean()),
                "% PS extremo": float(((ps <= 0.05) | (ps >= 0.95)).mean() * 100),
                "% PS extremo estrito": float(((ps < 0.05) | (ps > 0.95)).mean() * 100),
                "ps_min": float(ps.min()),
                "ps_max": float(ps.max()),
            }
        )
    positivity = pd.DataFrame(positivity_rows)
    save_table(positivity, diag / "positivity_overlap_summary.csv")
    generate_ps_plots(df, assignments, diag)

    a = df["transfused"].astype(int).to_numpy()
    y = df[outcome_col].astype(int).to_numpy()
    ps = df["propensity_score"].to_numpy(dtype=float)
    obs_pred = np.where(a == 1, df["mu1"].to_numpy(dtype=float), df["mu0"].to_numpy(dtype=float))
    model_rows = []
    auc, brier, ap = binary_metrics(a, ps)
    model_rows.append({"Modelo": "Propensity score P(A=1|X)", "AUC": auc, "Brier Score": brier, "Average Precision": ap, "Observacao": "discriminacao moderada/boa" if auc >= 0.75 else "discriminacao moderada"})
    auc, brier, ap = binary_metrics(y, obs_pred)
    model_rows.append({"Modelo": "Outcome model E[Y|A,X]", "AUC": auc, "Brier Score": brier, "Average Precision": ap, "Observacao": "calibracao/discriminacao moderada"})
    mask_t = a == 1
    auc, brier, ap = binary_metrics(y[mask_t], df.loc[mask_t, "mu1"].to_numpy(dtype=float))
    model_rows.append({"Modelo": "Outcome model em tratados", "AUC": auc, "Brier Score": brier, "Average Precision": ap, "Observacao": "avaliacao no grupo A=1"})
    mask_c = a == 0
    auc, brier, ap = binary_metrics(y[mask_c], df.loc[mask_c, "mu0"].to_numpy(dtype=float))
    model_rows.append({"Modelo": "Outcome model em controles", "AUC": auc, "Brier Score": brier, "Average Precision": ap, "Observacao": "avaliacao no grupo A=0"})
    model_diagnostics = pd.DataFrame(model_rows)
    save_table(model_diagnostics, diag / "aipw_nuisance_model_diagnostics.csv")

    rng = np.random.default_rng(int(config.get("target_trial", {}).get("seed", 42)))
    placebo_trial = trial.copy()
    placebo_trial["transfused"] = rng.permutation(placebo_trial["transfused"].astype(int).to_numpy())
    placebo_nuisance = crossfit_nuisance(placebo_trial, feature_cols, outcome_col, "transfused", config)
    random_trial = trial.copy()
    random_trial["random_common_cause"] = rng.normal(size=len(random_trial))
    random_nuisance = crossfit_nuisance(random_trial, feature_cols + ["random_common_cause"], outcome_col, "transfused", config)

    refuter_rows = []
    original_all = df[["stay_id"] + FINAL_GROUPS + ["transfused", outcome_col, "propensity_score", "mu0", "mu1", "aipw_score_ate", "overlap_weight", "iptw_weight"]].copy()
    placebo_all = placebo_nuisance.merge(assignments, on="stay_id", how="left")
    random_all = random_nuisance.merge(assignments, on="stay_id", how="left")
    for group in FINAL_GROUPS:
        original_effect = float(effects[(effects["group_id"].eq(group)) & (effects["split"].eq("all"))]["aipw_ate"].iloc[0])
        pmask = group_mask(placebo_all, group)
        rmask = group_mask(random_all, group)
        omask = group_mask(original_all, group)
        placebo_eff = summarize_effect(placebo_all.loc[pmask].copy(), outcome_col, "transfused")["aipw_ate"]
        random_eff = summarize_effect(random_all.loc[rmask].copy(), outcome_col, "transfused")["aipw_ate"]
        subset_vals = []
        group_df = original_all.loc[omask].copy()
        for _ in range(200):
            sub = group_df.sample(frac=0.8, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
            if sub["transfused"].nunique() == 2:
                subset_vals.append(summarize_effect(sub, outcome_col, "transfused")["aipw_ate"])
        refuter_rows.append(
            {
                "Grupo": group,
                "AIPW original": original_effect,
                "Placebo treatment": float(placebo_eff),
                "Random common cause": float(random_eff),
                "Subset refuter": float(np.mean(subset_vals)),
                "Subset refuter min": float(np.min(subset_vals)),
                "Subset refuter max": float(np.max(subset_vals)),
                "Subset direcao preservada": bool(np.sign(np.mean(subset_vals)) == np.sign(original_effect)),
            }
        )
    refuters = pd.DataFrame(refuter_rows)
    save_table(refuters, diag / "refutation_tests.csv")

    perturb_rows = []
    perturb_detail_rows = []
    for group, rule in RULES.items():
        original_effect = float(effects[(effects["group_id"].eq(group)) & (effects["split"].eq("all"))]["aipw_ate"].iloc[0])
        perturbed_effects = []
        for rule_variant in perturb_rule(rule, df):
            mask = rule_mask(df, rule_variant)
            sub = df.loc[mask].copy()
            if len(sub) < 30 or sub["transfused"].nunique() < 2:
                continue
            est = summarize_effect(sub, outcome_col, "transfused")
            eff = est["aipw_ate"]
            perturbed_effects.append(eff)
            perturb_detail_rows.append(
                {
                    "Grupo": group,
                    "rule_text": " AND ".join(f"{r['feature']} {r['op']} {float(r['threshold']):.6g}" for r in rule_variant),
                    "n": est["n"],
                    "treated": est["n_treated"],
                    "controls": est["n_control"],
                    "aipw_ate": eff,
                }
            )
        arr = np.asarray(perturbed_effects, dtype=float)
        perturb_rows.append(
            {
                "Grupo": group,
                "Efeito original": original_effect,
                "Faixa min": float(np.nanmin(arr)),
                "Faixa max": float(np.nanmax(arr)),
                "Faixa de efeitos sob perturbacao": f"[{np.nanmin(arr):.3f}; {np.nanmax(arr):.3f}]",
                "Direcao preservada?": "Sim" if np.all(np.sign(arr) == np.sign(original_effect)) else "Nao",
                "n_variantes_validas": int(np.isfinite(arr).sum()),
            }
        )
    threshold = pd.DataFrame(perturb_rows)
    save_table(threshold, diag / "threshold_robustness_summary.csv")
    save_table(pd.DataFrame(perturb_detail_rows), diag / "threshold_robustness_all_variants.csv")

    evalue_rows = []
    merged_effect = effects[effects["split"].eq("all")].set_index("group_id")
    for group in FINAL_GROUPS:
        mask = group_mask(assignments, group)
        sub = df.loc[mask].copy()
        risk1, risk0 = adjusted_risks(sub, outcome_col)
        ate = float(merged_effect.loc[group, "aipw_ate"])
        ci_low = float(merged_effect.loc[group, "aipw_ate_ci95_low"])
        ci_high = float(merged_effect.loc[group, "aipw_ate_ci95_high"])
        rr = risk1 / risk0
        if ate < 0:
            rr_limit = risk0 / np.clip(risk0 + ci_high, 1e-6, 1 - 1e-6)
        else:
            rr_limit = np.clip(risk0 + ci_low, 1e-6, 1 - 1e-6) / risk0
        evalue_rows.append(
            {
                "Grupo": group,
                "AIPW": ate,
                "IC95%": f"[{ci_low:.3f}; {ci_high:.3f}]",
                "risk1_aipw": risk1,
                "risk0_aipw": risk0,
                "RR_aproximado": rr,
                "E-value do efeito": evalue_from_rr(rr),
                "E-value do limite do IC": evalue_from_rr(rr_limit),
                "nota": "E-value aproximado a partir de riscos potenciais AIPW e IC de diferenca de risco.",
            }
        )
    evalues = pd.DataFrame(evalue_rows)
    save_table(evalues, diag / "evalue_sensitivity.csv")

    # Formatted tables for Overleaf.
    leakage_tex = leakage.rename(
        columns={
            "covariaveis_avaliadas": "Covariaveis avaliadas",
            "covariaveis_exclusivamente_pre_t0": "Covariaveis pre-$t_0$",
            "covariaveis_com_possivel_vazamento_temporal": "Possivel vazamento",
            "resultado_da_checagem": "Resultado",
        }
    )
    leakage_tex["Features suspeitas"] = leakage["features_suspeitas"].replace("", "Nenhuma")

    balance_tex = balance_summary.copy()
    for col in ["SMD medio antes", "SMD medio depois", "SMD maximo depois"]:
        balance_tex[col] = balance_tex[col].map(lambda x: fmt(x, 3))
    positivity_tex = positivity.copy()
    for col in ["PS medio tratados", "PS medio controles", "% PS extremo"]:
        positivity_tex[col] = positivity_tex[col].map(lambda x: fmt(x, 3))
    model_tex = model_diagnostics.copy()
    for col in ["AUC", "Brier Score"]:
        model_tex[col] = model_tex[col].map(lambda x: fmt(x, 3))
    ref_tex = refuters.copy()
    for col in ["AIPW original", "Placebo treatment", "Random common cause", "Subset refuter"]:
        ref_tex[col] = ref_tex[col].map(lambda x: fmt(x, 3))
    thresh_tex = threshold.copy()
    thresh_tex["Efeito original"] = thresh_tex["Efeito original"].map(lambda x: fmt(x, 3))
    eval_tex = evalues.copy()
    for col in ["AIPW", "E-value do efeito", "E-value do limite do IC"]:
        eval_tex[col] = eval_tex[col].map(lambda x: fmt(x, 2 if "E-value" in col else 3))

    tables = [
        latex_table(
            leakage_tex,
            "Checagem de vazamento temporal das covariaveis.",
            "tab:temporal_leakage",
            ["Covariaveis avaliadas", "Covariaveis pre-$t_0$", "Possivel vazamento", "Resultado"],
            "rrrr",
        ),
        latex_table(
            balance_tex,
            "Resumo do balanceamento por SMD nos grupos finais.",
            "tab:smd_balance",
            ["Grupo", "SMD medio antes", "SMD medio depois", "SMD maximo depois", "Variaveis com SMD > 0,1"],
            "lrrrr",
        ),
        latex_table(
            positivity_tex,
            "Diagnostico de positividade e suporte comum.",
            "tab:positivity",
            ["Grupo", "n", "PS medio tratados", "PS medio controles", "% PS extremo"],
            "lrrrr",
        ),
        latex_table(
            model_tex,
            "Desempenho dos modelos auxiliares do AIPW.",
            "tab:nuisance_models",
            ["Modelo", "AUC", "Brier Score", "Observacao"],
            "lrrl",
        ),
        latex_table(
            ref_tex,
            "Testes de refutacao dos efeitos causais por grupo.",
            "tab:refuters",
            ["Grupo", "AIPW original", "Placebo treatment", "Random common cause", "Subset refuter"],
            "lrrrr",
        ),
        latex_table(
            thresh_tex,
            "Analise de robustez dos limiares dos grupos finais.",
            "tab:threshold_robustness",
            ["Grupo", "Efeito original", "Faixa de efeitos sob perturbacao", "Direcao preservada?"],
            "lrlr",
        ),
        latex_table(
            eval_tex,
            "Sensibilidade aproximada a confundimento nao medido por E-value.",
            "tab:evalues",
            ["Grupo", "AIPW", "IC95%", "E-value do efeito", "E-value do limite do IC"],
            "lrrrr",
        ),
    ]
    (diag / "diagnostic_tables_for_overleaf.tex").write_text("\n\n".join(tables), encoding="utf-8")

    report_lines = [
        "# Diagnosticos causais finais",
        "",
        "Este diretorio contem as tabelas pedidas para fortalecer a secao de inferencia causal.",
        "",
        "## Arquivos principais",
        "",
        "- `temporal_leakage_check.csv`",
        "- `smd_balance_summary.csv`",
        "- `positivity_overlap_summary.csv`",
        "- `aipw_nuisance_model_diagnostics.csv`",
        "- `refutation_tests.csv`",
        "- `threshold_robustness_summary.csv`",
        "- `evalue_sensitivity.csv`",
        "- `diagnostic_tables_for_overleaf.tex`",
        "",
        "## Nota sobre PS extremo",
        "",
        "Os propensity scores foram previamente clipados em 0.05. Por isso, a coluna `% PS extremo` usa limite inclusivo (`<=0.05` ou `>=0.95`) para capturar observacoes encostadas no limite de positividade.",
    ]
    (diag / "causal_diagnostics_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(
        diag / "causal_diagnostics_metadata.json",
        {
            "n_feature_columns": len(feature_cols),
            "n_numeric_features_for_smd": len(numeric_features),
            "outcome": outcome_col,
            "groups": GROUPS,
            "note": "Diagnostics generated from current target_trial outputs; no new group search was run.",
        },
    )

    print(f"causal_diagnostics_complete outputs={diag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final causal diagnostics tables and figures.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    main(args.config_dir)
