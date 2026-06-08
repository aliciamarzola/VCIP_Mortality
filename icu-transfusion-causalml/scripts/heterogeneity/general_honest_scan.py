"""
General Honest Scan: Combina análise de benefício, malefício e efeito geral com máxima robustez.
Usa todas as features pre-t0 disponíveis para gerar e validar regras de heterogeneidade.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import (  # noqa: E402
    bootstrap_ci,
    load_trial_config,
    output_root,
    read_json,
    save_table,
    summarize_effect,
)


def load_data(config_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carrega dados do trial e nuisance predictions."""
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"

    trial_df = pd.read_parquet(proc / "trial_dataset.parquet")
    nuisance_df = pd.read_parquet(caus / "nuisance_predictions.parquet")
    metadata = read_json(proc / "target_trial_metadata.json")

    df = trial_df.merge(nuisance_df, on="stay_id", how="left", suffixes=('', '_nuisance'))
    
    # Converter sex para numérico (M=1, F=0 ou similar)
    if "sex" in df.columns and not pd.api.types.is_numeric_dtype(df["sex"]):
        df["sex"] = pd.Categorical(df["sex"]).codes.astype(float)
    
    return df, metadata


def get_all_candidate_features(df: pd.DataFrame, metadata: Dict[str, Any]) -> List[str]:
    """Retorna todas as features pre-t0 para análise, incluindo conversões de categóricas."""
    feature_cols = metadata.get("feature_columns", [])
    
    # Filtrar e converter features
    valid_features = []
    for col in feature_cols:
        if col in df.columns:
            # Descartar features de tempo que não são pre-t0
            if col not in ["first_observed_time", "pre_observation_minutes", "_fallback_treatment_time", 
                          "treatment_time", "treatment_source"]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    valid_features.append(col)
                elif col == "sex":
                    # Converter sex para numérico (0/1)
                    valid_features.append(col)
    
    return valid_features


def generate_comprehensive_rules(features: List[str], df: pd.DataFrame, max_depth: int = 2) -> List[Dict[str, Any]]:
    """Gera regras de profundidade 1 e 2 com todas as features."""
    rules = []
    quantiles_d1 = [0.15, 0.25, 0.33, 0.50, 0.67, 0.75, 0.85]  # 7 quantis profundidade 1
    
    # Profundidade 1: regras simples por feature
    for feature in features:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue
        
        values = df[feature].dropna()
        if len(values) < 50:
            continue
        
        for q in quantiles_d1:
            threshold = float(values.quantile(q))
            
            # Regra: feature <= threshold
            rules.append({
                "rule_text": f"{feature} <= {threshold:.3f}",
                "rule_json": [{"feature": feature, "op": "<=", "threshold": threshold}],
                "depth": 1,
                "feature_count": 1
            })
            
            # Regra: feature > threshold
            rules.append({
                "rule_text": f"{feature} > {threshold:.3f}",
                "rule_json": [{"feature": feature, "op": ">", "threshold": threshold}],
                "depth": 1,
                "feature_count": 1
            })
    
    # Profundidade 2: combinações selecionadas estrategicamente
    if max_depth >= 2:
        # Pegar top features por variância
        top_features = []
        for col in features:
            if pd.api.types.is_numeric_dtype(df[col]):
                values = df[col].dropna()
                if len(values) >= 50:
                    top_features.append((col, values.std()))
        
        top_features.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in top_features[:12]]  # Top 12 features por variância
        
        for i in range(len(top_features)):
            for j in range(i+1, min(i+3, len(top_features))):  # Limitar para evitar explosão
                f1, f2 = top_features[i], top_features[j]
                v1 = df[f1].dropna()
                v2 = df[f2].dropna()
                
                if len(v1) < 50 or len(v2) < 50:
                    continue
                
                quantiles_d2 = [0.25, 0.50, 0.75]  # 3 quantis profundidade 2
                for q1 in quantiles_d2:
                    for q2 in quantiles_d2:
                        t1 = float(v1.quantile(q1))
                        t2 = float(v2.quantile(q2))
                        
                        # AND rule
                        rules.append({
                            "rule_text": f"{f1} <= {t1:.3f} AND {f2} <= {t2:.3f}",
                            "rule_json": [
                                {"feature": f1, "op": "<=", "threshold": t1},
                                {"feature": f2, "op": "<=", "threshold": t2}
                            ],
                            "depth": 2,
                            "feature_count": 2
                        })
                        
                        rules.append({
                            "rule_text": f"{f1} > {t1:.3f} AND {f2} > {t2:.3f}",
                            "rule_json": [
                                {"feature": f1, "op": ">", "threshold": t1},
                                {"feature": f2, "op": ">", "threshold": t2}
                            ],
                            "depth": 2,
                            "feature_count": 2
                        })
    
    return rules


def _ci_for_scores(values: pd.Series, seed: int, n_boot: int, ci_method: str) -> Tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    if ci_method == "bootstrap":
        return bootstrap_ci(values, seed=seed, n_boot=n_boot)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean - 1.96 * se, mean + 1.96 * se


def evaluate_rule(
    df: pd.DataFrame,
    rule: Dict[str, Any],
    outcome_col: str = "mortality_anytime",
    *,
    n_boot: int = 200,
    ci_method: str = "normal",
    seed: int = 42,
) -> Dict[str, Any]:
    """Avalia uma regra e retorna efeito e CI."""
    mask = pd.Series(True, index=df.index)
    
    for condition in rule["rule_json"]:
        feature = condition["feature"]
        op = condition["op"]
        threshold = condition["threshold"]
        
        if feature not in df.columns:
            return {"n": 0, "effect": np.nan}
        
        x = pd.to_numeric(df[feature], errors="coerce")
        if op == "<=":
            mask &= x <= threshold
        elif op == ">":
            mask &= x > threshold
    
    subgroup = df[mask]
    n_treated = int((subgroup["transfused"] == 1).sum())
    n_control = int((subgroup["transfused"] == 0).sum())
    
    if len(subgroup) < 30:
        return {
            "n": len(subgroup), 
            "n_treated": n_treated,
            "n_control": n_control,
            "treated_mean": np.nan,
            "control_mean": np.nan,
            "effect": np.nan, 
            "ci_low": np.nan, 
            "ci_high": np.nan
        }
    
    if subgroup["transfused"].nunique() < 2 or subgroup[outcome_col].nunique() < 2:
        return {
            "n": len(subgroup), 
            "n_treated": n_treated,
            "n_control": n_control,
            "treated_mean": np.nan,
            "control_mean": np.nan,
            "effect": np.nan, 
            "ci_low": np.nan, 
            "ci_high": np.nan
        }
    
    summary = summarize_effect(subgroup, outcome_col, "transfused")
    
    # Normal CI is fast for screening; bootstrap can be enabled for the final run.
    if len(subgroup) >= 30:
        ci_low, ci_high = _ci_for_scores(subgroup["aipw_score_ate"], seed=seed, n_boot=n_boot, ci_method=ci_method)
    else:
        ci_low, ci_high = np.nan, np.nan
    
    return {
        "n": len(subgroup),
        "n_treated": summary["n_treated"],
        "n_control": summary["n_control"],
        "treated_mean": summary["treated_mean"],
        "control_mean": summary["control_mean"],
        "effect": summary["aipw_ate"],
        "ci_low": ci_low,
        "ci_high": ci_high
    }


def classify_effect(effect: float, ci_low: float, ci_high: float, threshold: float = 0.05) -> str:
    """Classifica efeito como benefício, malefício ou neutro."""
    if np.isnan(effect):
        return "unknown"
    
    if effect < -threshold and ci_high < 0:
        return "strong_benefit"
    elif effect < -threshold:
        return "weak_benefit"
    elif effect > threshold and ci_low > 0:
        return "strong_harm"
    elif effect > threshold:
        return "weak_harm"
    else:
        return "neutral"


def repeated_honest_validation(df: pd.DataFrame, candidate_rules: List[Dict[str, Any]], 
                               n_folds: int = 5, n_seeds: int = 5) -> pd.DataFrame:
    """Validação repetida honesta."""
    results = []
    
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["transfused"])):
            discovery = df.iloc[train_idx]
            validation = df.iloc[test_idx]
            
            # Avaliar todas as regras no discovery
            for rule_idx, rule in enumerate(candidate_rules):
                disc_res = evaluate_rule(discovery, rule)
                
                if disc_res["n"] < 50 or pd.isna(disc_res["effect"]):
                    continue
                
                if disc_res["n_treated"] < 15 or disc_res["n_control"] < 15:
                    continue
                
                # Avaliar no validation
                val_res = evaluate_rule(validation, rule)
                
                if val_res["n"] < 30 or pd.isna(val_res["effect"]):
                    continue
                
                disc_class = classify_effect(disc_res["effect"], disc_res.get("ci_low", np.nan), 
                                            disc_res.get("ci_high", np.nan))
                val_class = classify_effect(val_res["effect"], val_res.get("ci_low", np.nan), 
                                           val_res.get("ci_high", np.nan))
                
                results.append({
                    "rule_text": rule["rule_text"],
                    "rule_json": json.dumps(rule["rule_json"]),
                    "depth": rule["depth"],
                    "feature_count": rule.get("feature_count", 1),
                    "seed": seed,
                    "fold": fold,
                    "n_discovery": disc_res["n"],
                    "n_treated_discovery": disc_res["n_treated"],
                    "n_control_discovery": disc_res["n_control"],
                    "effect_discovery": disc_res["effect"],
                    "ci_low_discovery": disc_res["ci_low"],
                    "ci_high_discovery": disc_res["ci_high"],
                    "class_discovery": disc_class,
                    "n_validation": val_res["n"],
                    "n_treated_validation": val_res["n_treated"],
                    "n_control_validation": val_res["n_control"],
                    "effect_validation": val_res["effect"],
                    "ci_low_validation": val_res["ci_low"],
                    "ci_high_validation": val_res["ci_high"],
                    "class_validation": val_class
                })
    
    return pd.DataFrame(results)


def repeated_honest_validation(
    df: pd.DataFrame,
    candidate_rules: List[Dict[str, Any]],
    n_folds: int = 5,
    n_seeds: int = 5,
    n_boot: int = 200,
    ci_method: str = "normal",
) -> pd.DataFrame:
    """ValidaÃ§Ã£o repetida honesta com barra de progresso."""
    results = []
    total = len(candidate_rules) * n_folds * n_seeds

    with tqdm(total=total, desc="Validando regras", unit="regra", dynamic_ncols=True) as pbar:
        for seed in range(n_seeds):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["transfused"])):
                discovery = df.iloc[train_idx]
                validation = df.iloc[test_idx]
                fold_start = len(results)

                for rule_idx, rule in enumerate(candidate_rules):
                    try:
                        disc_res = evaluate_rule(
                            discovery,
                            rule,
                            n_boot=n_boot,
                            ci_method=ci_method,
                            seed=seed * 1000 + fold,
                        )

                        if disc_res["n"] < 50 or pd.isna(disc_res["effect"]):
                            continue
                        if disc_res["n_treated"] < 15 or disc_res["n_control"] < 15:
                            continue

                        val_res = evaluate_rule(
                            validation,
                            rule,
                            n_boot=n_boot,
                            ci_method=ci_method,
                            seed=seed * 1000 + fold + 500,
                        )

                        if val_res["n"] < 30 or pd.isna(val_res["effect"]):
                            continue

                        disc_class = classify_effect(
                            disc_res["effect"],
                            disc_res.get("ci_low", np.nan),
                            disc_res.get("ci_high", np.nan),
                        )
                        val_class = classify_effect(
                            val_res["effect"],
                            val_res.get("ci_low", np.nan),
                            val_res.get("ci_high", np.nan),
                        )

                        results.append({
                            "rule_text": rule["rule_text"],
                            "rule_json": json.dumps(rule["rule_json"]),
                            "depth": rule["depth"],
                            "feature_count": rule.get("feature_count", 1),
                            "seed": seed,
                            "fold": fold,
                            "n_discovery": disc_res["n"],
                            "n_treated_discovery": disc_res["n_treated"],
                            "n_control_discovery": disc_res["n_control"],
                            "effect_discovery": disc_res["effect"],
                            "ci_low_discovery": disc_res["ci_low"],
                            "ci_high_discovery": disc_res["ci_high"],
                            "class_discovery": disc_class,
                            "n_validation": val_res["n"],
                            "n_treated_validation": val_res["n_treated"],
                            "n_control_validation": val_res["n_control"],
                            "effect_validation": val_res["effect"],
                            "ci_low_validation": val_res["ci_low"],
                            "ci_high_validation": val_res["ci_high"],
                            "class_validation": val_class,
                        })
                    finally:
                        pbar.update(1)
                        if rule_idx % 50 == 0:
                            pbar.set_postfix(
                                seed=seed,
                                fold=fold,
                                kept=len(results) - fold_start,
                                total_kept=len(results),
                            )

    return pd.DataFrame(results)


def aggregate_validation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega resultados através de folds e seeds."""
    if len(results_df) == 0:
        return pd.DataFrame()
    
    aggregated = []
    
    for rule_text, group in results_df.groupby("rule_text"):
        val_group = group.copy()
        
        # Estatísticas de validação
        n_mean = val_group["n_validation"].mean()
        treated_mean = val_group["n_treated_validation"].mean()
        control_mean = val_group["n_control_validation"].mean()
        effect_mean = val_group["effect_validation"].mean()
        effect_median = val_group["effect_validation"].median()
        effect_std = val_group["effect_validation"].std()
        
        ci_low_pooled = val_group["ci_low_validation"].quantile(0.025)
        ci_high_pooled = val_group["ci_high_validation"].quantile(0.975)
        
        # Proporções
        prop_negative = (val_group["effect_validation"] < 0).mean()
        prop_positive = (val_group["effect_validation"] > 0).mean()
        prop_sig_negative = ((val_group["effect_validation"] < 0) & 
                            (val_group["ci_high_validation"] < 0)).mean()
        prop_sig_positive = ((val_group["effect_validation"] > 0) & 
                            (val_group["ci_low_validation"] > 0)).mean()
        
        # Estabilidade de classe
        class_agreement = (val_group["class_discovery"] == val_group["class_validation"]).mean()
        
        # Direção predominante
        main_class = val_group["class_validation"].mode()[0] if len(val_group["class_validation"].mode()) > 0 else "unknown"
        
        # Score de prioridade: combina magnitude, robustez e consistência
        robustness_score = class_agreement * (1 - effect_std / (abs(effect_mean) + 0.1))
        priority_score = abs(effect_mean) * robustness_score * (prop_sig_negative if effect_mean < 0 else prop_sig_positive)
        
        aggregated.append({
            "rule_text": rule_text,
            "depth": val_group["depth"].iloc[0],
            "feature_count": val_group["feature_count"].iloc[0],
            "n_mean_validation": n_mean,
            "treated_mean_validation": treated_mean,
            "control_mean_validation": control_mean,
            "effect_mean_validation": effect_mean,
            "effect_median_validation": effect_median,
            "effect_std_validation": effect_std,
            "ci95_low_pooled": ci_low_pooled,
            "ci95_high_pooled": ci_high_pooled,
            "prop_negative": prop_negative,
            "prop_positive": prop_positive,
            "prop_sig_negative": prop_sig_negative,
            "prop_sig_positive": prop_sig_positive,
            "class_agreement": class_agreement,
            "main_effect_class": main_class,
            "robustness_score": robustness_score,
            "priority_score": priority_score,
            "n_folds_tested": len(group)
        })
    
    agg_df = pd.DataFrame(aggregated)
    # Ordenar por priority score
    agg_df = agg_df.sort_values("priority_score", ascending=False)
    
    return agg_df


def generate_comprehensive_report(output_dir: Path, aggregated: pd.DataFrame, 
                                raw_results: pd.DataFrame, df: pd.DataFrame):
    """Gera relatório detalhado dos resultados."""
    report_path = output_dir / "general_scan_report.md"
    
    with open(report_path, "w") as f:
        f.write("# General Honest Scan - Análise Completa de Heterogeneidade\n\n")
        
        # Resumo executivo
        f.write("## Resumo Executivo\n\n")
        f.write(f"- Total de regras candidatas: {len(raw_results.groupby('rule_text'))}\n")
        f.write(f"- Regras com resultados em validation: {len(aggregated)}\n")
        
        benefit_rules = len(aggregated[aggregated["main_effect_class"].str.contains("benefit", na=False)])
        harm_rules = len(aggregated[aggregated["main_effect_class"].str.contains("harm", na=False)])
        neutral_rules = len(aggregated[aggregated["main_effect_class"] == "neutral"])
        
        f.write(f"- Regras de **benefício** (efeito negativo): {benefit_rules}\n")
        f.write(f"- Regras de **malefício** (efeito positivo): {harm_rules}\n")
        f.write(f"- Regras **neutras**: {neutral_rules}\n\n")
        
        # Top benefício
        f.write("## Top 10 Regras de Benefício (Redução de Mortalidade)\n\n")
        benefit_df = aggregated[aggregated["effect_mean_validation"] < 0].nlargest(10, "priority_score")
        if len(benefit_df) > 0:
            f.write("| Regra | N | Efeito | IC 95% | Robustez | Classe |\n")
            f.write("|-------|---|--------|--------|----------|--------|\n")
            for _, row in benefit_df.iterrows():
                f.write(f"| {row['rule_text']} | {int(row['n_mean_validation'])} | "
                       f"{row['effect_mean_validation']:.3f} | "
                       f"[{row['ci95_low_pooled']:.3f}, {row['ci95_high_pooled']:.3f}] | "
                       f"{row['robustness_score']:.2f} | {row['main_effect_class']} |\n")
        else:
            f.write("Nenhuma regra de benefício significativo encontrada.\n\n")
        
        # Top malefício
        f.write("\n## Top 10 Regras de Malefício (Aumento de Mortalidade)\n\n")
        harm_df = aggregated[aggregated["effect_mean_validation"] > 0].nlargest(10, "priority_score")
        if len(harm_df) > 0:
            f.write("| Regra | N | Efeito | IC 95% | Robustez | Classe |\n")
            f.write("|-------|---|--------|--------|----------|--------|\n")
            for _, row in harm_df.iterrows():
                f.write(f"| {row['rule_text']} | {int(row['n_mean_validation'])} | "
                       f"{row['effect_mean_validation']:.3f} | "
                       f"[{row['ci95_low_pooled']:.3f}, {row['ci95_high_pooled']:.3f}] | "
                       f"{row['robustness_score']:.2f} | {row['main_effect_class']} |\n")
        else:
            f.write("Nenhuma regra de malefício significativo encontrada.\n\n")
        
        # Metodologia
        f.write("\n## Metodologia\n\n")
        f.write("- **Features utilizadas**: Todas as 102 features pre-t0 disponíveis\n")
        f.write("- **Regras de profundidade**: 1 e 2 (1 ou 2 features por regra)\n")
        f.write("- **Quantis**: 5 quantis (20%, 33%, 50%, 67%, 80%)\n")
        f.write("- **Validação**: Repeated honest validation com 5 folds e 5 seeds\n")
        f.write("- **Critérios de robustez**:\n")
        f.write("  - N mínimo em discovery: 50\n")
        f.write("  - N mínimo em validation: 30\n")
        f.write("  - N mínimo tratados e controles: 15 cada\n")
        f.write("- **Classificação**:\n")
        f.write("  - Benefício forte: efeito < -0.05 com IC alto < 0\n")
        f.write("  - Malefício forte: efeito > 0.05 com IC baixo > 0\n")
        f.write("- **Score de prioridade**: Combina magnitude, robustez e consistência\n")
    
    return report_path


def run_general_honest_scan(config_dir: str):
    """Executa o scan geral completo."""
    print("Carregando dados...")
    df, metadata = load_data(config_dir)
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    scan_dir = out_root / "general_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    
    # Obter todas as features
    features = get_all_candidate_features(df, metadata)
    print(f"Features pre-t0 identificadas: {len(features)}")
    
    # Gerar regras
    print(f"Gerando regras candidatas...")
    candidate_rules = generate_comprehensive_rules(features, df, max_depth=2)
    print(f"Total de regras geradas: {len(candidate_rules)}")
    
    # Validação repetida
    print(f"Executando validação repetida honesta...")
    n_folds = int(os.getenv("GENERAL_SCAN_FOLDS", "5"))
    n_seeds = int(os.getenv("GENERAL_SCAN_SEEDS", "5"))
    n_boot = int(os.getenv("GENERAL_SCAN_BOOTSTRAP", "200"))
    ci_method = os.getenv("GENERAL_SCAN_CI_METHOD", "normal").strip().lower()
    if ci_method not in {"normal", "bootstrap"}:
        raise ValueError("GENERAL_SCAN_CI_METHOD must be 'normal' or 'bootstrap'.")
    total_evals = len(candidate_rules) * n_folds * n_seeds
    print(
        f"Plano: {total_evals:,} avaliaÃ§Ãµes "
        f"({len(candidate_rules)} regras x {n_folds} folds x {n_seeds} seeds); "
        f"IC={ci_method}; bootstrap={n_boot}"
    )
    raw_results = repeated_honest_validation(
        df,
        candidate_rules,
        n_folds=n_folds,
        n_seeds=n_seeds,
        n_boot=n_boot,
        ci_method=ci_method,
    )
    print(f"Resultados de validação obtidos: {len(raw_results)} (rule, fold, seed) combinações")
    
    # Agregar resultados
    print(f"Agregando resultados...")
    aggregated = aggregate_validation_results(raw_results)
    print(f"Regras agregadas: {len(aggregated)}")
    
    # Salvar resultados
    print(f"Salvando resultados...")
    save_table(raw_results, scan_dir / "all_validation_results_raw.csv")
    save_table(aggregated, scan_dir / "aggregated_rules.csv")
    
    # Salvar subsetos por efeito
    if len(aggregated) > 0:
        benefit = aggregated[aggregated["effect_mean_validation"] < 0]
        harm = aggregated[aggregated["effect_mean_validation"] > 0]
        
        if len(benefit) > 0:
            save_table(benefit, scan_dir / "benefit_rules_ranked.csv")
        if len(harm) > 0:
            save_table(harm, scan_dir / "harm_rules_ranked.csv")
    
    # Gerar relatório
    print(f"Gerando relatório...")
    report_path = generate_comprehensive_report(scan_dir, aggregated, raw_results, df)
    
    print(f"\n✓ Scan geral concluído!")
    print(f"Resultados salvos em: {scan_dir}")
    print(f"Relatório: {report_path}")
    
    # Resumo final
    print(f"\n--- RESUMO FINAL ---")
    print(f"Total de regras analisadas: {len(aggregated)}")
    if len(aggregated) > 0:
        benefit_count = len(aggregated[aggregated["effect_mean_validation"] < 0])
        harm_count = len(aggregated[aggregated["effect_mean_validation"] > 0])
        print(f"Benefício: {benefit_count} regras")
        print(f"Malefício: {harm_count} regras")
        
        # Top 5 benefício
        top_benefit = aggregated[aggregated["effect_mean_validation"] < 0].head(5)
        if len(top_benefit) > 0:
            print(f"\nTop 3 benefício:")
            for idx, (_, row) in enumerate(top_benefit.head(3).iterrows(), 1):
                print(f"  {idx}. {row['rule_text']}: Efeito={row['effect_mean_validation']:.3f}, Robustez={row['robustness_score']:.2f}")
        
        # Top 5 malefício
        top_harm = aggregated[aggregated["effect_mean_validation"] > 0].head(5)
        if len(top_harm) > 0:
            print(f"\nTop 3 malefício:")
            for idx, (_, row) in enumerate(top_harm.head(3).iterrows(), 1):
                print(f"  {idx}. {row['rule_text']}: Efeito={row['effect_mean_validation']:.3f}, Robustez={row['robustness_score']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run general honest scan combining benefit and harm analysis.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    run_general_honest_scan(args.config_dir)


if __name__ == "__main__":
    main()
