from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.pipeline_common import causal_dir, counterfactual_dir, evaluation_dir, load_config, mark_demonstrative_if_synthetic, processed_dir, read_json, resolve_path, safe_read_table  # noqa: E402


def _table_preview(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "_Nao disponivel._"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(rows)


def generate_report(config) -> None:
    proc = processed_dir(config)
    c_dir = causal_dir(config)
    eval_dir = evaluation_dir(config)
    cf_dir = counterfactual_dir(config)
    report_path = resolve_path(config, "report_file", "outputs/causal_inference/reports/research_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = read_json(proc / "data_source_metadata.json")
    feature_meta = read_json(proc / "feature_metadata.json")
    cohort = pd.read_parquet(proc / "cohort.parquet") if (proc / "cohort.parquet").exists() else pd.DataFrame()
    causal = safe_read_table(eval_dir / "causal_metrics.csv")
    predictive = safe_read_table(eval_dir / "predictive_metrics.csv")
    sensitivity = safe_read_table(eval_dir / "sensitivity_results.csv")
    effects = safe_read_table(c_dir / "effect_by_phenotype.csv")
    cf_metrics = safe_read_table(cf_dir / "model_metrics.csv")
    synthetic = mark_demonstrative_if_synthetic(config)
    imported_from_legacy = bool(feature_meta.get("imported_from_legacy", False))
    no_post_t0 = bool(feature_meta.get("no_post_t0_features", True))
    window_hours = feature_meta.get("temporal_tensor", {}).get("window_hours", metadata.get("window_hours", "NA"))

    warning = (
        "> **Aviso metodologico:** esta execucao usou dados sinteticos demonstrativos. "
        "As estimativas abaixo validam o funcionamento computacional do pipeline e nao devem ser interpretadas como resultados clinicos reais.\n"
        if synthetic
        else "> **Aviso metodologico:** as estimativas sao observacionais e dependem das hipoteses de ignorabilidade condicional, positividade e consistencia.\n"
    )

    n = len(cohort)
    n_t = int(cohort["transfused"].sum()) if "transfused" in cohort.columns and not cohort.empty else 0
    lines = [
        "# Inferencia Causal Temporal da Transfusao de Hemacias em UTI",
        "",
        warning,
        "",
        "## 1. Introducao",
        "",
        "Este relatorio descreve um pipeline reprodutivel para estimar efeitos causais medios, heterogeneos e contrafactuais da transfusao de concentrado de hemacias em pacientes criticos de UTI.",
        "",
        "## 2. Motivacao Clinica",
        "",
        "A transfusao de hemacias e uma intervencao frequente em UTI, mas seu efeito esperado pode variar conforme a trajetoria fisiologica pre-intervencao, a gravidade clinica e a reserva organica do paciente.",
        "",
        "## 3. Problema de Pesquisa",
        "",
        "O objetivo e estimar o efeito causal da transfusao sobre mortalidade e suporte organico, considerando covariaveis estaticas, series temporais pre-t0 e fenotipos fisiologicos latentes.",
        "",
        "## 4. Hipoteses",
        "",
        "A hipotese principal e que a transfusao nao possui efeito homogeneo em pacientes criticos. Fenotipos fisiologicos definidos por trajetorias temporais devem revelar subgrupos com beneficio, neutralidade ou risco associado.",
        "",
        "## 5. Objetivos",
        "",
        "- Estimar ATE e ATT para mortalidade.",
        "- Estimar CATE por fenotipo temporal.",
        "- Estimar ITE e riscos contrafactuais Y(1) e Y(0).",
        "- Identificar variaveis pre-t0 que distinguem beneficio e risco.",
        "",
        "## 6. Base de Dados e Coorte",
        "",
        f"Fonte dos dados: `{metadata.get('source', 'unknown')}`.",
        f"Numero de permanencias na coorte processada: {n}.",
        f"Numero de tratados/transfundidos: {n_t}.",
        f"Janela pre-intervencao: {window_hours} horas.",
        "",
        "## 7. Tratamento e Desfechos",
        "",
        "Tratamento A=1 corresponde a primeira transfusao de hemacias. Para controles, foi definido pseudo-t0 pela distribuicao temporal dos tratados. O desfecho primario configurado e mortalidade.",
        "",
        "## 8. DAG Causal Preliminar",
        "",
        "```text",
        "V -> X_t",
        "V -> A",
        "V -> Y",
        "U -> X_t",
        "U -> A",
        "U -> Y",
        "X_t -> A",
        "X_t -> Y",
        "A -> Y",
        "X_t -> X_t+1",
        "```",
        "",
        "## 9. Metodologia",
        "",
        (
            "Esta execucao importou artefatos do pipeline anterior de fenotipagem/transfusao e executou as etapas novas de efeitos individuais, contrafactuais e avaliacao. "
            if imported_from_legacy
            else "O pipeline constroi features, gera representacoes temporais, estima fenotipos por K-means, ajusta propensity score, aplica matching/pesos e estima efeitos heterogeneos por meta-learners e modelo contrafactual de duas cabecas separadas. "
        )
        + (
            "As features usadas nos modelos derivados foram mantidas sem colunas pos-t0/delta."
            if no_post_t0
            else "As features foram importadas com colunas pos-t0/delta do estudo anterior; por isso esta execucao deve ser interpretada como reaproveitamento exato do artigo, nao como estimacao estritamente pre-tratamento."
        ),
        "",
        "## 10. Resultados Causais",
        "",
        _table_preview(causal),
        "",
        "## 11. Efeitos por Fenotipo",
        "",
        _table_preview(effects),
        "",
        "## 12. Desempenho Preditivo",
        "",
        _table_preview(predictive if not predictive.empty else cf_metrics),
        "",
        "## 13. Analise de Sensibilidade",
        "",
        _table_preview(sensitivity),
        "",
        "## 14. Limitacoes",
        "",
        "- PEHE real nao e observavel sem contrafactuais verdadeiros.",
        "- Estimativas observacionais dependem de ausencia de confundimento nao medido.",
        "- Positividade e suporte comum devem ser avaliados antes de interpretacao clinica.",
        "- A execucao sintetica testa software, nao valida conclusoes clinicas.",
        "- Quando colunas pos-t0/delta forem mantidas do pipeline anterior, os resultados de modelos individuais nao devem ser apresentados como inferencia causal pre-tratamento estrita.",
        "",
        "## 15. Proximos Passos",
        "",
        "- Corrigir ou ressincronizar os Parquets reais truncados.",
        "- Validar definicoes finais de mortalidade em UTI versus hospitalar.",
        "- Revisar lista clinica de covariaveis pre-tratamento.",
        "- Adicionar analise semi-sintetica com efeito verdadeiro conhecido.",
        "- Comparar janelas de 24h e 48h nos dados reais.",
        "",
        "## 16. Referencias Principais",
        "",
        "- Hernan MA, Robins JM. Causal Inference: What If.",
        "- Athey S, Imbens G. Recursive partitioning for heterogeneous causal effects.",
        "- Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random forests.",
        "- Rubin DB. Estimating causal effects of treatments in randomized and nonrandomized studies.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scientific report.")
    parser.add_argument("--config-dir", default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    generate_report(config)
    print("report_generation_complete")


if __name__ == "__main__":
    main()
