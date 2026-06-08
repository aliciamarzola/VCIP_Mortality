from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_trial_utils import load_trial_config, output_root, read_json  # noqa: E402


def table_md(path: Path, max_rows: int = 12) -> str:
    if not path.exists():
        return "_Nao disponivel._"
    try:
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    except EmptyDataError:
        return "_Tabela vazia._"
    if df.empty:
        return "_Tabela vazia._"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        values = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_report(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"
    het = out_root / "heterogeneity"
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_json(proc / "target_trial_metadata.json")
    causal_meta = read_json(caus / "causal_estimator_metadata.json")
    scan_meta = read_json(het / "honest_scan_metadata.json")

    lines = [
        "# Target Trial Emulation e Inferencia Causal Honesta",
        "",
        "## Objetivo",
        "",
        "Esta versao refaz a analise como uma emulacao de target trial. O tempo zero e definido por elegibilidade clinica, o tratamento e definido em uma janela curta apos o tempo zero, e as covariaveis sao agregadas somente no periodo pre-t0.",
        "",
        "## Desenho",
        "",
        f"- Fonte: `{metadata.get('source')}`",
        f"- Sintetico: `{metadata.get('synthetic')}`",
        f"- N elegivel: `{metadata.get('n_eligible')}`",
        f"- N analitico: `{metadata.get('n_trial')}`",
        f"- Tratados: `{metadata.get('n_treated')}`",
        f"- Controles: `{metadata.get('n_control')}`",
        f"- Desfecho primario: `{metadata.get('primary_outcome')}`",
        "",
        "## Estimadores",
        "",
        f"- Estimador principal: `{causal_meta.get('estimator')}`",
        "- Estimandos reportados: diferenca bruta, IPTW, overlap weights, AIPW ATE e AIPW ATT.",
        "- A interpretacao confirmatoria exige ausencia de vazamento temporal e bom overlap.",
        "",
        "## Efeito Global",
        "",
        table_md(caus / "main_effects.csv"),
        "",
        "## Balanceamento",
        "",
        "As diferencas padronizadas devem ser avaliadas antes e depois dos pesos. Em geral, valores absolutos abaixo de 0,1 sao desejaveis.",
        "",
        table_md(caus / "balance_smd.csv"),
        "",
        "## Macrofenotipo K=2",
        "",
        "O K=2 e tratado como macrofenotipo fisiologico, nao como rotulo automatico de beneficio ou maleficio.",
        "",
        table_md(het / "effect_by_k2.csv"),
        "",
        "## Scan Honesto",
        "",
        "As regras sao descobertas apenas no split de discovery e estimadas no split de validation. Isto reduz circularidade em relacao a escolher e estimar o efeito no mesmo conjunto.",
        "",
        f"- Regras candidatas selecionadas: `{scan_meta.get('selected_count')}`",
        "",
        "### Regras Descobertas",
        "",
        table_md(het / "honest_scan_discovery_selected_rules.csv"),
        "",
        "### Validacao das Regras",
        "",
        table_md(het / "honest_scan_validation.csv"),
        "",
        "## Interpretacao",
        "",
        "A analise robusta deve priorizar os efeitos em validation. Regras descobertas com sinal forte em discovery mas sem reproducao em validation devem ser tratadas como instaveis. Os modelos preditivos nao devem ser usados como prova causal individual.",
        "",
        "## Proximos Passos",
        "",
        "1. Rodar com dados reais estritamente pre-t0.",
        "2. Revisar overlap e SMD por grupo.",
        "3. Validar regras B/M em split temporal ou coorte externa.",
        "4. Adicionar causal forest/DR-learner honesto como analise secundaria.",
    ]
    path = report_dir / "target_trial_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"target_trial_report_complete {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate target-trial causal report.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    generate_report(args.config_dir)


if __name__ == "__main__":
    main()
