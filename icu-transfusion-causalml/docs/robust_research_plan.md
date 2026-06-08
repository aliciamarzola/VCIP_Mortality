# Plano de Pesquisa Robusto

Este plano refaz a inferencia causal como uma emulacao de target trial com heterogeneidade honesta.

## Decisoes Principais

- O tratamento continua sendo transfusao de hemacias.
- K=2 e usado como macrofenotipo fisiologico, nao como rotulo de beneficio/maleficio.
- Beneficio/maleficio vem do efeito estimado contra controles comparaveis.
- O scan descobre regras apenas em `discovery`.
- O efeito final das regras e estimado apenas em `validation`.
- Modelos preditivos sao secundarios; a analise principal usa ATT/ATE, overlap, SMD e bootstrap.

## Pipeline Novo

```text
scripts/target_trial/build_target_trial.py
scripts/causal_estimands/doubly_robust.py
scripts/heterogeneity/k2_phenotype_effects.py
scripts/heterogeneity/honest_scan_discovery.py
scripts/heterogeneity/honest_scan_validation.py
scripts/reporting/generate_target_trial_report.py
```

Comando:

```bash
python scripts/run_robust_causal_pipeline.py --config-dir configs
```

## Outputs

```text
outputs/target_trial/processed/trial_dataset.parquet
outputs/target_trial/causal/main_effects.csv
outputs/target_trial/causal/balance_smd.csv
outputs/target_trial/heterogeneity/effect_by_k2.csv
outputs/target_trial/heterogeneity/honest_scan_discovery_selected_rules.csv
outputs/target_trial/heterogeneity/honest_scan_validation.csv
outputs/target_trial/reports/target_trial_report.md
```

## Interpretacao

O resultado confirmatorio deve priorizar o split de `validation`. Regras fortes apenas em `discovery` sao hipoteses, nao conclusoes. A versao final deve ser rodada na maquina com dados reais e features estritamente pre-t0.
