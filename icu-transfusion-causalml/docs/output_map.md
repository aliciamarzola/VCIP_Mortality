# Mapa dos Outputs

## Outputs do Projeto Atual

Raiz:

```text
outputs/causal_inference/
```

### `processed/`

Dados padronizados para inferencia causal.

- `cohort.parquet`: coorte com tratamento, `t0` e desfechos.
- `treatment.parquet`: tratamento `A`.
- `outcomes.parquet`: desfechos.
- `static_features.parquet`: covariaveis/agregacoes.
- `analysis_features.parquet`: tabela combinada para modelos.
- `data_source_metadata.json`: origem dos dados.
- `feature_metadata.json`: metadados das features.

### `phenotypes/`

Fenotipos e grupos derivados de trajetorias.

- `clusters.parquet`: fenotipo principal importado/estimado.
- `temporal_embeddings.npy`: embeddings temporais.
- `scan_group_assignments.parquet`: atribuicoes reconstruidas para K=2/3/5/6.
- `scan_group_summary.csv`: classificacao beneficio/neutro/risco por cluster.
- `scan_group_metadata.json`: parametros usados no scan.

### `causal/`

Resultados causais.

- `effect_estimates_psm.csv`: ATT pareado de fenotipagem/global e por fenotipo.
- `effect_by_scan_group.csv`: ATT por grupos do scan com IC95% bootstrap.
- `cate_estimates.csv`: ATE/ATT por modelos de CATE.
- `ite_estimates.parquet`: ITE por paciente.
- `ite_by_scan_group.csv`: resumo de ITE por grupos do scan.
- `patient_benefit_ranking.csv`: ranking individual por beneficio esperado.

### `counterfactual/`

Modelos contrafactuais preditivos.

- `predicted_y0_y1.parquet`: risco estimado sob nao transfundir e transfundir.
- `individual_treatment_effects.parquet`: ITE derivado de Y(1)-Y(0).
- `model_metrics.csv`: desempenho preditivo.

### `evaluation/`

Metricas agregadas.

- `predictive_metrics.csv`: AUC, AP, Brier, F1.
- `causal_metrics.csv`: resumo causal combinado.
- `sensitivity_results.csv`: sensibilidade, quando disponivel.

### `figures/`

Figuras do projeto atual.

- `scan_group_mortality_effects.png`: forest plot dos grupos do scan.
- `ite_distribution.png`
- `forest_plot_effect_by_phenotype.png`
- `counterfactual_risk_scatter.png`
- `calibration_curve.png`

### `reports/`

Relatorios em markdown.

- `research_report.md`: relatorio cientifico automatico.
- `scan_group_inference_report.md`: relatorio especifico dos grupos do scan.
- `repository_audit.md`: auditoria inicial do repositorio.

## Outputs do Artigo Anterior

Raiz:

```text
outputs/phenotype_construction/
```

### `runs/`

Execucoes preservadas do paper.

- `run_cal03_replace_full_w48/`
- `run_cal03_noreplace_w48/`

Arquivos mais importantes dentro de cada run:

```text
shared/
  t0_table.parquet
  outcomes_cohort.parquet

w48/
  features/baseline_features.parquet
  embeddings/embeddings_minirocket.parquet
  matching/matched_pairs.parquet
  reports/cluster_benefit_minirocket.csv
  reports/cluster_phenotype_minirocket.csv
```

Esses arquivos sao a fonte para `scripts/import_phenotype_outputs.py` e `scripts/causal_inference/scan_group_inference.py`.


