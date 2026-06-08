# Pipeline Fenotipagem do Paper

Este documento descreve o pipeline do artigo anterior de fenotipagem e matching transfusional. Ele foi preservado para reprodutibilidade, mas nao e o fluxo principal do projeto atual de inferencia causal temporal.

## Localizacao

Scripts:

```text
scripts/phenotype_construction/
```

Outputs:

```text
outputs/phenotype_construction/
```

## Objetivo do Pipeline Fenotipagem

O pipeline de fenotipagem:

1. constroi coorte e `t0`;
2. extrai features basais/pre-janela;
3. realiza matching transfundido-controle;
4. gera embeddings MiniRocket e TS2Vec;
5. aplica K-means;
6. produz o scan de beneficio/maleficio por cluster.

## Inputs Esperados

```text
dataset/
  timegrid_features/
  outputs_outcomes/
    outcomes_by_stay.csv
    outcomes_by_stay_full.csv
configs/
  lab_itemids.yaml
```

## Outputs Fenotipagems

Cada execucao fica em:

```text
outputs/phenotype_construction/runs/<run_id>/
```

Arquivos centrais:

```text
shared/
  t0_table.parquet
  outcomes_cohort.parquet
  schema_report.txt

w48/
  features/baseline_features.parquet
  embeddings/embeddings_minirocket.parquet
  embeddings/embeddings_ts2vec.parquet
  matching/matched_pairs.parquet
  matching/balance_diagnostics.csv
  reports/cluster_benefit_minirocket.csv
  reports/cluster_phenotype_minirocket.csv
```

## Como Rodar

Pipeline completo de fenotipagem:

```bash
python scripts/phenotype_construction/run_all_phenotyping.py --window 48 --run_id run_cal03_replace_full_w48
```

Etapas manuais:

```bash
python scripts/phenotype_construction/build_outcomes_cohort.py
python scripts/phenotype_construction/build_baseline_features.py --window 48
python scripts/phenotype_construction/match_controls.py --window 48 --replace
python scripts/phenotype_construction/embed_minirocket_temporal.py --window 48
python scripts/phenotype_construction/embed_ts2vec_temporal.py --window 48
python scripts/phenotype_construction/generate_phenotype_reports.py --window 48 --embedding minirocket
```

## Como o Projeto Atual Usa o Fenotipagem

O projeto atual nao precisa rerodar esse pipeline para fazer a analise causal. Ele pode importar os artefatos ja existentes:

```bash
python scripts/import_phenotype_outputs.py \
  --config-dir configs \
  --phenotype-root outputs/phenotype_construction \
  --run-id run_cal03_replace_full_w48 \
  --window 48 \
  --embedding minirocket \
  --keep-post-t0-features
```

E depois reconstruir os grupos do scan:

```bash
python scripts/causal_inference/scan_group_inference.py \
  --config-dir configs \
  --phenotype-root outputs/phenotype_construction \
  --run-id run_cal03_replace_full_w48 \
  --window 48 \
  --embedding minirocket \
  --k-list 2,3,5,6 \
  --bootstrap 500
```

## Observacao Metodologica

Os artefatos de fenotipagems sao fundamentais para preservar a historia do artigo, mas devem ser interpretados separadamente dos resultados novos em `outputs/causal_inference/`.


