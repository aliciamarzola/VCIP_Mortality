# Inferencia Causal Temporal da Transfusao de Hemacias em UTI

Projeto de pesquisa em Causal Machine Learning para estimar efeitos medios, heterogeneos e contrafactuais da transfusao de concentrado de hemacias em pacientes criticos de UTI.

O fluxo atual **reaproveita o paper anterior**: coorte, `t0`, embeddings, clusters, matching e scan. Os scripts que recriavam tudo do zero foram arquivados para reduzir confusao.

## Estrutura

```text
configs/
dataset/
docs/
scripts/
  causal_inference/
  phenotype_construction/
  experimental_reconstruction/
  common/
outputs/
  causal_inference/
  phenotype_construction/
  _archive_stale/
```

## Como Rodar

Fluxo completo baseado no reaproveitamento do paper:

```bash
python scripts/causal_inference/run_all.py \
  --config-dir configs \
  --phenotype-root outputs/phenotype_construction \
  --run-id run_cal03_replace_full_w48 \
  --window 48 \
  --embedding minirocket \
  --k-list 2,3,5,6
```

Etapa central da analise por grupos do scan:

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

## Outputs

Resultados atuais:

```text
outputs/causal_inference/
  processed/
  phenotypes/
  causal/
  counterfactual/
  evaluation/
  figures/
  reports/
```

Artefatos do paper anterior:

```text
outputs/phenotype_construction/
  runs/
```

`outputs/_archive_stale/` guarda apenas resultados antigos colocados em quarentena para nao misturar execucoes antigas/sinteticas com a analise ativa.

## Analise Recomendada

- `K=5` ou `K=6`: analise principal para grupos `benefit_apparent`, `neutral_apparent` e `harm_apparent`.
- `K=2`: macrofenotipo orientado pelo artigo.
- `K=3`: sensibilidade, pois nao separa bem grupos de maleficio no output atual.

## Documentacao

- [Organizacao do projeto](docs/project_organization.md)
- [Protocolo de pesquisa](docs/research_protocol.md)
- [Mapa dos outputs](docs/output_map.md)
- [Pipeline legado](docs/legacy_pipeline.md)

## Limitacoes

Os resultados sao observacionais e dependem de ignorabilidade condicional, positividade, consistencia e ausencia de vazamento temporal. Estes resultados nao substituem decisao clinica.
