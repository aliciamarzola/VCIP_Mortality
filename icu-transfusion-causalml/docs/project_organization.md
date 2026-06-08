# Organizacao do Projeto

Este repositorio separa explicitamente o estudo anterior de fenotipagem/transfusao e o projeto atual de inferencia causal temporal.

## Estrutura Principal

```text
configs/
dataset/
scripts/
  import_phenotype_outputs.py
  causal_forest.py
  counterfactual_model.py
  counterfactual_evaluation.py
  evaluation.py
  scan_group_inference.py
  generate_report.py
  run_all.py
  experimental_reconstruction/
  phenotype_construction/
outputs/
  causal_inference/
  phenotype_construction/
docs/
```

## Separacao Conceitual

### `outputs/phenotype_construction/`

Contem artefatos do artigo anterior. Esta pasta deve ser tratada como fonte legada e quase imutavel.

Uso principal:

- recuperar `t0`, coorte, matching, embeddings e tabelas de scan do artigo;
- reproduzir o raciocinio original de fenotipos de beneficio/risco;
- servir como camada de descoberta de heterogeneidade.

### `outputs/causal_inference/`

Contem os resultados do projeto atual:

- dados padronizados;
- efeitos causais;
- modelos contrafactuais;
- grupos do scan;
- relatorios.

## Scripts Atuais

O fluxo atual usa nomes de acao, sem numeracao:

```text
import_phenotype_outputs.py
causal_forest.py
counterfactual_model.py
counterfactual_evaluation.py
evaluation.py
scan_group_inference.py
generate_report.py
run_all.py
```

## Scripts Arquivados

Scripts de reconstrucao completa que nao sao usados no fluxo atual:

```text
scripts/experimental_reconstruction/
```

Scripts do paper anterior:

```text
scripts/phenotype_construction/
```

## Regra de Ouro

- `phenotype_construction` responde: "o que o artigo anterior produziu?"
- `causal_inference` responde: "qual inferencia causal estamos fazendo agora?"
- `experimental_reconstruction` guarda codigo que reconstruiria tudo, mas nao deve poluir o fluxo principal.



