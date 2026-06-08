# Scripts

## Fluxo Atual: Inferencia Causal

Scripts do fluxo ativo:

```text
scripts/causal_inference/
```

Comando principal:

```bash
python scripts/causal_inference/run_all.py \
  --config-dir configs \
  --phenotype-root outputs/phenotype_construction \
  --run-id run_cal03_replace_full_w48 \
  --window 48 \
  --embedding minirocket \
  --k-list 2,3,5,6
```

## Construcao/Fenotipagem do Artigo

Scripts que constroem a coorte, baseline features, matching, embeddings e relatorios de fenotipagem:

```text
scripts/phenotype_construction/
```

## Reconstrucao Experimental

Scripts que reconstruiriam etapas do pipeline causal do zero. Nao fazem parte do fluxo principal porque a analise atual reaproveita os artefatos de fenotipagem ja validados:

```text
scripts/experimental_reconstruction/
```

## Utilitarios

```text
scripts/common/
```
