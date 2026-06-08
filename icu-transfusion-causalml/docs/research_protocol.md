# Protocolo de Pesquisa

## Titulo

Inferencia causal temporal do efeito da transfusao de hemacias em pacientes criticos de UTI usando trajetorias fisiologicas e fenotipos latentes.

## Questao Principal

O efeito da transfusao de concentrado de hemacias sobre mortalidade e suporte organico varia conforme fenotipos fisiologicos derivados de trajetorias pre-intervencao?

## Hipotese

A transfusao nao possui efeito homogeneo. Pacientes com perfusao preservada, anemia dinamica e menor falencia organica podem apresentar beneficio aparente, enquanto pacientes com falencia organica, hipoperfusao e alta carga de suporte podem apresentar neutralidade ou risco associado.

## Unidade de Analise

Permanencia em UTI (`stay_id`).

## Tratamento

`A = 1`: paciente recebeu transfusao de concentrado de hemacias.

`A = 0`: paciente controle nao recebeu transfusao.

## Ponto de Decisao

Para tratados, `t0` e a primeira transfusao. Para controles, `pseudo-t0` e definido por compatibilidade temporal com os tratados.

## Desfecho Primario

`mortality_anytime`.

## Desfechos Secundarios

- `vm_time_hours`
- `ventilation_hours`
- `any_vasopressor`
- `nee_mcgkgmin_max`
- `rrt_on`
- `icu_los_hours`
- `sofa_delta`, quando disponivel

## Estrategia Analitica

### 1. Fonte Legada

Reaproveitar do artigo anterior:

- coorte;
- `t0`;
- matching;
- embeddings MiniRocket;
- clusters;
- scan de beneficio/maleficio;
- tabelas de balanceamento.

Esses artefatos ficam em `outputs/phenotype_construction/`.

### 2. Camada Causal Atual

Padronizar os artefatos em `outputs/causal_inference/` e executar:

- PSM/ATT de fenotipagem;
- CATE/ITE por modelos de ML;
- modelo contrafactual de dois modelos;
- inferencia por grupos do scan.

### 3. Analise Primaria Recomendada

Usar `K=5` ou `K=6` como analise principal de heterogeneidade operacional, pois esses valores separam:

- `benefit_apparent`;
- `neutral_apparent`;
- `harm_apparent`.

O `K=2` deve ser usado como macrofenotipo orientado pelo artigo:

- `macro_benefit_relative`;
- `macro_risk_high_support`.

O `K=3` deve ser sensibilidade, pois no output atual nao separa grupos de maleficio.

## Estimandos

### ATT por grupo

Estimando principal:

```text
E[Y(1) - Y(0) | A = 1, grupo = g]
```

Interpretacao: entre os pacientes transfundidos do grupo `g`, qual foi a diferenca estimada em relacao aos controles pareados?

### CATE por grupo

Estimando exploratorio:

```text
E[Y(1) - Y(0) | grupo = g]
```

### ITE

Estimando exploratorio individual:

```text
Y_i(1) - Y_i(0)
```

## Interpretacao dos Grupos

- `benefit_apparent`: mortalidade menor nos transfundidos pareados.
- `harm_apparent`: mortalidade maior nos transfundidos pareados.
- `neutral_apparent`: efeito pequeno ou IC cruzando zero.
- `macro_risk_high_support`: grupo com maior carga de suporte organico, mesmo quando mortalidade pareada nao aumenta.

## Limitacoes

- Estudo observacional.
- Risco de confundimento residual.
- Risco de circularidade se grupos forem descobertos e testados no mesmo conjunto.
- Necessidade de verificar se features sao estritamente pre-`t0`.
- PEHE individual verdadeiro nao e observavel.

## Proximos Passos Metodologicos

1. Rodar uma versao estritamente pre-`t0` dos features.
2. Repetir a inferencia por grupos com bootstrap e sensibilidade por K.
3. Comparar `K=2`, `K=5` e `K=6`.
4. Validar overlap e balanceamento dentro de cada grupo.
5. Produzir tabela final com ATT, IC95%, n pares e desfechos secundarios.



