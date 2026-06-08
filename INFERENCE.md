# INFERENCE.md — Processo de Inferência do VCIP no Notebook `VCIP-Transfusion.ipynb`

## Índice

1. [O Que "Inferência" Significa no VCIP](#1-o-que-inferência-significa-no-vcip)
2. [O Que o Modelo Prevê](#2-o-que-o-modelo-prevê)
3. [Quais Pacientes São Usados](#3-quais-pacientes-são-usados)
4. [Carregamento do Checkpoint (Seção 18)](#4-carregamento-do-checkpoint-seção-18)
5. [Geração de Sequências Candidatas](#5-geração-de-sequências-candidatas)
6. [O ELBO Como Proxy de Ranking](#6-o-elbo-como-proxy-de-ranking)
7. [GRP — Ground Truth Ranking Position](#7-grp--ground-truth-ranking-position)
8. [RCS — Ranking Correlation Score](#8-rcs--ranking-correlation-score)
9. [Parâmetros Ajustáveis e Seus Efeitos](#9-parâmetros-ajustáveis-e-seus-efeitos)
10. [Limitações deste Protótipo](#10-limitações-deste-protótipo)

---

## 1. O Que "Inferência" Significa no VCIP

No VCIP, "inferência" **não significa** fazer uma predição pontual de desfecho para um paciente. Significa encontrar, dentre todas as sequências de transfusão possíveis para os próximos τ slots, qual tem maior probabilidade de levar o paciente a um estado fisiológico alvo (`Y_target`).

O problema que o VCIP resolve durante a inferência é:

```
â* = argmin_â  ELBO(â | H_t, Y_target)
```

Menor ELBO = maior verossimilhança de atingir `Y_target`. O modelo não tenta prever o contrafactual (o que aconteceria *se* transfundíssemos) — ele diretamente ranqueia sequências pela plausibilidade causal de atingir o alvo.

No contexto deste notebook, a inferência tem um objetivo avaliativo: verificar se o modelo aprende a reconhecer que a **sequência real observada nos dados** — o que o médico realmente fez — é uma boa escolha, relativamente a sequências aleatórias. Isso é medido pelo GRP (Seção 7).

---

## 2. O Que o Modelo Prevê

O desfecho alvo neste protótipo é **`mortality_anytime` ∈ {0, 1}** — mortalidade em qualquer momento durante a internação.

O decodificador do modelo generativo (`p_θ(Y | Z_{t+τ})`) é uma MLP que mapeia o estado latente final para uma probabilidade Bernoulli. Durante a inferência, esse decodificador é usado para calcular o ELBO de reconstrução: quão compatível é o estado latente induzido pela sequência candidata com o desfecho real observado?

**O `Y_target` implícito durante a avaliação:** como usamos o `val_f` (ou `test_f`), o "alvo" é o desfecho real de cada paciente — mortality real, já observada. O modelo é avaliado por sua capacidade de ranquear a sequência real acima de sequências aleatórias.

**Nota sobre o horizonte:** com `τ = 4` slots de 60 minutos (4 horas), o modelo considera as consequências de transfundir ou não nos próximos 4 timesteps do histórico. O desfecho `mortality_anytime` é o mesmo em todos os slots (repetido), então o sinal de aprendizado é estável ao longo do horizonte.

---

## 3. Quais Pacientes São Usados

A inferência usa o **conjunto de teste** (`test_f`) — `cfg.exp.test = True`. O conjunto de teste contém aproximadamente **222 pacientes** (15% de 1.484, arredondado), separados por `subject_id` com estratificação por status de transfusão.

**Por que test_f e não val_f?** O modelo foi selecionado via `val_loss` (early stopping monitora `val_f`): o checkpoint escolhido é aquele que minimizou a perda no conjunto de validação. Usar `val_f` para reportar GRP/RCS seria uma avaliação enviesada — o modelo já foi otimizado indiretamente para esse conjunto. O `test_f` nunca influenciou nenhuma decisão de treinamento ou seleção.

O loop interno de `optimize_interventions_discrete_onetime` processa cada paciente, mas **quebra após os primeiros 100** (`if i > 99: break`). Portanto:

- **Número de pacientes avaliados: até 100** do conjunto de teste

O batch size na inferência é `val_batch_size` (definido no config do dataset), mas como os pacientes são processados individualmente no loop de otimização de sequências, efetivamente 1 paciente por vez por iteração de otimização.

---

## 4. Carregamento do Checkpoint (Seção 18)

O checkpoint é carregado **sem** usar `VAEModel.load_from_checkpoint` do PyTorch Lightning. O motivo é técnico: `save_hyperparameters(config)` (linha 24 de `vae_model.py`) serializa o `DictConfig` do OmegaConf dentro do arquivo `.ckpt`. No PyTorch 2.6, o padrão de `torch.load` mudou para `weights_only=True`, o que bloqueia a desserialização de `DictConfig` (não está na allowlist de globals seguros).

**Solução adotada**: carregar apenas o `state_dict` e reutilizar o objeto `model` já inicializado em memória:

```python
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model_inf = model
model_inf.eval()
```

`weights_only=False` é seguro aqui porque o arquivo `.ckpt` foi gerado localmente pela Seção 17, não baixado de fontes externas. O `model` em memória já tem a configuração correta (DictConfig, dataset_collection), então apenas os pesos treinados precisam ser carregados.

---

## 5. Geração de Sequências Candidatas

Para cada paciente, o modelo precisa comparar a sequência real com sequências alternativas. O processo é executado por `generate_perturbed_sequences` em `VCIP-ICML/src/utils/helper_functions.py`.

### 5.1 Composição das k=100 sequências candidatas

No modo `multilabel` (nosso caso — `cfg.dataset.treatment_mode = 'multilabel'`), com `perturb_ratio = 0.5`:

| Tipo | Quantidade (k=100) | Como gerado |
|---|---|---|
| **Aleatório** | 50 | `torch.bernoulli(tensor de 0.5)` — cara ou coroa por slot |
| **Perturbado** | 50 | Cópia da seq. real com `flip_ratio=0.2` bits invertidos aleatoriamente |
| **Real** | 1 | Adicionada como o (k+1)-ésimo elemento |

**Total avaliado pelo ELBO:** 101 sequências (100 candidatas + 1 real).

### 5.2 Por que misturar aleatório e perturbado?

- **Aleatório (50):** cobre o espaço de sequências de forma ampla — serve como baseline "ruído"
- **Perturbado (50):** explora o vizinhança da sequência real — testa variantes que diferem em 1-2 slots da decisão do médico; mais informativos clinicamente

### 5.3 Estrutura de cada sequência

Cada sequência candidata tem shape `(τ,)` — um valor binário por slot de horizonte. Com `τ = 4`, uma sequência é algo como `[0, 1, 0, 0]` (transfundir apenas no segundo slot).

**Total de sequências possíveis:** 2^τ = 2^4 = 16. Com k=100 candidatos, o espaço inteiro é coberto múltiplas vezes — neste protótipo com τ pequeno, `k` pode ser reduzido sem perda.

---

## 6. O ELBO Como Proxy de Ranking

Cada sequência candidata `â_i` recebe um score ELBO calculado por `calculate_elbo(optimize_a=True)`. O flag `optimize_a=True` seleciona o modo inferência do ELBO, que **não inclui `loss_hy`** (a perda do modelo auxiliar de histórico — irrelevante durante inferência).

O ELBO de inferência tem três termos:

```
ELBO(â_i) = λ_kl · KL_loss
           + λ_y  · reconstruction_loss       # -log p_θ(Y_target | Z_{t+τ})
           + λ_action · g_formula_correction   # Σ_s log p_θ(a_s|Z_s) - log p_θ(â|H_t)
```

**Como interpretar:**

- **KL_loss:** regularização — penaliza trajetórias latentes muito diferentes do prior. Igual para todas as sequências do mesmo paciente (depende de H_t, não de â).
- **reconstruction_loss:** quanto o estado latente final Z_{t+τ} induzido por â_i é compatível com o Y_target real do paciente. **Este é o termo mais discriminativo.**
- **g_formula_correction:** corrige o viés de confundimento — penaliza sequências que o médico dificilmente prescreveria dado o estado real do paciente, separando "sequência comum" de "sequência causalmente justificada".

**Ranking:** sequências com menor ELBO são melhores candidatas. O modelo ordena os 101 ELBOs em ordem crescente e verifica em qual posição está a sequência real (o 101º elemento da lista, adicionado por último).

---

## 7. GRP — Ground Truth Ranking Position

### 7.1 Cálculo

```python
true_sequence_rank = np.sum(elbos < elbos[-1]) + 1  # quantas candidatas têm ELBO menor que a real?
grp = (k + 1 - true_sequence_rank) / k
```

- `elbos[-1]` = ELBO da sequência real (adicionada como último elemento)
- `true_sequence_rank = 1` → a sequência real tem o menor ELBO de todas → é a melhor → GRP = 1.0
- `true_sequence_rank = k+1` → todas as candidatas têm ELBO menor → a real é a pior → GRP = 0.0

### 7.2 Interpretação

| GRP | Interpretação |
|---|---|
| **≥ 0.75** | Excelente — modelo ranqueia a seq. real no top 25% na maioria dos casos |
| **0.60–0.75** | Bom — melhoria clara sobre o baseline aleatório |
| **0.50–0.60** | Marginal — leve melhoria; possível subajuste ou ELBO mal calibrado |
| **< 0.50** | Abaixo do baseline — modelo penaliza a seq. real; revisar `λ_action` |

### 7.3 Baseline aleatório

Um modelo sem nenhum aprendizado teria GRP médio ≈ **0.50** — a sequência real estaria em posição aleatória entre k+1 opções. Qualquer GRP significativamente acima de 0.50 indica que o modelo aprendeu algo útil.

### 7.4 No paper original

No dataset de tumor sintético com τ=6, o VCIP atinge GRP médio ≈ **0.88**, enquanto baselines (CRN, RMSN, CT, ACTIN) ficam entre 0.40 e 0.60. Para este protótipo com mortalidade real e τ=4, resultados entre 0.55 e 0.70 seriam considerados satisfatórios.

---

## 8. RCS — Ranking Correlation Score

### 8.1 Definição

Correlação de Spearman entre:
- **Ranking do modelo** (por ELBO crescente — menor ELBO = melhor posição)
- **Ranking verdadeiro** (por `true_loss` crescente — menor perda real = melhor)

```
RCS = Spearman(rank_by_ELBO, rank_by_true_loss)
```

### 8.2 Limitação crítica neste protótipo

O `true_loss` é calculado por `simulate_output_after_actions`, que nesta implementação **retorna zeros** para todas as sequências:

```python
def simulate_output_after_actions(self, H_t, actions, scaling_params=None):
    return np.zeros((actions.shape[0], 1), dtype=np.float32)
```

Consequência: `true_loss` é constante para todos os candidatos → ranking verdadeiro é indefinido → **RCS = NaN** para todos os pacientes.

O RCS só seria computável com um simulador real (ou um modelo auxiliar) que conseguisse estimar o desfecho contrafactual para cada sequência candidata — o que exigiria dados observacionais adicionais ou um modelo preditivo calibrado separadamente.

### 8.3 O que fazer com RCS = NaN

Ignorar neste protótipo. O GRP é a métrica principal de avaliação.

---

## 9. Parâmetros Ajustáveis e Seus Efeitos

### 9.1 `k` — número de sequências candidatas

```python
case_infos = model_inf.optimize_interventions_discrete_onetime(k=100)
```

| Valor de `k` | Efeito |
|---|---|
| **k < 16** | Com τ=4 (2^4=16 possíveis), k pequeno não cobre o espaço — GRP pode ser inflacionado artificialmente |
| **k = 100** (padrão) | Bom equilíbrio entre cobertura e custo computacional |
| **k = 500** | Avaliação mais rigorosa — ranking mais estável, GRP mais confiável |
| **k > 1000** | Custo alto de memória/tempo; marginal ganho de informação para τ pequeno |

### 9.2 `INFER_TAU` — horizonte de inferência (τ)

```python
INFER_TAU = cfg.exp.tau   # padrão: 4 slots = 4 horas
model_inf.tau = INFER_TAU
model_inf.config.exp.tau = INFER_TAU
```

| τ | Sequências possíveis | Interpretação clínica |
|---|---|---|
| **τ = 1** | 2 (transfundir / não) | Decisão imediata apenas |
| **τ = 4** (padrão) | 16 | Plano de 4 horas |
| **τ = 8** | 256 | Plano de 8 horas; k=100 não cobre o espaço adequadamente |
| **τ = 12** | 4.096 | Horizonte amplo; k deve ser aumentado para ≥500 |

**Atenção:** τ deve ser ≤ `cfg.exp.tau` com que o modelo foi **treinado**. Usar τ maior na inferência que no treino extrapola para regiões não vistas pelo modelo.

### 9.3 `cfg.exp.num_samples` — amostras Monte Carlo

```python
cfg.exp.num_samples = 10   # padrão típico
```

O ELBO envolve uma expectativa sobre Z (variável latente). Na prática, é estimado via Monte Carlo: amostrar Z `num_samples` vezes e calcular a média do ELBO.

| Valor | Efeito |
|---|---|
| **num_samples = 1** | Rápido, mas ELBO ruidoso — ranking menos confiável |
| **num_samples = 10** (padrão) | Boa relação custo/variância |
| **num_samples = 50** | ELBO mais estável; custo ~5× maior; recomendado para avaliação final |

### 9.4 `cfg.exp.test` — qual split usar

```python
model_inf.config.exp.test = True   # True = test_f (padrão do notebook)
```

| Valor | Split usado | Quando usar |
|---|---|---|
| **True** (padrão) | `test_f` (~222 pacientes, 15% do total) | Avaliação final — resultado reportável |
| **False** | `val_f` (~222 pacientes, 15% do total) | Depuração ou inspeção qualitativa — **não reportar** |

**Regra de ouro:** o modelo foi selecionado via `val_loss` (early stopping). Reportar GRP/RCS no `val_f` seria avaliação com dados que influenciaram a escolha do checkpoint — resultado inflado. Use `val_f` apenas para inspecionar comportamento durante o desenvolvimento; o número definitivo vem sempre do `test_f`.

### 9.5 `cfg.exp.opt_epochs` — iterações de otimização discreta

O `optimize_interventions_discrete_onetime` usa o Straight-Through Estimator para otimizar logits contínuos → binarização. O número de épocas de otimização por paciente controla quantas iterações de gradient descent são feitas para refinar â*.

| Valor | Efeito |
|---|---|
| **opt_epochs = 1** | Sem otimização real — só avalia candidatas pré-geradas |
| **opt_epochs = 10–50** | Otimização leve; sequências refinadas próximas às candidatas |
| **opt_epochs = 100+** | Convergência melhor para τ grande; custo proporcional |

---

## 10. Limitações deste Protótipo

### 10.1 Janela temporal restrita

O `raw_temporal.parquet` cobre apenas as **48 horas pré-T0**. Isso significa:
- O histórico H_t é truncado a 48 slots de 60 minutos
- Não há informação pós-transfusão — o sinal causal mais importante (o que aconteceu *depois* de transfundir) está ausente
- O modelo aprende relações pré-T0, não a dinâmica de resposta à transfusão

### 10.2 Desfecho binário estático

`mortality_anytime` é o mesmo valor em todos os 47 slots temporais de um paciente. O modelo não consegue diferenciar "morreu 2 horas após T0" de "morreu 30 dias após T0". Para o VCIP funcionar bem clinicamente, o desfecho alvo deveria ser um estado fisiológico dinâmico (ex: Hb em t+τ) — o que exige o timegrid completo de 5 minutos pós-T0.

### 10.3 RCS indefinido

`simulate_output_after_actions` retorna zeros → `true_loss` constante → RCS = NaN. A métrica só seria computável com:
- Um simulador PK/PD de resposta à transfusão, ou
- Um modelo de outcome separado treinado nos dados reais

### 10.4 Potencial de overfitting ao confundimento

Os grupos M1/M2/M3 (aparente malefício no AIPW) refletem parcialmente pacientes mais graves que receberam mais transfusões *porque* estavam mais graves. O Prior de Intervenção (g-formula) corrige parte desse viés, mas a qualidade da correção depende do λ_action calibrado — e com apenas 48h de janela, o modelo não vê o confundimento temporal completo.

### 10.5 Limite de 100 pacientes

O loop de `optimize_interventions_discrete_onetime` quebra em `i > 99`. Para avaliação mais robusta (intervalos de confiança mais estreitos), modificar o código para processar todos os pacientes do split. Com ~222 pacientes no val_f, o custo adicional é ~2.2×.

---

## Resumo do Fluxo de Inferência

```
Seção 17 (treino) → melhor checkpoint (.ckpt)
                           │
                           ▼
Seção 18 (inferência)
  │
  ├── Carrega state_dict do checkpoint (via torch.load, weights_only=False)
  │
  ├── model_inf.optimize_interventions_discrete_onetime(k=100)
  │     │
  │     ├── Para cada paciente do val_f (até 100):
  │     │     ├── Extrai H_t (histórico) do CIPDataset
  │     │     ├── Gera 101 sequências: 50 aleatórias + 50 perturbadas + 1 real
  │     │     ├── Calcula ELBO para cada sequência (Monte Carlo, num_samples=10)
  │     │     ├── Ordena pelo ELBO (menor = melhor)
  │     │     └── Registra posição da seq. real → true_sequence_rank
  │     │
  │     └── Retorna lista de case_infos (dict por paciente)
  │
  └── case_infos salvo em memória
         │
         ▼
Seção 19 (avaliação)
  ├── GRP por paciente: (k+1 − true_sequence_rank) / k
  ├── GRP médio, mediana, p25, p75
  ├── RCS: NaN (simulate_output_after_actions retorna zeros)
  └── Visualização: histograma do GRP + distribuição de posições
```
