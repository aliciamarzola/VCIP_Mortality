# CLAUDE.md — icu-transfusion-causalml

## Propósito do Projeto

Inferência causal do efeito da **transfusão de concentrado de hemácias (PRBC)** em pacientes críticos de UTI usando dados do MIMIC-IV. O objetivo final é treinar um **VCIP (Variational Counterfactual Intervention Planning to Achieve Target Outcomes
) - disponível no PDF vcip.pdf** — modelo contrafactual que estima, para cada paciente e momento, se a transfusão traria benefício, malefício ou seria neutra, permitindo decisão clínica personalizada.

O projeto atual (pipeline de ensaio-alvo) é uma **etapa intermediária**: já produz inferência causal com estimandos AIPW, IPTW e overlap weighting por grupos de heterogeneidade. O VCIP é a etapa seguinte, dependente de dados temporais completos.

O objetivo é comparar a inferência causal por estimadores causais (baseline) com o resultado do modelo contrafactual VCIP. Ele será aplicado a partir do raw_temporal.parquet

---

## Coorte

### Origem

Os dados são do **MIMIC-IV** (ICU). O pipeline atual **importa** artefatos do pipeline legado (`cluster-transfusion`, run `cal03_replace_full_w48`) — não processa o MIMIC bruto diretamente.

### Critérios de Elegibilidade

| Critério | Valor | Onde definido |
|---|---|---|
| Hemoglobina ≤ 8 g/dL | Limiar de anemia severa — T0 é o primeiro momento em que Hb cruza esse valor | `target_trial_config.yaml` |
| Idade ≥ 18 anos | Adultos apenas | `config.yaml` |
| Primeira internação em UTI | Por `subject_id`, ordenado por `min_time` | `build_cohort.py` |
| Observação mínima antes de T0 ≥ 6h | Garante janela de features pré-T0 | `target_trial_config.yaml` |
| Observação total ≥ 24h | Exclui stays muito curtos | `config.yaml` |

### Tamanho da Coorte

- **1.484 pacientes elegíveis** (506 tratados + 978 controles, razão ≈ 1:2)
- O pipeline legado tinha ~5.800 pacientes; a coorte atual é o subconjunto com Hb ≤ 8 g/dL

### Definição de T0

- **Tratados**: T0 = timestamp da primeira transfusão de PRBC
- **Controles**: pseudo-T0 = `min_time + mediana(offset dos tratados)` — garante comparabilidade temporal
- Coluna `pseudo_t0` (flag 0/1) identifica qual tipo é cada paciente

---

## Pipeline de Dados

### Resolução do Timegrid

| Nível | Resolução | Cobertura | Localização |
|---|---|---|---|
| Timegrid bruto (legado) | **5 minutos** | Internação completa | Servidor remoto |
| `raw_temporal.parquet` | **60 minutos** (média dos bins de 5 min) | 48h pré-T0 apenas | Servidor remoto |
| `trial_dataset.parquet` | Escalar (agregação) | 48h pré-T0 → 1 linha/paciente | **Local** |

### Tratamento de Missings

1. **No timegrid de 5 min** (legado, `impute.py`): forward-fill por `stay_id` ordenado por `tbin` → defaults estáticos → flags `*_imputed` (int8)
2. **No `trial_dataset.parquet`** (features estáticas): `dropna()` — NaN propaga se a variável não foi medida na janela de 48h. Sem imputação adicional.
3. **No tensor temporal** (`build_temporal_tensor`): ffill dentro de cada série → fill com mediana da partição de treino → z-score normalization

---

## Arquivos Parquet Disponíveis Localmente

Todos em `outputs/target_trial/`:

| Arquivo | Papel | Shape |
|---|---|---|
| `processed/eligibility.parquet` | Critérios de elegibilidade por paciente | 1.484 × 7 |
| `processed/treatment_assignment.parquet` | T0 e status de tratamento | 1.484 × 11 |
| **`processed/trial_dataset.parquet`** | **Coorte principal: features pré-T0 + desfechos** | **1.484 × 121** |
| `causal/nuisance_predictions.parquet` | Propensity score, μ0, μ1, pesos AIPW | 1.484 × 10 |
| `final_groups/final_group_assignments.parquet` | Grupos B1, B2, M1, M2, M3 | 1.484 × 10 |
| `final_groups/final_group_patient_counterfactuals.parquet` | Y(0), Y(1), ITE por paciente | 924 × 10 |
| `vcip_lite/vcip_lite_individual_counterfactuals.parquet` | Y(0), Y(1) modelo simplificado | 1.484 × 23 |
| `legacy_crosswalk/legacy_k2_assignments.parquet` | Crosswalk com K=2 do artigo legado | 2.968 × 17 |

### Features do `trial_dataset.parquet`

**Estáticas**: `age`, `sex`, `bmi`

**Temporais pré-T0 (janela 48h)** — 9 agregações para cada variável:
`_mean`, `_median`, `_min`, `_max`, `_std`, `_first`, `_last`, `_delta`, `_slope`

Variáveis: `hemoglobin`, `lactate`, `creatinine`, `platelets`, `heart_rate`, `mbp`, `resp_rate`, `spo2`, `sofa`, `any_vasopressor`, `ventilation_on`

**Desfechos**: `mortality_anytime` (primário), `ventilation_hours`, `rrt_on`, `any_vasopressor`, `nee_mcgkgmin_max`, `icu_los_hours`

**Grupos de heterogeneidade (AIPW)**:
- **B1, B2**: benefício aparente da transfusão (AIPW ATE ≈ −0.31, −0.35)
- **M1, M2, M3**: malefício aparente (AIPW ATE ≈ +0.24, +0.45, +0.31)

---

## Arquitetura VCIP — Objetivo Futuro

### Propósito

O VCIP aprende uma política de intervenção: dado o histórico fisiológico do paciente até o instante `t`, estimar `Y(1) − Y(0)` — o benefício contrafactual de transfundir vs. não transfundir naquele momento.

### Spec de Features (seção 4 do protocolo de pesquisa)

**Grupo A — Sinais Vitais (por slot de 5 min)**
FC, PAM, SBP, DBP, SpO2, FR, Temperatura, GCS

**Grupo B — Laboratório (por slot, forward-filled)**
Hb, Hct, Plaquetas, INR, Lactato, Creatinina, Bilirrubinas, Leucócitos, Na, K, pH, pCO2, pO2, Base Excess, HCO3

**Grupo C — Terapias concorrentes**
Norepinefrina (µg/kg/min), Ventilação mecânica, Cristalóides (mL/h), Furosemida (mg/h), PRBC (At), FFP/plaquetas, Heparina

**Features derivadas (Δ e rolling)**
- ΔHb em 30 min, 2h, 6h (= 6, 24, 72 slots de 5 min)
- Rolling mean/std em janelas de 1h (12 slots) e 4h (48 slots)
- Tempo desde última transfusão
- Carga cumulativa de vasopressor (4h)
- Balanço hídrico cumulativo
- Hora do dia: `sin(2π·h/24)`, `cos(2π·h/24)`
- Dia de internação em UTI

**Desfecho alvo**: `[Hb, PAM, Lactato, SpO2]` no instante `t+τ`
**Dimensões**: `dim_x ≈ 70`, `dim_y = 4`, `dim_v ≈ 20`, `tau = 24 slots (2h)`, `max_t = 864 slots (72h)`

### Janelamento (Rolling Window)

```python
# Por internação i com T slots, t_min = 48 (4h mínimo de contexto):
for t in range(t_min, T - tau):
    sample = {
        "X": X[0:t],          # histórico de covariáveis
        "A": A[0:t+tau],      # histórico + intervenções futuras
        "Y": Y[0:t],          # histórico de desfechos
        "V": V,               # features estáticas
        "Y_target": Y[t+tau]  # alvo
    }
# Subsampling a cada 12 slots (1h) para reduzir correlação
```

### Divisão Treino/Val/Teste

Por `subject_id` — nunca por `stay_id` — para evitar vazamento entre internações do mesmo paciente. Estratificação por `transfused`.

---

## Como o `raw_temporal.parquet` Poderia ser Usado para o VCIP

### Limitações Críticas (por que não é suficiente)

1. **Janela errada**: cobre apenas as 48h **pré-T0** — o VCIP precisa da internação completa (o que acontece pós-transfusão é o sinal de aprendizado)
2. **Resolução**: 60 min em vez de 5 min → max_t = 72 slots em vez de 864; ΔHb(30 min) é inviável
3. **Variáveis faltando**: não tem GCS, INR, bilirrubinas, Na, K, gasometria, dose específica de vasopressor por droga, cristalóides/furosemida, FFP
4. **Sem sinal de imputação**: foward fill e defaults estáticos foram usados no time-grid anterior de 5 minutos mas, ao passar pro time-grid de 60 minutos do raw_temporal, a rastreabilidade dessa imputação foi perdida.

### O Que É Viável com `raw_temporal.parquet`

Um **protótipo de VCIP com escopo reduzido**:
- Variáveis disponíveis: Hb, Lactato, Creatinina, Plaquetas, FC, PAM, FR, SpO2, SOFA, vasopressor (binário), ventilação
- Resolução de 60 min → janela de 48h = 48 timesteps por paciente
- Target reduzido: `[Hb, PAM, Lactato, SpO2]` ainda é calculável
- Serve para **validar a arquitetura** antes de ter o timegrid completo

Código-chave: `dataset/feature_engineering.py:build_temporal_tensor()` já implementa o tensor com ffill + mediana + z-score para esse caso.

---

## VCIP — Algoritmo, Pipeline e Aplicação para Transfusão de Hemácias

### 1. Por Que o VCIP e Não Apenas os Estimadores Causais Existentes

O pipeline atual (AIPW, IPTW, overlap weighting) estima o **efeito médio** da transfusão nos grupos B1/B2/M1/M2/M3. Esse resultado é estático — ele diz "em média, pacientes do grupo B1 se beneficiam de transfundir", mas **não** diz quando transfundir, em qual sequência, nem o que fazer quando o estado do paciente muda hora a hora.

O VCIP resolve um problema fundamentalmente diferente:

> **Dado o histórico fisiológico completo deste paciente até agora, qual é a sequência ótima de transfusões nas próximas τ horas para atingir um estado-alvo específico?**

A diferença central está na pergunta. Estimadores causais clássicos perguntam: *"Se eu tivesse transfundido, o desfecho seria melhor?"* — isso exige estimar um contrafactual que não pode ser observado. O VCIP redefine o problema: em vez de estimar o contrafactual diretamente (o que acumula erros de predição irrastreáveis), o VCIP modela diretamente a **probabilidade de atingir o estado-alvo** dado a sequência de intervenção e o histórico.

Isso tem uma consequência prática importante: enquanto métodos como CRN e RMSN ficam cada vez piores conforme o horizonte de predição τ aumenta (erros se acumulam), o VCIP mantém performance estável ou melhora — porque ele não depende de predições intermediárias encadeadas.

---

### 2. A Intuição Fundamental

Imagine dois planos de transfusão para o mesmo paciente:

- **Plano A**: transfundir agora → o modelo de contrafactual prevê Hb = 8.5 g/dL em 4 horas (próximo ao alvo)
- **Plano B**: não transfundir → o modelo prevê Hb = 7.8 g/dL (mais longe do alvo)

O método clássico escolheria o Plano A. **Mas**: se o modelo de contrafactual tiver erro de ±0.8 g/dL nesse paciente, os rankings podem estar invertidos — e não há como saber, porque o contrafactual verdadeiro é inobservável.

O VCIP não tenta prever o desfecho de cada plano. Em vez disso, ele pergunta: **dada toda a trajetória fisiológica observada deste paciente, qual plano tem maior probabilidade de levar ao alvo?** Isso é aprendido diretamente dos dados históricos de pacientes reais que atingiram (ou não) estados similares — sem precisar observar o contrafactual de nenhum indivíduo.

---

### 3. O Algoritmo VCIP

O VCIP é um **modelo generativo variacional com estrutura causal temporal**. Ele reformula o objetivo de otimização de intervenção como um problema de verossimilhança condicional e o resolve via inferência variacional (VAE), usando a g-formula de Robins para garantir identificabilidade causal a partir de dados observacionais.

#### 3.1 Representação do Paciente

Cada paciente é representado por uma trajetória temporal. Em cada instante `t`, o modelo conhece:

- **H_t** (histórico): todas as covariáveis X, intervenções passadas A e desfechos Y observados até t, mais as features estáticas V. É tudo que um médico veria na beira do leito.
- **â_{t,τ}** (plano de intervenção): a sequência de decisões de transfusão para os próximos τ passos — essa é a variável de decisão.
- **Y_target**: o estado fisiológico que se deseja atingir em t+τ (ex: Hb ∈ [7, 9] g/dL, com PAM ≥ 65 mmHg e Lactato ≤ 2 mmol/L).

O **objetivo de otimização** é encontrar:

```
â* = argmax_â  P(Y[â_{t,τ}] = Y_target | H_t)
```

Ou seja: a sequência de transfusões que **maximiza a probabilidade** de atingir o alvo, condicionada no histórico do paciente.

#### 3.2 O Modelo Causal com Variáveis Latentes

O VCIP assume que o estado real do paciente nunca é completamente observável — H_t captura o que medimos, mas o estado fisiológico verdadeiro inclui reservas funcionais, resposta inflamatória sistêmica e outras dinâmicas ocultas. Por isso, o modelo introduz **estados latentes Z_s** em cada passo s ∈ {t, t+1, ..., t+τ}.

A estrutura causal segue o grafo:

```
H_t → Z_t → Z_{t+1} → Z_{t+2} → ... → Z_{t+τ} → Y_{t+τ}
              ↑            ↑                ↑
             a_t         a_{t+1}          a_{t+τ-1}
```

Cada Z_s encapsula o estado fisiológico latente no passo s. As intervenções a_s agem sobre Z_s para determinar Z_{s+1}. O desfecho observável Y_{t+τ} é gerado a partir do estado latente final Z_{t+τ}.

Essa estrutura tem dois modelos paralelos:

**Modelo Generativo p_θ** — "o que o modelo aprendeu sobre a dinâmica do mundo":
- Como H_t determina o estado inicial Z_t
- Como cada intervenção a_s transforma Z_s em Z_{s+1}
- Como o estado latente final Z_{t+τ} se traduz em desfechos observáveis

**Modelo de Inferência q_φ** — "dado que Y_target foi atingido, qual trajetória latente faz sentido?":
- Componente variacional (tipo VAE encoder)
- Durante o treinamento, este modelo **vê Y_target** — ele guia o espaço latente para regiões causalmente compatíveis com atingir o alvo
- Durante a inferência, apenas p_θ é usado (Y_target futuro não está disponível)

#### 3.3 Identificabilidade Causal pela G-Formula

O problema central de treinar com dados observacionais é o **confundimento por indicação**: pacientes mais anêmicos recebem mais transfusões, criando uma associação espúria entre transfusão e piora (os mais doentes foram mais transfundidos *porque* estavam mais doentes).

O VCIP usa a **g-formula de Robins (1986)** para corrigir isso. A ideia é que a verossimilhança interventional — o que queremos otimizar — pode ser expressa em termos de quantidades observáveis. O VCIP constrói dois ELBOs (Evidence Lower Bounds):

- **ELBO₁**: lower bound da verossimilhança interventional `log P(Y[â] = Y_target | H_t)`
- **ELBO₂**: lower bound da verossimilhança observacional `log P(Y_{t+τ} = Y_target | H_t, â_{t,τ})`

O teorema central do paper demonstra que, sob as assunções de consistência, positividade sequencial e ignorabilidade sequencial, otimizar ELBO₁ pode ser aproximado por otimizar:

```
ELBO₂  −  Σ_{s=t}^{t+τ-1} E[log p_θ(a_s | Z_s)]  +  log p_θ(â_{t,τ} | H_t)
```

O terceiro termo (g-formula) é a diferença entre o log-prob da intervenção passo a passo dado o estado latente e o log-prob da sequência completa dado o histórico. Ele mede e corrige o desvio entre o regime observacional (médicos prescrevendo com base no estado clínico visível) e o regime interventional (transfundir independentemente do estado).

---

### 4. Pipeline — Componente por Componente

#### 4.1 Pré-processamento e Janelamento

O ponto de partida é o timegrid temporal dos pacientes elegíveis (Hb ≤ 8 g/dL, ≥ 4h de histórico antes de T0). Para cada paciente e cada instante t válido, o pipeline cria uma amostra de treinamento:

- O **histórico H_t** vai do início da internação até t
- A **sequência de intervenção â_{t,τ}** são as τ decisões de transfusão seguintes
- O **alvo Y_target** são os desfechos observados em t+τ

Como os históricos têm comprimentos variáveis, eles são padded com zeros e acompanhados de uma máscara binária. O subsampling a cada hora reduz a autocorrelação entre amostras do mesmo paciente.

#### 4.2 History Encoder (LSTM)

**Pergunta que responde:** *"O que sei sobre este paciente agora?"*

Um LSTM processa a sequência temporal `[x_t, a_{t-1}, y_t]` passo a passo, acumulando informação sobre a trajetória clínica. A cada passo, o LSTM integra o estado clínico atual, a intervenção anterior e o desfecho anterior. As features estáticas V (idade, sexo, peso, diagnóstico) são incorporadas via projeção separada e fundidas com o hidden state final.

A saída é um vetor **h_t** — uma "impressão digital" do contexto clínico completo do paciente até t. Esse vetor condensa informações que um médico experiente integraria mentalmente: a tendência de queda da Hb nas últimas horas, a instabilidade hemodinâmica, a resposta a intervenções anteriores.

#### 4.3 Codificador de Intervenção (LSTM de Sequência)

**Pergunta que responde:** *"Como esta sequência de transfusões se estrutura ao longo do tempo?"*

Um segundo LSTM processa a sequência de decisões â_{t,τ} = [a_t, ..., a_{t+τ-1}]. Em cada passo s, o hidden state representa o contexto de intervenção acumulado até s. Esse contexto é usado nas transições de estado latente — permitindo que o modelo saiba não apenas "vou transfundir?", mas "qual é o padrão de intervenção desta sequência?"

Isso é importante porque `[1,0,0,0,1,0,0,0]` (transfusão fracionada) tem dinâmica fisiológica diferente de `[1,1,0,0,0,0,0,0]` (dose concentrada no início), mesmo que a quantidade total seja igual.

#### 4.4 Modelo Generativo p_θ — Dinâmica Fisiológica

**Pergunta que responde:** *"Dada uma sequência de transfusões, como o estado do paciente deve evoluir?"*

O modelo generativo tem três sub-componentes:

**Estado inicial — p_θ(Z_t | H_t):** uma MLP que recebe h_t e produz os parâmetros (média e desvio-padrão) de uma Gaussiana sobre o estado latente inicial Z_t. Este é o prior do modelo — dado o histórico, qual estado fisiológico latente é consistente?

**Transição de estado — p_θ(Z_{s+1} | Z_s, ctx_s):** uma MLP que recebe o estado latente atual Z_s concatenado com o contexto de intervenção ctx_s e produz a distribuição do próximo estado Z_{s+1}. Este componente aprende a dinâmica real: como transfundir 1 unidade de PRBC num paciente com Hb de 6.5 g/dL afeta o estado fisiológico nas próximas 2 horas, dependendo de se ele está em vasopressores, ventilado, etc.

**Decodificador de desfecho — p_θ(Y_{t+τ} | Z_{t+τ}):** uma MLP que mapeia o estado latente final para os desfechos observáveis (Hb, PAM, Lactato, SpO2). É o mapeamento "estado interno → o que o médico vê no monitor".

#### 4.5 Modelo de Inferência q_φ — Posterior Variacional

**Pergunta que responde:** *"Dada uma sequência de intervenções e sabendo que Y_target foi atingido, qual trajetória latente faz sentido?"*

O componente chave do VAE. Tem dois sub-componentes análogos ao modelo generativo, mas com parâmetros independentes:

**Posterior inicial — q_φ(Z_t | H_t):** análogo ao prior p_θ, mas com parâmetros próprios otimizados para aproximar a verdadeira posterior.

**Posterior de transição — q_φ(Z_{s+1} | Y_target, ctx_s, Z_s):** o componente diferencia o VAE de um modelo generativo simples — ele **condicionado em Y_target**. Durante o treinamento, o modelo de inferência sabe para onde o paciente foi, e usa isso para inferir quais trajetórias latentes são compatíveis. Isso "ensina" o modelo generativo sobre quais regiões do espaço latente levam a desfechos bons vs. ruins.

#### 4.6 Prior de Intervenção — O Modelo do Médico

**Pergunta que responde:** *"Quanto do padrão de transfusão nos dados reflete a condição do paciente vs. o viés de quem prescreveu?"*

Este componente modela como médicos tomam decisões de transfusão, e é essencial para a correção de confundimento. Tem duas distribuições:

**Passo único — p_θ(a_s | Z_s):** dado o estado latente atual do paciente, qual é a probabilidade de um médico transfundir? Uma MLP aprende essa relação — estados com Hb latente muito baixa têm alta probabilidade de receber transfusão.

**Sequência completa — p_θ(â_{t,τ} | H_t):** qual é a probabilidade da sequência de decisões completa dado o histórico? Um LSTM auto-regressivo modela cada decisão condicionada nas anteriores e no histórico h_t.

A diferença entre esses dois termos compõe a **g-formula**: ela captura o quanto a política observacional dos médicos se desvia do que o estado latente por si só justificaria — e é exatamente isso que precisa ser corrigido para que as recomendações do modelo sejam causalmente válidas.

#### 4.7 Loss ELBO — O Objetivo de Treinamento

O ELBO é minimizado durante o treinamento (equivale a maximizar a verossimilhança de atingir Y_target). Tem três termos com papéis distintos:

**Termo 1 — KL Divergence (regularização):**
`Σ_s KL(q_φ(Z_s|·) || p_θ(Z_s|·))`
Força o espaço latente inferido (q_φ, que vê Y_target) a ser próximo do espaço generativo (p_θ, que não vê). Impede que o modelo de inferência "memorize" trajetórias em vez de generalizar. Durante a inferência (quando Y_target futuro não está disponível), apenas p_θ é usado — portanto p_θ precisa ter aprendido a representar bem o espaço latente por conta própria.

**Termo 2 — Reconstrução (objetivo principal):**
`- E[log p_θ(Y_target | Z_{t+τ})]`
Maximiza a probabilidade de que o estado latente final Z_{t+τ} corresponda ao Y_target observado. É a pressão fundamental de aprendizado: as trajetórias latentes induzidas pela sequência de intervenção devem convergir para regiões do espaço que decodificam para o alvo clínico.

**Termo 3 — G-formula (correção de confundimento):**
`λ · [Σ_s E[log p_θ(a_s|Z_s)] − log p_θ(â_{t,τ}|H_t)]`
Penaliza sequências de intervenção que são plausíveis como política do médico mas não são justificadas pelo estado latente. Quando λ=1 (valor teórico), isso garante que otimizar o ELBO observacional aproxima otimizar a verossimilhança interventional verdadeira.

---

### 5. Treinamento

O treinamento usa os pares `(H_t, â_{t,τ}, Y_{t+τ})` extraídos do timegrid dos pacientes elegíveis. Em cada batch:

1. O History Encoder codifica H_t → h_t para cada paciente
2. O Codificador de Intervenção processa â_{t,τ} → contexto por passo
3. O modelo de inferência q_φ usa h_t + Y_target para inferir a trajetória Z_t...Z_{t+τ}
4. O modelo generativo p_θ "tenta acompanhar" via o termo KL
5. O decodificador verifica se Z_{t+τ} corresponde ao Y_target observado
6. O prior de intervenção calcula o ajuste da g-formula
7. Backpropagation atualiza todos os parâmetros θ e φ conjuntamente

**O que cada componente aprende ao longo das épocas:**
- O LSTM de histórico aprende quais padrões temporais (tendência de queda de Hb, instabilidade de PAM) são mais preditivos do benefício de transfundir
- O espaço latente aprende a representar estados fisiológicos causalmente relevantes — estados que levam a desfechos diferentes têm representações diferentes em Z
- A transição de estado aprende a dinâmica de resposta à transfusão — como a Hb responde dependendo do status de vasopressor, ventilação, volume de hidratação
- O prior de intervenção aprende o "padrão prescritivo" dos médicos do MIMIC — a distribuição de quando e quanto transfundir, para que o modelo possa corrigir esse viés

A divisão treino/val/teste é por `subject_id` para evitar vazamento entre internações do mesmo paciente, estratificada por status de transfusão.

---

### 6. Inferência — Sequência Ótima de Transfusão (Algorithm 1)

Após o treinamento, os parâmetros θ e φ são congelados. Para um novo paciente em um instante t:

**Entrada:**
- H_t = histórico clínico real do paciente até t (codificado em h_t pelo History Encoder)
- Y_target = alvo clínico desejado (ex: `[Hb=8.5, PAM=70, Lactato=1.8, SpO2=97]`)
- τ = horizonte de decisão (ex: 24 slots = 2 horas com resolução de 5 min)

**Processo de otimização:**

A sequência â_{t,τ} se torna a variável a ser aprendida. O modelo inicializa â aleatoriamente (ou com todos zeros — "não transfundir como baseline") e então iterativamente:

1. Passa â pelo Codificador de Intervenção → contexto por passo
2. Rola a trajetória latente Z_t → Z_{t+τ} usando o modelo generativo p_θ
3. Calcula o ELBO da sequência atual — quão provável é que esta sequência leve a Y_target, dado H_t?
4. Atualiza â via gradient descent na direção que reduz o ELBO

Para intervenções binárias (transfundir: sim/não), o VCIP otimiza logits contínuos e usa o **Straight-Through Estimator** para que o gradiente passe pela binarização, mantendo diferenciabilidade.

**Saída:**
A sequência â* = a sequência que minimizou o ELBO após `opt_epochs` iterações. Em termos clínicos, â*[s] = 1 indica que o modelo recomenda transfundir no slot s, e â*[s] = 0 indica que não transfundir é mais provável de atingir Y_target.

**Exemplo de saída para um paciente:**
```
â* = [0, 0, 1, 0, 0, 1, 0, 0]  (τ = 8 slots = 40 min com resolução 5 min)
      →  transfundir nos slots 3 e 6, aguardar nos demais
      →  ELBO_ótimo = -2.41   (vs. ELBO_nunca = -1.03, ELBO_sempre = -1.87)
```
Menor ELBO = maior probabilidade de atingir Y_target.

---

### 7. Avaliação — GRP e RCS

Como os contrafactuais verdadeiros são inobserváveis em dados reais, a avaliação usa um framework de **ranking**:

**Ground Truth Ranking Position (GRP):**
Para cada paciente de teste, geram-se k sequências candidatas (aleatórias + perturbações da sequência real). O modelo ranqueia todas as k+1 sequências (incluindo a real) pelo ELBO. O GRP mede a posição normalizada da sequência real:

```
GRP = (k + 1 − ξ) / k
```

onde ξ é a posição da sequência real no ranking (1 = melhor). GRP = 1.0 significa que o modelo sempre identifica a sequência verdadeira como a melhor candidata.

**Ranking Correlation Score (RCS):**
Spearman entre o ranking predito pelo modelo e o ranking verdadeiro (baseado na distância real ao alvo). RCS ∈ [-1, 1], com valores próximos de 1 indicando que o modelo ordena corretamente as sequências por qualidade.

No paper original (tumor dataset, γ=4), o VCIP atinge GRP médio ≈ 0.88 e RCS médio > 0.70 para τ=6, enquanto os baselines (CRN, RMSN, CT, ACTIN) ficam entre 0.40 e 0.60 e degradam com τ crescente.

---

### 8. Aplicação Específica: Transfusão de Hemácias no MIMIC-IV

#### Definição do Problema para Este Projeto

| Componente | Definição |
|---|---|
| **Pacientes** | 1.484 elegíveis com Hb ≤ 8 g/dL na coorte atual |
| **T0** | Primeiro momento em que Hb cruza 8 g/dL |
| **Intervenção A_t** | Transfusão de PRBC (binário: sim/não por slot); ou dose em unidades (contínuo) |
| **Covariáveis X_t** | Hb, Hct, Plaquetas, INR, Lactato, Creatinina, FC, PAM, SpO2, FR, SOFA, vasopressor, ventilação (+ features derivadas Δ e rolling) |
| **Features estáticas V** | Idade, sexo, BMI, diagnóstico de admissão, grupos B1/B2/M1/M2/M3 do AIPW |
| **Desfecho alvo Y_target** | `[Hb, PAM, Lactato, SpO2]` em t+τ; ou Hb isolada para protótipo |
| **Horizonte τ** | 24 slots de 5 min (2h) para timegrid completo; 4 slots de 60 min (4h) para protótipo com raw_temporal |

#### Confundimento por Indicação no Contexto de Transfusão

O confundimento por indicação é especialmente severo nesta coorte: pacientes com Hb mais baixa, mais instáveis hemodinamicamente e em uso de vasopressores têm maior probabilidade de receber transfusão — mas também têm pior prognóstico independentemente da transfusão. Os grupos M1/M2/M3 (aparente malefício no AIPW) provavelmente refletem parcialmente esse fenômeno.

O Prior de Intervenção do VCIP captura exatamente esse padrão: ao modelar `p_θ(a_s | Z_s)` (probabilidade de transfundir dado o estado latente), o modelo aprende que PRBC é mais provável quando Hb latente está baixa e o estado hemodinâmico está deteriorando. A g-formula então corrige esse viés ao pontuar sequências candidatas durante a inferência.

#### Integração com os Estimadores Causais Existentes

Os grupos de heterogeneidade (B1, B2, M1, M2, M3) derivados do AIPW podem ser incorporados ao VCIP como features estáticas em V. Isso permite que o modelo aprenda políticas de transfusão diferenciadas por grupo — por exemplo, um limiar de Hb mais conservador para pacientes do grupo M3.

Adicionalmente, o VCIP pode ser usado para **validar** a classificação AIPW: se o VCIP recomenda transfusão frequentemente para pacientes do grupo B1 e raramente para M3, isso corrobora a distinção entre os grupos. Se as recomendações forem indiferentes à classificação AIPW, isso sugere que a heterogeneidade capturada pelo AIPW não se traduz em heterogeneidade de política ótima.

#### Protótipo com `raw_temporal.parquet` (Etapa Imediata)

Com os dados disponíveis atualmente, é possível treinar um VCIP de escopo reduzido:

- **dim_x ≈ 11**: Hb, Lactato, Creatinina, Plaquetas, FC, PAM, FR, SpO2, SOFA, vasopressor (binário), ventilação
- **dim_y = 1 ou 4**: Hb isolada ou `[Hb, PAM, Lactato, SpO2]`
- **dim_v ≈ 7**: idade, sexo, BMI, grupo AIPW (one-hot), pseudo_t0
- **τ = 4 slots**: horizonte de 4h com resolução horária
- **max_t = 48 slots**: janela de 48h pré-T0 (limitação do raw_temporal)

Este protótipo serve para validar a arquitetura, verificar convergência do ELBO e calibrar hiperparâmetros antes de ter o timegrid completo (5 min, internação completa) disponível.

O ponto de entrada de dados é `dataset/feature_engineering.py:build_temporal_tensor()`, que já implementa ffill + mediana + z-score para o tensor temporal do raw_temporal.

---



| Arquivo | Papel |
|---|---|
| `configs/target_trial_config.yaml` | Critérios de elegibilidade, T0, features, desfechos, estimadores causais |
| `configs/config.yaml` | Configuração geral do pipeline (cohort, preprocessing, variáveis) |
| `configs/paths.yaml` | Paths de dados e outputs |
| `scripts/target_trial/build_target_trial.py` | Orquestra a construção do `trial_dataset.parquet` |
| `dataset/build_cohort.py` | Coorte + T0 + raw_temporal (DuckDB sobre timegrid) |
| `dataset/feature_engineering.py` | Agregações estáticas e tensor temporal |
| `dataset/preprocess.py` | Preprocessing completo (features + tensor + outcomes) |
| `dataset/_archive/legacy_pipeline/src/impute.py` | Imputação do timegrid de 5 min (ffill + defaults) |
| `explore_parquets.ipynb` | Notebook para inspecionar os 10 parquets disponíveis localmente |
