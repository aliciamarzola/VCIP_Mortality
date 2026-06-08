# EVALUATION.md — Avaliação do VCIP no Notebook `VCIP-Transfusion.ipynb`

Documento técnico sobre como o VCIP é avaliado nas Seções 18–22E do notebook, quais métricas foram implementadas e quais não puderam ser reproduzidas por limitações dos dados observacionais.

---

## Índice

1. [Estrutura Geral da Avaliação](#1-estrutura-geral-da-avaliação)
2. [GRP — Ground Truth Ranking Position (Seção 19)](#2-grp--ground-truth-ranking-position-seção-19)
3. [Enumeração Exaustiva de Sequências (Seção 20A)](#3-enumeração-exaustiva-de-sequências-seção-20a)
4. [Validação Cruzada VCIP × AIPW (Seções 20B/C, 22D)](#4-validação-cruzada-vcip--aipw-seções-20bc-22d)
5. [Perfil Clínico por Recomendação (Seções 21, 22E)](#5-perfil-clínico-por-recomendação-seções-21-22e)
6. [Avaliação tau=1: ELBO Diff como Escore Causal (Seções 22C–E)](#6-avaliação-tau1-elbo-diff-como-escore-causal-seções-22ce)
7. [Métricas do Paper Não Implementadas](#7-métricas-do-paper-não-implementadas)
8. [O que Seria Necessário para Reproduzir o Paper Completo](#8-o-que-seria-necessário-para-reproduzir-o-paper-completo)
9. [Resumo Comparativo](#9-resumo-comparativo)

---

## 1. Estrutura Geral da Avaliação

O notebook contém dois modelos treinados com configurações distintas:

| Modelo | Seções | tau | Avaliação | Status |
|---|---|---|---|---|
| **tau=4** (baseline) | 17–21 | 4 | GRP, enumeração 2^τ, perfil clínico | ⚠ Estruturalmente quebrado — recomenda â*=[0,0,0,0] para todos os pacientes |
| **tau=1** (reformulação) | 22A–22E | 1 | ELBO diff, validação AIPW, características clínicas | ✓ Funcionalmente correto |

O modelo tau=4 é mantido no notebook como referência e para documentar o diagnóstico do problema estrutural (ver `TRAINING.md` Seção 5.0). A avaliação clinicamente significativa é feita no tau=1.

A avaliação usa os splits **val + test combinados** nas Seções 22C–E (maior n para análise clínica). A Seção 19 (GRP) usa apenas **test** para garantir avaliação sem viés do early stopping.

---

## 2. GRP — Ground Truth Ranking Position (Seção 19)

### O que mede

O GRP avalia se o modelo reconhece a decisão real do médico (observada no MIMIC-IV) como melhor do que sequências alternativas geradas aleatoriamente.

```
GRP = (k + 1 − ξ) / k

onde:
  k  = número de sequências candidatas (excluindo a real)
  ξ  = posição da sequência real no ranking do ELBO (1 = melhor)

GRP = 1.0 → modelo sempre identifica a sequência real como a melhor
GRP = 0.5 → ranking aleatório (baseline sem aprendizado)
GRP < 0.5 → modelo penaliza a sequência real
```

### Implementação

Para o modelo tau=4 (Seção 19):

- **k = 100** candidatas por paciente: 50 aleatórias + 50 perturbações da sequência real (flip de um bit aleatório)
- **Split**: apenas `test_f` — o modelo nunca viu esses pacientes; `val_f` foi usado exclusivamente para early stopping, não para gradientes
- Cada candidata é avaliada pelo ELBO usando `calculate_elbo(..., optimize_a=True)` com `num_samples=10` amostras Monte Carlo

### Interpretação

- GRP próximo de 1.0 indica que o modelo aprendeu a reconhecer padrões de intervenção clinicamente apropriados
- GRP ≈ 0.5 indica que o ELBO não discrimina sequências — o modelo não aprendeu nada útil sobre quando transfundir
- Para o modelo tau=4 deste notebook: GRP esperado ≈ 0.5 (o modelo aprende â*=[0,0,0,0] sempre, pois 98.5% dos batches de treino têm zero tratamento no target)

### No paper original

No dataset sintético de tumor (quimioterapia/radioterapia), o VCIP atinge **GRP ≈ 0.88** para τ=6, enquanto os baselines (CRN, RMSN, CT, ACTIN) ficam entre 0.40 e 0.60 e degradam com τ crescente. O VCIP é o único método que mantém GRP estável conforme τ aumenta.

---

## 3. Enumeração Exaustiva de Sequências (Seção 20A)

### O que faz

Para tau=4, há 2^τ = **16 sequências binárias possíveis**. Em vez de amostrar candidatas aleatoriamente (como na Seção 19), a Seção 20A enumera todas as 16 sequências de forma determinística e identifica â* = argmin ELBO para cada paciente.

### Implementação

```python
seq_vals_list = list(product([0.0, 1.0], repeat=4))   # 16 tuplas
# Para cada paciente: calcula ELBO de todas as 16 sequências
# â* = sequência com menor ELBO
# elbo_gap = ELBO(â*) − min(todos ELBOs) = 0 por construção
#             OU ELBO(â*) − ELBO(sequência real)
```

- **Split**: test_f + val_f combinados — o objetivo aqui é análise clínica, não medir qualidade do modelo; train_f excluído porque o modelo otimizou diretamente esses exemplos
- `vcip_recommends_transfusion = 1` se â* contém ao menos um slot com A=1
- `elbo_gap` = diferença entre ELBO da sequência real e ELBO ótimo (quanto a decisão real piora em relação a â*)

### Limitação para tau=1

Com tau=1, existem exatamente 2 sequências: `[0]` e `[1]`. A enumeração collapsa para uma comparação direta de ELBO. As Seções 22C–E adotam esse caso diretamente.

---

## 4. Validação Cruzada VCIP × AIPW (Seções 20B/C, 22D)

### A hipótese de consistência

Os grupos fenotípicos B1, B2, M1, M2, M3 foram derivados pelo pipeline AIPW a partir de features **estáticas** pré-T0 (médias, medianas, slopes). O VCIP foi treinado nas séries **temporais** dos mesmos pacientes, sem nunca ver os rótulos de grupo ou os estimandos AIPW.

Se ambos os métodos capturam o mesmo fenômeno causal subjacente — a resposta fisiológica heterogênea à transfusão — então:

> **Grupos B1/B2** (AIPW: ATE ≈ −0.31 a −0.35, transfusão reduz mortalidade) → VCIP deve recomendar transfusão com **maior frequência**
>
> **Grupos M1/M2/M3** (AIPW: ATE ≈ +0.24 a +0.45, transfusão aumenta mortalidade) → VCIP deve recomendar transfusão com **menor frequência**

Esta é uma **validação cruzada genuína**: os dois métodos são independentes em dados e treinamento.

### Teste estatístico implementado

```python
ct = pd.crosstab(df['benefit_group'], df['vcip_rec_transfusion'])
chi2, p, _, _ = chi2_contingency(ct)
# H0: recomendação VCIP independe de grupo AIPW (B vs M)
# H1: B1/B2 recebem mais recomendações de transfusão que M1/M2/M3
```

Um p < 0.05 seria evidência de consistência entre os dois métodos causais. Um p ≥ 0.05 pode indicar: (a) os métodos capturam aspectos diferentes do fenômeno; (b) o VCIP não convergiu para uma política informativa; ou (c) amostra insuficiente.

### Visualizações

- Taxa de recomendação VCIP por grupo AIPW (B1, B2, M1, M2, M3, None) — barras com n por grupo
- ELBO diff por grupo (violinplot) — `elbo_diff = ELBO([1]) − ELBO([0])`, negativo = modelo prefere transfundir
- Comparação agregada B1+B2 vs M1+M2+M3: taxa de recomendação e concordância com a decisão real do médico

---

## 5. Perfil Clínico por Recomendação (Seções 21, 22E)

### O que analisa

Caracterização dos pacientes segundo a decisão do VCIP:

| Grupo | Definição |
|---|---|
| **VCIP=1** | Modelo recomenda transfundir (ELBO([1]) < ELBO([0]) no tau=1; â* ≥ 1 slot no tau=4) |
| **VCIP=0** | Modelo recomenda não transfundir |

### Features analisadas (Seção 22E)

**Vitais clínicos** — último valor observado antes de T0 (bin mais recente no grid horário):

| Feature | Unidade | Por que relevante |
|---|---|---|
| Hemoglobina | g/dL | Gatilho primário da decisão de transfundir (limiar da coorte: Hb ≤ 8) |
| Lactato | mmol/L | Indicador de hipoperfusão tissular — Hb baixa + lactato alto = indicação forte |
| PAM | mmHg | Estabilidade hemodinâmica — instável contraindica transfusão agressiva |
| SOFA | score | Gravidade global — alto SOFA correlaciona com malefício no AIPW |
| Creatinina | mg/dL | Disfunção renal — associada aos grupos de malefício |
| SpO₂ | % | Saturação de oxigênio — separa anemia de hipoxemia de outra causa |
| Frequência cardíaca | bpm | Taquicardia compensatória à anemia |
| FR | rpm | Frequência respiratória |

**Demográficos**: idade (desnormalizada via `scaler_V`), sexo, BMI.

**Clínicos binários**: vasopressor em uso, ventilação mecânica, taxa de mortalidade, taxa de transfusão real.

### Testes estatísticos

- **Contínuos**: t-test de Welch (variâncias não assumidas iguais)
- **Binários**: teste χ² de independência
- Marcação de `*` para p < 0.05

### ELBO como medida de certeza do modelo (Seção 21)

O ELBO ótimo do paciente também indica confiança do modelo:

- **ELBO baixo**: existe uma sequência que claramente leva ao desfecho observado — paciente "interpretável" pelo modelo
- **ELBO alto**: nenhuma sequência melhora muito o score — pode indicar paciente atípico, história insuficiente, ou caso onde a transfusão é irrelevante para o desfecho

---

## 6. Avaliação tau=1: ELBO Diff como Escore Causal (Seções 22C–E)

Com tau=1, a avaliação se simplifica para uma comparação binária por paciente:

```
elbo_diff = ELBO([1]) − ELBO([0])

elbo_diff < 0  →  VCIP prefere transfundir (maior plausibilidade causal de atingir Y_target)
elbo_diff > 0  →  VCIP prefere não transfundir
elbo_diff ≈ 0  →  modelo indiferente — não discrimina as duas decisões
```

### Por que elbo_diff é mais informativo que a decisão binária

A decisão binária (`vcip_rec_transfusion ∈ {0, 1}`) captura apenas a direção. O `elbo_diff` captura também a **magnitude da preferência** — um diff de −5.0 indica muito mais confiança do que −0.1. A distribuição de `elbo_diff` por grupo AIPW é, portanto, mais rica que a taxa de recomendação.

### GRP no contexto tau=1

Com k=1 (apenas 2 sequências possíveis), o GRP assume apenas dois valores:

```
GRP = 1.0  →  modelo identificou a sequência real como a melhor das duas  (ξ=1)
GRP = 0.5  →  modelo identificou a sequência errada como a melhor        (ξ=2)
```

O GRP tau=1 é equivalente à acurácia de classificação binária: `GRP_médio = acurácia`. Baseline aleatório permanece 0.5.

---

## 7. Métricas do Paper Não Implementadas

### 7.1 RCS — Ranking Correlation Score

**O que é**: correlação de Spearman entre o ranking das sequências candidatas ordenadas pelo ELBO do modelo e o ranking verdadeiro (baseado no desfecho real que cada sequência produziria).

```
RCS = Spearman(ranking_ELBO, ranking_true_loss)
```

**Por que não é calculável no MIMIC-IV**:

O ranking verdadeiro requer saber o desfecho contrafactual de cada sequência candidata — o que teria acontecido ao paciente *se* o médico tivesse seguido a sequência `â` em vez da sequência real. Esses contrafactuais são, por definição, não observáveis em dados observacionais.

No paper original, o dataset de tumor possui um **simulador fisiológico diferencial** baseado em equações PK/PD (farmacocinética/farmacodinâmica) que resolve o estado do tumor para qualquer sequência de doses de quimioterapia e radioterapia. A função `simulate_output_after_actions(H_t, â)` retorna o desfecho simulado para a sequência `â`.

No MIMIC-IV, esse simulador não existe. A implementação atual retorna zeros para qualquer sequência:

```python
def simulate_output_after_actions(self, H_t, actions, scaling_params=None):
    return np.zeros((actions.shape[0], 1), dtype=np.float32)
```

Com `true_loss` idêntico (zero) para todas as candidatas, o ranking verdadeiro é constante → Spearman indefinido → **RCS = NaN**.

**Benchmark do paper**: RCS > 0.70 para τ=6, enquanto baselines ficam entre 0.30 e 0.50.

### 7.2 Avaliação Multi-τ

**O que é**: o paper avalia GRP e RCS para τ ∈ {2, 4, 6, 8}, demonstrando que o VCIP mantém performance estável ou melhora com τ crescente — ao contrário de CRN, RMSN e CT, que degradam significativamente.

**Por que não é feito aqui**: os dados do `raw_temporal` cobrem apenas as 48h **pré-T0**. Com resolução horária e τ em slots de 1h:
- O tratamento real está sempre no slot 47 (último antes de T0)
- Para τ ≥ 2, o target precisaria se estender para além de T0 — região sem dados
- A reformulação tau=1 foi necessária justamente porque τ > 1 não tem suporte nos dados disponíveis

Para avaliar múltiplos τ seria necessário o timegrid de 5 min cobrindo a internação completa (incluindo o período pós-transfusão).

### 7.3 Comparação com Baselines (CRN, RMSN, CT, ACTIN)

**O que é**: o paper compara o VCIP contra quatro baselines de planejamento de intervenção temporal:
- **CRN** (Counterfactual Recurrent Network)
- **RMSN** (Recurrent Marginal Structural Network)
- **CT** (Causal Transformer)
- **ACTIN** (Autoregressive Causal Temporal Intervention Network)

**Por que não é feito aqui**: todos esses baselines foram implementados e avaliados no dataset sintético de tumor com simulador. Sem RCS e sem GRP confiável (modelo tau=4 quebrado), a comparação não teria base. Adicionalmente, esses modelos precisariam ser retreinados do zero para o contexto de transfusão — o que está fora do escopo deste protótipo.

### 7.4 Política de Otimização com Gradiente (Algorithm 1 completo)

**O que é**: a inferência do VCIP otimiza logits contínuos `ã ∈ ℝ^τ` via gradient descent, convertendo para binário com o **Straight-Through Estimator**:

```python
# Logit contínuo, derivável
a_logit = torch.zeros(tau, requires_grad=True)
optimizer = torch.optim.Adam([a_logit], lr=lr_a)

for _ in range(opt_epochs):
    a_soft    = torch.sigmoid(a_logit)          # contínuo em [0,1]
    a_binary  = (a_soft > 0.5).float()          # discretiza
    a_ste     = a_soft + (a_binary - a_soft).detach()  # STE: gradiente passa pela binarização
    elbo, _, _ = model.calculate_elbo(H_t, Y_target, a_ste, optimize_a=True)
    elbo.backward()
    optimizer.step()
```

**Status no notebook**: implementado na Seção 18 (tau=4) mas produz sempre â*=[0,0,0,0] por causa do problema estrutural do treino. Para tau=1, o espaço de busca é tão pequeno (2 sequências) que a enumeração exaustiva é preferível e foi usada nas Seções 22C–E.

### 7.5 Avaliação de Benefício Absoluto (Potential Outcome)

**O que é**: uma extensão natural do GRP seria medir se os pacientes para os quais o VCIP recomenda transfusão realmente têm menores taxas de mortalidade — comparando `Y(1)` vs `Y(0)` estimados. O AIPW já faz isso via os estimandos de efeito individual.

**Por que não implementado diretamente**: o desfecho real `Y` é observado para a decisão que foi tomada, não para a contrafactual. O `vcip_lite_individual_counterfactuals.parquet` disponível localmente contém estimativas `Y(0)` e `Y(1)` por paciente (modelo simplificado), mas integrá-las à avaliação do VCIP requereria cuidado metodológico para evitar circularidade (os grupos B1/B2/M1/M2/M3 foram derivados desse mesmo arquivo).

---

## 8. O que Seria Necessário para Reproduzir o Paper Completo

| Requisito | O que existe agora | O que falta |
|---|---|---|
| **Simulador fisiológico** | `simulate_output_after_actions` retorna zeros | Modelo de outcome treinado separadamente (ex: LSTM de mortalidade) para gerar desfechos contrafactuais |
| **Timegrid completo** | 48h pré-T0, resolução 1h | Internação completa (pré + pós-T0), resolução 5 min |
| **Múltiplos τ** | Apenas tau=1 (funcional) e tau=4 (quebrado) | τ ∈ {2, 4, 8, 12, 24} slots com dados pós-T0 |
| **Baselines** | Nenhum | CRN, RMSN, CT, ACTIN treinados no mesmo dataset |
| **Desfecho fisiológico dinâmico** | `mortality_anytime` (estático, binário) | `[Hb, PAM, Lactato, SpO₂]` em t+τ (contínuo, temporal) |
| **Variáveis adicionais** | 11 vitais do raw_temporal | GCS, INR, bilirrubinas, Na, K, gasometria, dose específica de vasopressor, cristalóides, FFP |

---

## 9. Resumo Comparativo

### O que o paper mede vs. o que está implementado

| Métrica | Paper original | tau=4 (Seções 18–21) | tau=1 (Seções 22C–E) |
|---|---|---|---|
| **GRP** | ✓ GRP ≈ 0.88 (τ=6) | ✓ Implementado (esperado ≈ 0.5 por bug estrutural) | ✓ GRP = acurácia binária |
| **RCS** | ✓ RCS > 0.70 (τ=6) | ✗ NaN (sem simulador contrafactual) | ✗ NaN |
| **Multi-τ** | ✓ τ ∈ {2,4,6,8} | ✗ Apenas τ=4 | ✗ Apenas τ=1 |
| **Baselines** | ✓ CRN, RMSN, CT, ACTIN | ✗ | ✗ |
| **Validação causal externa** | ✗ (dataset sintético) | ✓ Validação AIPW (χ²) | ✓ Validação AIPW (χ², elbo_diff por grupo) |
| **Características clínicas** | ✗ | ✓ Seção 21 (age_z, sex, bmi_z, grupo) | ✓ Seção 22E (vitais desnormalizados, t-test, χ²) |
| **ELBO diff como escore** | ✗ | ✗ (tau=4 produz diff≈0 para todos) | ✓ elbo_diff = ELBO([1])−ELBO([0]) por paciente |

### Contribuição específica deste notebook

A ausência de simulador fisiológico e dados pós-T0 impede reproduzir o RCS e a avaliação multi-τ do paper. Em compensação, este notebook introduz uma **validação que o paper original não tem**: a consistência cruzada entre o VCIP e os estimandos AIPW. O AIPW nunca viu as séries temporais; o VCIP nunca viu os estimandos. Se os dois métodos concordam em quais grupos se beneficiam da transfusão, isso é evidência convergente e independente do fenômeno causal — mais convincente do que qualquer métrica calculada sobre um único pipeline.
