# MODEL_PARAMETERS.md — VCIP para Transfusão MIMIC-IV

> Guia de referência para os parâmetros do modelo VCIP implementado em `VCIP-ICML/`.  
> Cobre o que cada parâmetro faz mecanisticamente, como ele afeta o treinamento e o que esperar de bons valores para este dataset (1.484 pacientes, 48 bins de 1h, desfecho binário de mortalidade).

---

## 1. Dimensões do Dataset (`cfg.dataset`)

Esses parâmetros definem a **forma dos tensores** e o **modo de predição**. São fixos para um dado dataset — alterar exige reprocessar os dados.

| Parâmetro | Valor atual | O que define | Impacto de mudar |
|---|---|---|---|
| `input_size` | 11 | Nº de sinais vitais/labs em X (Hb, Lact, Creat, Plaq, FC, PAM, FR, SpO₂, SOFA, vaso, vent) | Aumentar → mais informação clínica, mais parâmetros nos LSTMs |
| `treatment_size` | 1 | Dimensão de A (transfusão: 0/1) | 1 para binário, >1 para múltiplos tratamentos simultâneos |
| `output_size` | 1 | Dimensão de Y (mortality_anytime) | Aumentar para multi-desfecho (ex: [Hb, PAM, Lactato]) |
| `static_size` | 8 | Nº de features estáticas em V (age, sex, bmi, B1–M3) | Aumentar→ encoder de histórico vê mais contexto por paciente |
| `max_seq_length` | 48 | Comprimento máximo da sequência (bins de 1h) | Define o limite superior da janela pré-T0 |
| `min_seq_length` | 25 | Comprimento mínimo aceito após imputação | Pacientes com menos bins são descartados |
| `projection_horizon` τ | 4 | Quantos passos à frente o VCIP otimiza | Ver seção 5 para análise detalhada |
| `val_batch_size` | 64 | Batch de validação (lido por `AuxiliaryModel`) | Menor → mais preciso, mais lento |

### `autoregressive` (True)

Quando `True`, o Y do passo anterior (`prev_outputs`) é concatenado como input do LSTM de histórico a cada timestep. Isso cria dependência temporal explícita entre desfechos consecutivos.

- **Por que True:** mortalidade tem forte autocorrelação temporal — o estado de risco do paciente na hora t depende do que aconteceu na hora t−1.
- **Desligar (False):** elimina essa dependência; útil se Y for esparso ou ruidoso.

### `predict_X` (True)

Quando `True`, os vitais X são incluídos como input adicional no LSTM de histórico (além das features estáticas e do tratamento anterior). A sigla vem de "predict vitals as context".

- **Com `lambda_X = 0`** (configuração atual): X entra como input do LSTM mas **não** há penalidade de reconstrução de X no ELBO. O modelo usa os vitais para codificar o estado mas não é forçado a regenerá-los.
- **Com `lambda_X > 0`**: adiciona reconstrução de vitais como tarefa auxiliar — pode regularizar o espaço latente Z, mas aumenta o custo computacional.
- **Efeito prático no input size:** LSTM history input = X(11) + V(8) + A(1) + Y(1) = **21**

### `output_mode` (`bernoulli`)

Define o decoder do desfecho Y:

| Valor | Decoder | Perda | Quando usar |
|---|---|---|---|
| `bernoulli` | `sigmoid(logit)` | BCE | Y binário (morte: 0/1) ← **caso atual** |
| `gaussian` | N(μ, σ²) | MSE | Y contínuo (ex: Hb em g/dL) |

> **Atenção:** a mudança para `bernoulli` exige patch em `vae_model.py` (já aplicado).

### `treatment_mode` (`multilabel`)

| Valor | Prior de intervenção | Quando usar |
|---|---|---|
| `multilabel` | Bernoulli independente por dimensão | A binário (0/1) ← **caso atual** |
| `beta` | Distribuição Beta | A contínuo em [0,1] (ex: dose normalizada) |

---

## 2. Arquitetura do Modelo (`cfg.model`)

### 2.1 Espaço Latente

#### `z_dim` = 16

**O que é:** dimensão do vetor de estado latente Z em cada timestep s. Z_s ∈ ℝ^`z_dim` representa o "estado fisiológico oculto" do paciente — o que o médico não vê diretamente nos monitores, mas que o modelo infere.

**Mecanismo:** todo o grafo causal gira em torno de Z:
```
H_t → Z_t → Z_{t+1} → ... → Z_{t+τ} → Y_{t+τ}
               ↑          ↑
              a_t        a_{t+τ-1}
```

**Como escolher:**
- Muito pequeno (ex: 4–8): o espaço latente não consegue representar a heterogeneidade fisiológica → underfitting
- Muito grande (ex: 64+): o espaço latente memoriza pacientes individuais → overfitting, KL diverge
- **Bom range para 1.484 pacientes:** 8–32. Começar com 16, monitorar o KL loss.

**Diagnóstico:** se `kl_loss → 0` durante o treino, o modelo de inferência colapsa no prior — `z_dim` pode estar grande demais ou `lambda_kl` alto demais.

#### `model.lr` = 1e-3

Learning rate do otimizador Adam do `VAEModel` principal (generative + inference models). Separado do `exp.lr` que é usado pelo `AuxiliaryModel`.

---

### 2.2 Modelo de Inferência q_φ (`cfg.model.inference`)

O modelo de inferência é o **encoder variacional**: vê Y_target durante o treino e infere a distribuição posterior q_φ(Z_s | H_t, â, Y_target). Durante a inferência (Seção 18), apenas p_θ é usado.

#### `hidden_dim` = 32

Tamanho do hidden state do LSTM de inferência (`lstm`). Esse LSTM processa sequencialmente os estados latentes Z_0...Z_{τ} condicionado no contexto de intervenção e em Y_target.

**Guideline:**
- ≥ `z_dim` (senão gargalo desnecessário)
- Para 1.484 pacientes: 32–64 é adequado. 128+ tende a overfitting.

#### `num_layers` = 2

Profundidade do LSTM de inferência. LSTMs com 2 camadas capturam padrões temporais de mais alta ordem (ex: tendência de queda de Hb modulada pela instabilidade hemodinâmica).

**Atenção:** com `num_layers=1`, o PyTorch emite `UserWarning: dropout option adds dropout after all but last recurrent layer` — é inofensivo, mas indica que o dropout não é aplicado. Use `num_layers ≥ 2` para que o dropout tenha efeito.

#### `hiddens_F_mu` e `hiddens_F_logvar` = [32]

MLPs que mapeiam o hidden state do LSTM → μ e log σ² da posterior q_φ(Z_s).

- `[32]`: uma camada oculta de 32 neurônios → Linear(hidden_dim, 32) → ELU → Linear(32, z_dim)
- `[-1]`: projeção direta (linear), sem camada oculta

**Bom valor:** `[32]` ou `[64]`. Usar `-1` (sem oculta) é válido se o LSTM já for expressivo o suficiente. Múltiplas camadas (`[64, 32]`) raramente ajudam para datasets desta escala.

#### `predict_y_history` = [32]

MLP auxiliar que prediz Y a partir do estado latente + tratamento: Z_s + A_s → Ŷ. Essa rede é treinada conjuntamente pelo termo `lambda_hy`, funcionando como regularizador: força Z a ser preditivo de Y mesmo sem ver Y_target diretamente.

- **Efeito prático:** reduz o colapso do espaço latente em regiões que não codificam informação sobre desfechos.
- **Desligar (`lambda_hy = 0`):** Z pode se tornar menos interpretável, mas o ELBO principal não muda.

#### `inference.do` = True

Liga o dropout no LSTM de inferência durante o treino. Como `dropout` global é 0.2 e `num_layers = 2`, o dropout é aplicado entre a 1ª e a 2ª camada do LSTM.

---

### 2.3 Modelo Generativo p_θ (`cfg.model.generative`)

O modelo generativo aprende a **dinâmica fisiológica**: como intervenções transformam estados latentes e como estados latentes geram desfechos observáveis.

#### `hidden_dim` = 32

Hidden size do LSTM generativo principal (`lstm`), que processa a sequência de estados latentes Z_t...Z_{t+τ} integrada com os contextos de intervenção. Input do LSTM: Z_s (z_dim) + contexto de ação (treatment_hidden_dim) = 16 + 16 = 32.

#### `treatment_hidden_dim` = 16

Tamanho do hidden state dos LSTMs de ação (`action_encoder` e `reverse_action_encoder`). Esses LSTMs codificam a sequência â_{t,τ} — tanto na direção forward quanto backward — em representações contextuais por timestep.

**Por que separado de `hidden_dim`:** as intervenções de transfusão têm dinâmica mais simples (binária, esparsa) do que o estado fisiológico completo. Um encoder menor (16 vs. 32) reduz parâmetros sem perda de expressividade.

**Relação importante:** `inference.input_size` depende de `treatment_hidden_dim` — se você mudar um, o outro deve ser compatível.

#### `hiddens_F_mu` e `hiddens_F_logvar` = [32] (generativo)

MLPs para a distribuição de transição p_θ(Z_{s+1} | Z_s, ctx_s): o "prior" do modelo generativo. Input: hidden state do LSTM (hidden_dim) + contexto de ação (treatment_hidden_dim) = 64.

**O que o modelo aprende aqui:** como 1 unidade de PRBC transforma o estado latente nas próximas 4 horas, dependendo do contexto clínico (vasopressores, ventilação, Hb basal).

#### `hiddens_decoder` = [32]

MLP que mapeia Z_{t+τ} → Y_{t+τ} (desfecho final). Com `output_mode=bernoulli`, a saída é um logit que passa por sigmoid → probabilidade de morte.

**Nota:** o mesmo parâmetro `hiddens_decoder` controla **4 decoders** no código:
- `decoder`: Z → logit Y (Bernoulli, sem contexto de ação)
- `decoder_p`: Z → [μ_Y, σ_Y] (Gaussiana)
- `decoder_pa`: Z + ctx_a → [μ_Y, σ_Y] (com ação)
- `decoder_x`: Z → X reconstruído (quando `predict_X=True`)

#### `hidden_action_decoder` = [32]

MLP para p_θ(â | H_t): probabilidade da **sequência completa** de intervenções dado o histórico. Input: ctx_action (treatment_hidden_dim) + Z_s (z_dim) = 32.

Papel na g-formula: `log p_θ(â|H_t)` é o denominador da correção de confundimento.

#### `hidden_action_decoder_step` = [32]

MLP para p_θ(a_s | Z_s): probabilidade de **um passo** de intervenção dado o estado latente atual. Input: Z_s (z_dim = 16).

Papel na g-formula: `Σ_s log p_θ(a_s|Z_s)` é o numerador. A diferença numerador − denominador captura o viés prescritivo.

---

### 2.4 Modelo Auxiliar / History Encoder (`cfg.model.auxiliary`)

O `AuxiliaryModel` é o **LSTM de histórico** — o componente que lê a trajetória clínica completa (X, A, Y ao longo do tempo) e a comprime em h_t, a "impressão digital" do paciente.

#### `hidden_dim` = 32

Hidden size do LSTM auxiliar. Saída h_t ∈ ℝ^32. Esse vetor é a entrada principal do modelo de inferência q_φ.

**Nota de dimensionamento:** o `InferenceModel` espera `history_dim = config['model']['auxiliary']['hidden_dim']` — os dois devem estar alinhados.

#### `num_layers` = 2

Profundidade do LSTM de histórico. Com 2 camadas, o modelo aprende representações hierárquicas do histórico clínico (ex: camada 1: tendências horárias, camada 2: padrões ao longo de dias).

#### `hiddens_G_y` e `hiddens_G_x` = [32]

MLPs auxiliares para predição de Y e X a partir do estado latente, usadas no treinamento do AuxiliaryModel. Controlam a expressividade dessas predições auxiliares.

---

## 3. Hiperparâmetros de Treinamento (`cfg.exp`)

### 3.1 Otimização básica

| Parâmetro | Valor | O que faz | Bom range |
|---|---|---|---|
| `batch_size` | 128 | Nº de pacientes por batch de treino | 64–256. Maior → gradientes mais estáveis mas menos atualizações por época |
| `val_batch_size` | 64 | Batch de validação | 32–128. Pode ser maior que o de treino (sem gradiente) |
| `epochs` | 150 | Máximo de épocas | 100–300. Com early stopping, raramente chega ao limite |
| `patience` | 25 | Épocas sem melhora no val_loss para parar | 15–40. Muito baixo → para cedo; alto → desperdiça tempo |
| `exp.lr` | 1e-3 | Learning rate do `AuxiliaryModel` (Adam) | 1e-4 a 5e-3. Começar com 1e-3, decair se oscilação |
| `model.lr` | 1e-3 | Learning rate do `VAEModel` (generative + inference) | Idem |
| `weight_decay` | 1e-5 | Regularização L2 (Adam) | 1e-6 a 1e-4. Muito alto → underfitting |
| `dropout` | 0.2 | Dropout entre camadas dos LSTMs | 0.1–0.4. Para N=1038 no treino, 0.2 é conservador mas razoável |

### 3.2 EMA (Exponential Moving Average)

#### `weights_ema` = True / `beta` = 0.999

O EMA mantém uma cópia "suavizada" dos pesos do modelo durante o treino: `θ_ema = 0.999·θ_ema + 0.001·θ`. Os pesos EMA são usados na avaliação e na inferência.

**Por que usar:** redes neurais profundas oscilam entre épocas — os melhores pesos instantâneos muitas vezes são "sorte" em um batch. O EMA estabiliza, dando performance mais consistente e geralmente melhor no teste.

**`beta` (decay):**
- 0.999 → suavização muito forte (memória de ~1000 steps) — boa para treinos longos
- 0.99 → mais reativa (memória de ~100 steps) — melhor se o modelo aprender rápido
- Diminuir para 0.99 se o EMA demorar demais para refletir melhorias reais

### 3.3 Monte Carlo ELBO

#### `num_samples` = 10

O ELBO é estimado por amostragem de Monte Carlo: `num_samples` amostras Z^(i) ∼ q_φ(Z | ·) são geradas via reparameterization trick, e a média das log-verossimilhanças é usada.

**Trade-off:**
- Mais amostras → estimativa mais precisa do ELBO → gradientes mais estáveis → treino mais lento
- Menos amostras → mais ruidoso → pode divergir em fases iniciais

**Valores típicos:** 5–20. No paper original: 10. Reduzir para 5 se memória GPU for um gargalo; aumentar para 20–50 se o ELBO oscilar muito no início.

---

## 4. Lambdas do ELBO — O Coração do Treinamento

O ELBO minimizado no treino (sem otimização de â) é:

```
ELBO = λ_reg · reg_loss
     + λ_kl  · KL(q_φ || p_θ)
     + λ_step · action_loss_step
     + λ_action · predict_action_loss
     + λ_hy · loss_hy
```

Durante a **inferência** (otimização de â*), `loss_hy` é excluído (Y_target futuro não está disponível):

```
ELBO_inf = λ_reg · reg_loss
          + λ_kl  · KL
          + λ_step · action_loss_step
          + λ_action · predict_action_loss
```

### `lambda_Y` / `lambda_reg` = 1.0

**O que é:** peso do termo de reconstrução de Y — a verossimilhança do desfecho dado o estado latente final: `-E[log p_θ(Y_{t+τ} | Z_{t+τ})]`.

Com `output_mode=bernoulli`:
```python
reg_loss = BCE(sigmoid(decoder(Z_{t+τ})), Y_target)
```

**Por que 1.0:** é o objetivo principal — o modelo deve aprender a prever mortalidade. Reduzir prejudica diretamente a capacidade preditiva.

**Diagnóstico:** monitorar `reg_loss` separadamente. Deve cair consistentemente nas primeiras 30–50 épocas. Se estabilizar muito cedo (ex: BCE > 0.65 = predição aleatória para 50% de mortalidade), o modelo não está aprendendo.

### `lambda_kl` = 1.0

**O que é:** peso da KL divergência entre o modelo de inferência q_φ e o modelo generativo p_θ:

```
KL(q_φ(Z_s | H_t, â, Y_target) || p_θ(Z_s | H_t, â))
```

**Por que esse termo existe:** durante o treino, q_φ vê Y_target e tende a "trapacear" — aprender regiões de Z muito específicas que memorizam o desfecho. A KL força q_φ a permanecer próximo de p_θ, que **não** vê Y_target. Isso garante que p_θ (único disponível na inferência) funcione bem.

**Fenômeno clássico — KL collapse:** se `lambda_kl` for muito alto, q_φ ≈ p_θ e o modelo ignora Y_target durante o treino. O ELBO parece bom mas a qualidade das amostras latentes é ruim.

**KL annealing (estratégia recomendada):** começar com `lambda_kl = 0.0` nas primeiras 20–30 épocas (deixar o modelo aprender o sinal de reconstrução primeiro), depois crescer linearmente até 1.0. Implementar no loop de treino.

**Bons valores:** 0.5–1.0 com annealing.

### `lambda_action` = 1.0 ← Correção de Confundimento

**O que é:** peso da g-formula — o ajuste causal central do VCIP:

```
predict_action_loss = Σ_s log p_θ(a_s | Z_s) - log p_θ(â | H_t)
```

- `Σ_s log p_θ(a_s | Z_s)`: quanto cada passo de intervenção é explicado **pelo estado latente** do paciente (o que o estado clínico justificaria)
- `log p_θ(â | H_t)`: quanto a sequência completa é plausível dado o **histórico observado** (o que os médicos do MIMIC fizeram)

**Interpretação:** quando um médico transfundiu pacientes com Hb alta (talvez por guideline institucional, não pelo estado clínico real), a g-formula penaliza isso — discrimina entre "intervenção justificada pelo estado latente" e "intervenção pelo protocolo observacional".

**Por que é crítico para este dataset:** o confundimento por indicação é severo — pacientes mais graves recebem mais transfusões E têm pior prognóstico. Sem a g-formula (`lambda_action = 0`), o modelo aprende que "transfundir = pior desfecho" porque está confundindo gravidade com tratamento.

**Guideline:**
- `lambda_action = 0.0`: sem correção de confundimento (modelo observacional, não causal)
- `lambda_action = 1.0`: correção completa (teoricamente correta sob as assunções causais)
- `lambda_action > 1.0`: penaliza ainda mais vieses, mas pode ser instável

**Diagnóstico:** monitorar `predict_action_loss` separadamente. Deve ser próximo de 0 quando o modelo aprender que as intervenções observadas são bem explicadas pelo estado latente. Se for muito negativo (log-probs muito baixos), revisar se os grupos M1/M2/M3 estão bem representados no treino.

### `lambda_step` = 0.1

**O que é:** peso do termo de consistência passo a passo da política de intervenção:

```
action_loss = Σ_{s=0}^{τ-2} loss(p_θ(a_{s+1} | Z_s), a_{s+1})
```

Diferente de `lambda_action` (que usa o state Z para explicar a ação), esse termo força que a **próxima ação** seja previsível dado o estado latente atual. É uma forma de regularização temporal da política.

**Por que 0.1 (pequeno):** esse termo é auxiliar — ele estabiliza a aprendizado da política mas não deve dominar o ELBO principal. Valores altos (>0.5) tendem a forçar políticas de intervenção triviais (sempre 0 ou sempre 1).

### `lambda_hy` = 0.5

**O que é:** peso da predição auxiliar de Y a partir do histórico (sem ver Y_target):

```
loss_hy = BCE(predict_y_history_net(Z_s, A_s), Y_observado)
```

**Papel:** regulariza o espaço latente para que Z codifique informação sobre desfechos — evita que o espaço latente seja usado apenas para reconstrução de X ou para memorizar sequências de ação.

**Nota:** esse termo é excluído durante a otimização de â* (Seção 18), porque Y_target futuro não está disponível. É apenas para treino.

**Guideline:** 0.3–0.7. Muito alto → Z vira um preditor direto de Y e perde a estrutura latente.

### `lambda_X` = 0.0

**O que é:** peso da reconstrução dos vitais X a partir do estado latente.

**Por que 0.0:** com `predict_X=True`, os vitais entram como input do LSTM mas não há objetivo de reconstrução. Habilitar (`lambda_X > 0`) adiciona a tarefa de reconstruir X_{t+s} a partir de Z_s — pode ajudar o espaço latente a capturar dinâmicas fisiológicas mais ricas, mas aumenta o custo computacional e pode competir com a reconstrução de Y.

**Quando habilitar:** se `reg_loss` (BCE de mortalidade) convergir rápido demais e o GRP na avaliação for baixo, aumentar para 0.1–0.3 para forçar Z a capturar mais estrutura dos vitais.

### `beta_bound` = -10.0

**O que é:** clamp mínimo aplicado ao log-ELBO durante o cálculo:

```python
elbo = torch.clamp(elbo, min=beta_bound)
```

**Por que existe:** nas primeiras épocas, os gradientes do ELBO podem ser extremamente negativos (Z aleatório, decoder não treinado → log-probabilidades → −∞). O clamp previne explosão de gradiente.

**Guideline:** −10 é conservador. Se o ELBO demorar demais para sair desse piso, aumentar para −20 (menos agressivo). Se explodir no início, reduzir para −5.

---

## 5. Horizonte de Intervenção τ (`exp.tau` = 4)

**O que é:** quantos passos à frente o VCIP otimiza a sequência â* = [a_t, a_{t+1}, ..., a_{t+τ-1}].

Com resolução de 1h: τ = 4 significa otimizar as próximas **4 horas** de decisão de transfusão.

**Trade-offs:**

| τ | Interpretação clínica | Desafio computacional |
|---|---|---|
| 1 | Decisão imediata: transfundir agora? | Trivial; pouco benefício sobre modelos estáticos |
| 2–4 | Janela de turno (2–4h) | **Adequado para este dataset** |
| 6–8 | Plantão completo (6–8h) | ELBO degrada; mais difícil otimizar |
| 12+ | Muito longo para dados com resolução 1h | Não recomendado sem dados de 5 min |

**Por que 4 para `raw_temporal`:** com resolução de 1h e apenas 48 timesteps de histórico pré-T0, τ = 4 permite um horizonte clinicamente relevante (4h = decisão de um turno) sem degradar a estimativa do ELBO.

**Para monitorar:** na avaliação (Seção 19), rodar com τ ∈ {2, 4, 6} e verificar se o GRP se mantém acima de 0.7. Se cair abaixo de 0.5 para τ > 4, o modelo está limitado pela resolução horária dos dados.

---

## 6. Parâmetros de Avaliação (`exp.repeats` = 5)

### `repeats` = 5

Número de repetições para estimar GRP e RCS:

- **GRP (Ground Truth Ranking Position):** para cada paciente, gera k=100 sequências candidatas aleatórias + perturbações, ranqueia pelo ELBO, verifica a posição da sequência real.
  - GRP = 1.0 → o modelo sempre identifica corretamente a sequência real
  - GRP ≈ 0.5 → performance aleatória (baseline)
  - **Alvo para este projeto:** GRP > 0.70

- **RCS (Ranking Correlation Score):** Spearman entre ranking predito e ranking verdadeiro (por distância ao desfecho real).
  - RCS ∈ [−1, 1]; alvo: > 0.60

O `repeats = 5` realiza 5 rodadas independentes de amostragem de candidatos para estimar variância do GRP/RCS.

---

## 7. Resumo: O Que Monitorar Durante o Treino

| Métrica | Esperado (início) | Esperado (final) | Problema se... |
|---|---|---|---|
| `train_loss` (ELBO total) | Alto / negativo grande | Decrescente, estabiliza | Explode → reduzir lr ou `lambda_kl` |
| `reg_loss` (BCE mortalidade) | ~0.69 (log(2)) | < 0.60 | Não cai → modelo não aprende Y |
| `kl_loss` (KL divergência) | ~0 | 0.1–2.0 | → 0 = colapso q_φ → p_θ; muito alto = checar `lambda_kl` |
| `predict_action_loss` (g-formula) | Variável | Próximo de 0 | Muito negativo → revisar `lambda_action` |
| `val_loss` | Semelhante ao treino | Desce junto com treino | Diverge do treino → overfitting → aumentar dropout/weight_decay |
| GRP (avaliação final) | — | > 0.70 | < 0.50 = performance aleatória |
| RCS (avaliação final) | — | > 0.60 | Negativo = modelo ranqueia inversamente |

---

## 8. Config Completa de Referência (valores atuais)

```python
cfg = OmegaConf.create({
    'dataset': {
        'input_size': 11,          # vitais: Hb, Lact, Creat, Plaq, FC, PAM, FR, SpO2, SOFA, vaso, vent
        'treatment_size': 1,       # transfused: 0/1
        'output_size': 1,          # mortality_anytime: 0/1
        'static_size': 8,          # age, sex, bmi, B1, B2, M1, M2, M3
        'output_mode': 'bernoulli', # BCE em vez de MSE
        'treatment_mode': 'multilabel',
        'max_seq_length': 48,       # 48h pré-T0 @ 1h/bin
        'min_seq_length': 25,
        'projection_horizon': 4,    # τ = 4h de horizonte
        'autoregressive': True,
        'predict_X': True,          # vitais como input do LSTM
        'val_batch_size': 64,
        'seed': 42,
    },
    'model': {
        'z_dim': 16,                # espaço latente Z
        'lr': 1e-3,                 # lr do VAEModel (vae_model.py:31)
        'inference': {
            'hidden_dim': 32,       # LSTM q_φ
            'num_layers': 2,
            'hiddens_F_mu': [32],
            'hiddens_F_logvar': [32],
            'predict_y_history': [32],
            'do': True,
        },
        'generative': {
            'hidden_dim': 32,       # LSTM p_θ
            'treatment_hidden_dim': 16,  # encoder de ação
            'num_layers': 2,
            'entropy_lambda': 1.0,
            'hiddens_F_mu': [32],
            'hiddens_F_logvar': [32],
            'hiddens_decoder': [32],
            'hidden_action_decoder': [32],
            'hidden_action_decoder_step': [32],
        },
        'auxiliary': {
            'hidden_dim': 32,       # LSTM de histórico h_t
            'num_layers': 2,
            'hiddens_G_y': [32],
            'hiddens_G_x': [32],
        },
    },
    'exp': {
        'batch_size': 128,
        'val_batch_size': 64,
        'epochs': 150,
        'patience': 25,
        'lr': 1e-3,                 # lr do AuxiliaryModel
        'weight_decay': 1e-5,
        'dropout': 0.2,
        'num_samples': 10,          # Monte Carlo ELBO
        'weights_ema': True,
        'beta': 0.999,              # EMA decay
        'lambda_X': 0.0,            # sem reconstrução de vitais
        'lambda_Y': 1.0,            # BCE de mortalidade ← principal
        'lambda_kl': 1.0,           # KL regularização
        'lambda_step': 0.1,         # consistência temporal de ação
        'lambda_action': 1.0,       # g-formula (causalidade)
        'lambda_hy': 0.5,           # predição auxiliar de Y
        'lambda_reg': 1.0,
        'beta_bound': -10.0,        # clamp ELBO (estabilidade)
        'tau': 4,                   # horizonte de otimização â*
        'repeats': 5,               # GRP/RCS
    },
})
```

---

*Última atualização: 2026-05-25 | Modelo: VCIP-ICML adaptado para MIMIC-IV, desfecho Bernoulli, N=1.484 pacientes, τ=4h@1h/bin*
