# TRAINING.md — Pipeline de Preparação e Treinamento do VCIP

Documento técnico sobre como os dados do MIMIC-IV são transformados em tensores e como o modelo VCIP é treinado no notebook `VCIP-Transfusion.ipynb`.

---

## 1. Fonte dos dados

O ponto de partida é o `raw_temporal.parquet`, um subconjunto do MIMIC-IV já processado pelo pipeline legado (`cluster-transfusion`). Ele contém séries temporais de 11 variáveis clínicas para **3.088 pacientes**, com resolução de 1 hora.

O treinamento usa apenas os **1.484 pacientes elegíveis** (Hb ≤ 8 g/dL na janela pré-T0), filtrados via `treatment_assignment.parquet`.

---

## 2. Construção do Tensor Temporal (Seção 14)

### 2.1 Janela temporal

Para cada paciente, é criada uma janela de **48 horas antes de T0** (o momento-índice, onde Hb cruza 8 g/dL pela primeira vez). Cada hora vira um **bin** de 0 a 47.

```
bin 0  = T0 − 48h
bin 47 = T0 − 1h
```

### 2.2 Grid completo e obs_mask

É criado um produto cartesiano `(1.484 pacientes) × (48 bins)`, formando um MultiIndex completo. Isso garante que todo paciente tenha exatamente 48 timesteps no tensor, mesmo que não haja medição em todos os horários.

A `obs_mask` registra onde há medição **original** (antes de qualquer imputação):
- `1` → bin com ao menos uma medição real no raw_temporal
- `0` → bin sem medição (preenchido por forward-fill)

```python
obs_mask : (1484, 48)  # float32
```

Na prática, a cobertura mediana é de **28 bins** por paciente (min=6, max=47), ou seja, ~58% dos timesteps são observados diretamente.

### 2.3 Forward-fill

O tensor é reindexado no grid completo e os NaN são preenchidos com **forward-fill por paciente** — cada bin vazio herda o valor do bin anterior mais recente do mesmo paciente.

Após o forward-fill, **~43% dos valores** ainda são NaN (pacientes sem medição nos bins iniciais da janela). Esses serão preenchidos na Seção 15 com a mediana do treino.

### 2.4 Variáveis do tensor X

```
hemoglobin, lactate, creatinine, platelets,
heart_rate, mbp, resp_rate, spo2,
sofa, any_vasopressor, ventilation_on
```

Shape final: `X_raw : (1484, 48, 11)` — 1.484 pacientes × 48 timesteps × 11 variáveis.

### 2.5 Tensores auxiliares

Além do tensor temporal X, são construídos:

| Tensor | Shape | Conteúdo |
|---|---|---|
| `obs_mask` | (1484, 48) | 1 onde há medição original |
| `V_raw` | (1484, 8) | Features estáticas: age, sex_bin, bmi, B1, B2, M1, M2, M3 |
| `A` | (1484,) | Intervenção em T0: 1 = transfundido |
| `Y` | (1484,) | Desfecho: `mortality_anytime` (Bernoulli 0/1) |

As features estáticas incluem os **grupos de heterogeneidade AIPW** (B1, B2, M1, M2, M3) derivados da análise causal anterior, codificados como one-hot.

---

## 3. Divisão Treino / Val / Teste (Seção 15)

### 3.1 Por subject_id, não por stay_id

O split é feito no nível de **subject** (paciente real), não de *stay* (internação). Isso evita que duas internações do mesmo paciente apareçam em splits diferentes, o que causaria vazamento de informação.

Proporções: **70% treino / 15% validação / 15% teste**, stratificadas por `transfused` (status de transfusão) para garantir representação balanceada em todos os splits.

```
Treino : 1.038 pacientes  | transfundidos: 34.1%  | mortalidade: 43.4%
Val    :   223 pacientes  | transfundidos: 34.1%  | mortalidade: 50.2%
Teste  :   223 pacientes  | transfundidos: 34.1%  | mortalidade: 45.7%
```

### 3.2 Imputação de NaN residual (pós-forward-fill)

Os NaN que permanecem após o forward-fill (bins no início da janela sem nenhuma medição prévia) são preenchidos com a **mediana de cada variável calculada exclusivamente no conjunto de treino**:

```python
train_med_X = np.nanmedian(X[idx_train].reshape(-1, 11), axis=0)
# Aplicada a todos os splits (sem vazar informação do val/teste)
```

### 3.3 Normalização z-score

O z-score é ajustado (**fit**) apenas nos dados de treino e depois aplicado (transform) a todos os splits:

```python
scaler_X = StandardScaler()
scaler_X.fit(X[idx_train].reshape(-1, 11))   # fit NO TREINO APENAS
X = scaler_X.transform(X.reshape(-1, 11)).reshape(1484, 48, 11)
```

Isso garante que o modelo nunca "veja" a distribuição do val/teste durante o treino — o erro mais comum em pipelines de ML com dados temporais.

Para as features estáticas (V), o mesmo procedimento é aplicado apenas às variáveis contínuas (age, bmi); as flags binárias (sex_bin, B1..M3) ficam sem normalização.

### 3.4 Salvamento em disco

Todos os tensores normalizados e os scalers são salvos em `vcip_tensors.pkl`:

```python
{
  'X': (1484, 48, 11),    # tensor temporal z-scored
  'obs_mask': (1484, 48),
  'V': (1484, 8),          # features estáticas normalizadas
  'A': (1484,),            # intervenção
  'Y': (1484,),            # desfecho
  'idx_train': ...,        # índices posicionais
  'idx_val': ...,
  'idx_test': ...,
  'scaler_X': StandardScaler,   # para denormalizar previsões
  'scaler_V': StandardScaler,
  'train_med_X': (11,),         # medianas do treino (para novos pacientes)
}
```

---

## 4. Preparação do DatasetCollection (Seção 16)

### 4.1 Por que não usar MIMIC4MortalityDatasetCollection

A classe `MIMIC4MortalityDatasetCollection` leria os parquets do zero e refaria todo o preprocessing (~34s extras). Pior: seu z-score interno é calculado sobre todos os 1.484 pacientes **antes** do split, causando **data leakage** (a distribuição do val/teste contamina o z-score aplicado ao treino).

As Seções 14-15 já fizeram o preprocessing corretamente. A Seção 16 apenas empacota os tensores do pkl no formato que o VAEModel consome.

### 4.2 Estrutura dos tensores para o CIPDataset

O `CIPDataset` (classe responsável por fatiar os tensores em janelas H_t e targets) espera um dict com estas chaves:

```python
{
  'sequence_lengths':   (n,)          # bins observados − 1 por paciente
  'prev_treatments':    (n, 47, 1)    # A no slot anterior (t-1)
  'current_treatments': (n, 47, 1)    # A no slot atual (t)
  'static_features':    (n, 47, 8)    # V repetida em cada timestep (3D!)
  'prev_outputs':       (n, 47, 1)    # Y no slot anterior
  'outputs':            (n, 47, 1)    # Y corrente (alvo)
  'active_entries':     (n, 47, 1)    # máscara de bins válidos
  'vitals':             (n, 47, 11)   # X (t=1 em diante, pós [1:])
  'current_covariates': (n, 47, 11)   # igual a vitals
}
```

> **Por que `static_features` é 3D?**  
> O `AuxiliaryModel.build_representations()` faz:
> ```python
> x = torch.cat((batch['vitals'], batch['static_features']), dim=-1)  # (B, h, 11+8)
> ```
> Para que o `torch.cat` funcione com `vitals` de shape `(B, h, 11)`, o `static_features`
> precisa ser `(B, h, 8)`. Por isso o vetor V de cada paciente é replicado via `np.tile`
> ao longo da dimensão temporal antes de entrar no dict.

### 4.3 DatasetWrapper e DatasetCollectionWrapper

São dois objetos Python simples (sem herança de PyTorch Dataset):

- **`DatasetWrapper`**: envolve o dict de um split com `.data` e `.__len__`. Implementa `simulate_output_after_actions()` (retorna zeros — não há simulador contrafactual real para dados observacionais).
- **`DatasetCollectionWrapper`**: agrupa os três `DatasetWrapper` (treino/val/teste) e expõe os atributos que o `VAEModel` acessa: `.train_f`, `.val_f`, `.test_f`, `.train_scaling_params`, `.projection_horizon`, `.has_vitals`, `.autoregressive`.

---

## 5. Treinamento tau=4 (Seções 17–19) — Baseline Quebrado

> **Nota**: o treino tau=4 descrito nesta seção produz um modelo que **nunca recomenda transfusão** para nenhum paciente. Isso não é um bug de código — é uma incompatibilidade estrutural entre o design do VCIP e a estrutura dos dados. A Seção 6 documenta a reformulação tau=1 que corrige o problema.

### 5.0 Por que o tau=4 não funciona neste dataset

O `CIPDataset` sorteia `history_length h ∈ [20, 42]` e define:

```
H_t    = current_treatments[0 : h]       ← histórico
target = current_treatments[h : h+tau]   ← onde o tratamento deveria estar
```

O tratamento (`A=1`) está na posição **45** de `current_treatments` (último bin observado, ≈bin 46 do tensor original). O target só inclui essa posição quando `h = 42` — **1 de 23 valores possíveis**:

| h | target window | tratamento em target? |
|---|---|---|
| 42 | [42:46] | ✓ posição 45 está dentro |
| 41 | [41:45] | ✗ |
| … | … | ✗ |

Resultado: `P(A≠0 no target) = 4,3% × 34,1% tratados = **1,5%** das amostras`. Em 98,5% dos batches, `target['current_treatments'] = zeros`. O modelo aprende que â* = `[0,0,0,0]` é a resposta correta — o que matematicamente é verdade dado o que viu, mas clinicamente significa "nunca recomendar transfusão".

**Causa raiz**: o `raw_temporal` cobre apenas as 48h **pré-T0**. O VCIP foi projetado para dados onde o tratamento ocorre *durante* o período de observação e o modelo aprende "se transfundiu no slot t, o desfecho mudou assim". Com todos os dados antes da decisão, o modelo nunca vê a dinâmica pós-intervenção.

---

### 5.1 Arquitetura do loop (PyTorch Lightning)

O `VAEModel` é uma `pl.LightningModule`. O `pl.Trainer` orquestra o loop de épocas e chama os métodos do modelo:

```
trainer.fit(model)
  └─ model.train_dataloader()    →  CIPDataset(train_f.data) + get_dataloader()
  └─ model.val_dataloader()      →  CIPDataset(val_f.data)   + get_dataloader()
  └─ model.training_step(batch)  →  calcula ELBO, backprop manual
  └─ model.validation_step(batch)→  calcula ELBO, sem gradiente
  └─ model.on_*_epoch_end()      →  imprime losses médias
```

O modelo usa `automatic_optimization=False` (otimização manual) — em cada passo de treino, ele próprio chama `opt.zero_grad()`, `self.manual_backward(elbo)` e `opt.step()`.

### 5.2 CIPDataset e janelamento

Para cada época, o `CIPDataset` amostra aleatoriamente vários **comprimentos de histórico** `h` no intervalo [20, 43]. Para cada comprimento, todos os pacientes do split são processados.

Para um dado `h` e um paciente, a amostra gerada é:
```
H_t  = dados[:h]       # histórico de comprimento h (H_t no paper)
target = dados[h:h+τ]  # próximos τ=4 timesteps (alvo)
```

Isso é equivalente a **data augmentation temporal**: o modelo vê cada paciente em múltiplos pontos da sua trajetória por época, sem repetir exatamente o mesmo corte.

### 5.3 O que o ELBO calcula em cada batch

O ELBO (Evidence Lower Bound) é calculado pela função `calculate_elbo()` em `vae_model.py`. Para cada passo `s` de 0 a τ−1:

1. **Encoder de inferência q_φ** recebe o estado latente atual, o contexto da ação e o desfecho alvo (Y_target) → produz `(q_mu, q_logvar)` da posterior Z_s
2. **Modelo generativo p_θ** recebe o estado latente atual e o contexto da ação → produz `(p_mu, p_logvar)` do prior Z_s
3. `num_samples=10` amostras Monte Carlo são tiradas da posterior para estabilizar os gradientes
4. No último passo τ−1, o decodificador prediz Y_{t+τ} a partir do estado latente final

A loss final é a soma ponderada:

```
ELBO = λ_reg · reg_loss          # BCE(Y_predito, Y_target) — reconstrução
     + λ_kl  · KL_loss           # KL(q_φ || p_θ) — regularização do espaço latente
     + λ_step · action_loss      # p_θ(a_{s+1} | Z_s) — coerência das ações
     + λ_action · g_formula      # g-formula de Robins (correção de confundimento)
     + λ_hy · loss_hy            # predição auxiliar de Y dado só o histórico
```

Valores padrão das lambdas:

| Lambda | Valor | Papel |
|---|---|---|
| `λ_reg` | 1.0 | Peso da reconstrução do desfecho |
| `λ_kl` | 1.0 | Regularização do espaço latente |
| `λ_step` | 0.1 | Transição de estado passo a passo |
| `λ_action` | 1.0 | **G-formula**: corrige confundimento por indicação |
| `λ_hy` | 0.5 | Predição auxiliar (encoder de histórico) |

> O `loss_hy` é calculado apenas no modo treino (`optimize_a=False`). Durante a inferência (otimização de â*), ele é excluído porque o Y futuro não está disponível.

### 5.4 G-formula e correção de confundimento

O `λ_action` é o parâmetro mais crítico do VCIP. Ele implementa a **g-formula de Robins** — a correção que permite estimar o efeito causal de uma sequência de intervenções a partir de dados observacionais.

O confundimento por indicação é severo nesta coorte: pacientes mais anêmicos e hemodinamicamente instáveis são mais transfundidos, mas também têm pior prognóstico independentemente da transfusão. A g-formula mede o quanto a política dos médicos (baseada no estado observável) diverge do que o estado fisiológico latente por si só justificaria, e penaliza o modelo quando essa divergência não é contabilizada.

```python
action_loss = p_θ(a_{s+1} | Z_s)           # passo a passo: dado o estado latente
predict_action_loss = p_θ(â_{t,τ} | H_t)   # sequência: dado o histórico observável

g_formula = Σ_s action_loss - predict_action_loss
```

### 5.5 Otimizador e callbacks

| Componente | Configuração |
|---|---|
| Otimizador | Adam, lr=1e-3, weight_decay=1e-5 |
| Épocas máx | 150 |
| Early stopping | `val_loss`, paciência=25 épocas |
| Checkpoints | Top-3 por `val_loss`, em `my_outputs/mimic4_mortality/vcip/42/` |
| Device | GPU (CUDA) se disponível, CPU caso contrário |

### 5.6 Métricas monitoradas

| Métrica | Origem | Significado |
|---|---|---|
| `train_loss` | `training_step` | ELBO no treino (menor = melhor) |
| `val_loss` | `validation_step` | ELBO na validação (monitorado pelo ES) |
| `val_reg_loss` | `validation_step` | Só o termo de reconstrução BCE |
| `train_reg_loss` | `on_train_epoch_end` | Média da reconstrução no treino |
| `val_reg_loss` | `on_validation_epoch_end` | Média da reconstrução na validação |

---

## 6. Reformulação tau=1 (Seção 22) — Versão Corrigida

### 6.1 Mudança de pergunta

Em vez de "planejar uma sequência de τ=4 decisões nas próximas 4 horas" (impossível sem dados pós-T0), o modelo passa a responder:

> **Dado todo o histórico de 48h pré-T0, devo transfundir em T0 — sim ou não?**

Isso mapeia perfeitamente para a estrutura dos dados disponíveis.

### 6.2 Treatment tensor com posição fixa

Em vez de colocar `A[i]` no último bin observado de cada paciente (posição variável), o tratamento vai para a **posição fixa 46** do tensor original (= posição **45** de `current_treatments`) para **todos** os pacientes:

```python
# Seção 14 (tau=4, variável por paciente):
treatment_tensor[i, seq_lengths[i]-1, 0] = A[i]   # posição varia entre 24–47

# Seção 22 (tau=1, fixo para todos):
treatment_tensor[i, 46, 0] = A[i]                 # sempre posição 46 = slot 45 em current_treatments
```

### 6.3 FixedHistoryCIPDataset

O `CIPDataset` padrão sorteia `history_length h` aleatoriamente. Para tau=1, isso é substituído por uma subclasse que força `h = 45` sempre:

```python
class FixedHistoryCIPDataset(CIPDataset):
    def __init__(self, data, config, train=False):
        super().__init__(data, config, train=train)
        self.history_lengths     = np.array([45])   # sempre h=45
        self.repeats             = 1
        self.samples_per_history = len(data['outputs'])
```

Com `h = 45` e `tau = 1`:

```
H_t    = current_treatments[0:45]    ← 45 timesteps pré-T0 (A=0 em todos)
target = current_treatments[45:46]   ← slot 45 = A[i] para todos os pacientes ✓
```

Cobertura: **100% das amostras** têm o tratamento real no target (vs. 1,5% no tau=4).

### 6.4 Treino com dataloaders externos

Como `train_dataloader()` e `val_dataloader()` do `VAEModel` instanciam `CIPDataset` internamente, o `FixedHistoryCIPDataset` é passado diretamente ao `trainer.fit()`:

```python
trainer1.fit(model1,
             train_dataloaders=train_dl1,   # FixedHistoryCIPDataset
             val_dataloaders=val_dl1)       # FixedHistoryCIPDataset
```

O PyTorch Lightning ignora `train_dataloader()` do modelo quando dataloaders são passados explicitamente ao `fit()`.

### 6.5 Reutilização do AuxiliaryModel

O `AuxiliaryModel` é treinado separadamente usando o `DataLoader` simples (não `CIPDataset`). Sua arquitetura e treinamento não dependem de `tau`. Por isso, o `AuxiliaryModel` treinado nas Seções 16–17 é **reutilizado diretamente** no tau=1, sem re-treino.

### 6.6 Inferência: apenas 2 sequências

Com tau=1, há exatamente **2 sequências possíveis**. Para cada paciente:

```python
seq0 = tensor([[[0.]]])   # não transfundir
seq1 = tensor([[[1.]]])   # transfundir

elbo0, _, _ = model1.calculate_elbo(H_t, Y_target, seq0, optimize_a=True)
elbo1, _, _ = model1.calculate_elbo(H_t, Y_target, seq1, optimize_a=True)

vcip_recomenda_transfusao = (elbo1 < elbo0)
```

O `elbo_diff = ELBO([1]) − ELBO([0])` funciona como **escore de plausibilidade causal**:
- Negativo → modelo prefere transfundir para este paciente
- Positivo → modelo prefere não transfundir
- Distribuição de `elbo_diff` por grupo AIPW (B1/B2 vs M1/M2/M3) é a principal métrica de validação

### 6.7 Teste de consistência com AIPW

A hipótese testada (Seção 22D):

> Se VCIP e AIPW capturam o mesmo fenômeno causal, a taxa de recomendação de transfusão pelo VCIP deve ser **maior nos grupos B1/B2** (benefício segundo AIPW) do que nos grupos **M1/M2/M3** (malefício).

Isso é uma validação cruzada entre dois métodos independentes: o AIPW nunca viu as séries temporais; o VCIP nunca viu os estimandos AIPW.

---

## 7. Fluxo completo — diagrama

```
raw_temporal.parquet
        │
        ▼
[Seção 14] Grid 1484×48 + forward-fill + obs_mask
        │
        ▼
[Seção 15] Split por subject_id → NaN→mediana(treino) → z-score(treino) → vcip_tensors.pkl
        │
        ▼
[Seção 16] Carrega pkl → make_split() → DatasetWrapper × 3 → DatasetCollectionWrapper
        │
        ▼
[Seção 17] AuxiliaryModel.fit() + VAEModel(cfg, dataset_collection)
        │
        ├─── [tau=4, QUEBRADO] ──────────────────────────────────────────────────┐
        │                                                                         │
        ▼                                                                         │
[Seção 17] pl.Trainer.fit(model)                                                  │
              ├─ CIPDataset: fatia H_t=[0:h], target=[h:h+τ], h aleatório         │
              ├─ tratamento na posição seq_len−1 (variável por paciente)           │
              ├─ 98.5% batches com zero tratamento no target                       │
              └─ modelo aprende â*=[0,0,0,0] para todos os pacientes               │
        │                                                                         │
        ▼                                                                         │
[Seção 18] load_from_checkpoint → optimize_interventions_discrete_onetime()       │
        │                                                                         │
        ▼                                                                         │
[Seções 19-21] GRP, análise por grupo AIPW (0% de recomendação)    ◄─────────────┘
        │
        ├─── [tau=1, REFORMULAÇÃO] ─────────────────────────────────────────────┐
        │                                                                        │
        ▼                                                                        │
[Seção 22A] Novo treatment tensor: A[i] em posição fixa 46 para todos           │
        │                                                                        │
        ▼                                                                        │
[Seção 22B] FixedHistoryCIPDataset (h=45 fixo) + trainer1.fit()                 │
              ├─ target=[45:46] sempre contém A[i] → 100% cobertura              │
              └─ AuxiliaryModel reutilizado (não depende de tau)                 │
        │                                                                        │
        ▼                                                                        │
[Seção 22C] Inferência: ELBO([0]) vs ELBO([1]) por paciente                     │
              └─ elbo_diff = ELBO([1])−ELBO([0]); negativo = recomendar          │
        │                                                                        │
        ▼                                                                        │
[Seção 22D] Validação cruzada VCIP × AIPW: taxa de recomendação                 │
              └─ hipótese: B1/B2 > M1/M2/M3 em recomendações  ◄─────────────────┘
```

---

## 8. Limitações deste protótipo

| Limitação | Causa | Status |
|---|---|---|
| Janela só pré-T0 | `raw_temporal` cobre apenas 48h antes de T0 | **Estrutural**: o VCIP não vê o que acontece após a transfusão |
| Resolução de 1h | Timegrid bruto é de 5 min; raw_temporal já é agregado a 1h | ΔHb(30min) e outros deltas de curto prazo são inviáveis |
| tau=4 nunca recomenda transfusão | 98.5% dos batches têm zero tratamento no target | **Endereçado** pela reformulação tau=1 (Seção 22) |
| `simulate_output_after_actions` retorna zeros | Não há simulador contrafactual para dados observacionais | RCS indefinido; apenas GRP (ELBO) + validação AIPW disponíveis |
| Desfecho binário estático | `mortality_anytime` é repetida em todos os τ slots | O modelo aprende mortalidade como variável estacionária |
| tau=1 limita recomendação a T0 | Sem dados pós-T0, não há como planejar sequências futuras | Para recomendação temporal, precisa-se do timegrid completo |

Para o VCIP completo (conforme o paper), seria necessário o timegrid de 5 min cobrindo a internação completa (incluindo pós-transfusão), com as variáveis adicionais do Grupo B e C descritas no `CLAUDE.md`.
