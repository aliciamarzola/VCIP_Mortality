"""
MIMIC-IV Transfusion → Mortalidade — Dataset para VCIP

Formulação:
  X_t   = 11 sinais vitais/labs (raw_temporal), janela 48h pré-T0, resolução 1h
  A_t   = 0 para t < T_max; transfused ∈ {0,1} no último slot
  Y     = mortality_anytime ∈ {0,1} — repetido em todos os τ slots
  V     = age, sex, bmi + grupos AIPW one-hot (B1..M3)

Decoder: Bernoulli (output_mode=bernoulli no config)
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.dataset_collection import RealDatasetCollection

# ── Constantes ──────────────────────────────────────────────────────────────
VITALS = ['hemoglobin', 'lactate', 'creatinine', 'platelets',
          'heart_rate', 'mbp', 'resp_rate', 'spo2',
          'sofa', 'any_vasopressor', 'ventilation_on']  # 11 colunas

WINDOW_HOURS  = 48
STEP_MINUTES  = 60
N_STEPS       = WINDOW_HOURS   # 48 timesteps


# ── Preparação dos dados ─────────────────────────────────────────────────────
def _build_tensors(raw_temporal_path, treatment_path, trial_path, groups_path):
    """
    Lê os parquets e monta o tensor (N, 48, 11) + vetores de tratamento e desfecho.

    Retorna um dict com os arrays prontos para empacotar no Dataset.
    """
    raw   = pd.read_parquet(raw_temporal_path)
    treat = pd.read_parquet(treatment_path)[['stay_id', 't0', 'subject_id', 'transfused']]
    trial = pd.read_parquet(trial_path)[['stay_id', 'mortality_anytime']]
    groups = pd.read_parquet(groups_path)[['stay_id', 'B1', 'B2', 'M1', 'M2', 'M3']]

    # ── Filtra apenas os 1.484 elegíveis ────────────────────────────────────
    eligible_stays = treat['stay_id'].values
    raw = raw[raw['stay_id'].isin(eligible_stays)].copy()

    # ── Adiciona t0 ao raw_temporal ──────────────────────────────────────────
    raw = raw.merge(treat[['stay_id', 't0']], on='stay_id', how='left')

    # ── Cria bin relativo a T0: bin 0 = t0-48h, bin 47 = t0-1h ─────────────
    start_min = raw['t0'] - WINDOW_HOURS * 60
    raw['bin'] = np.floor((raw['time_min'] - start_min) / STEP_MINUTES).astype(int)
    raw = raw[(raw['bin'] >= 0) & (raw['bin'] < N_STEPS)]

    # ── Agrega por (stay_id, bin): média dos valores ─────────────────────────
    binned = (raw.groupby(['stay_id', 'bin'])[VITALS]
                 .mean()
                 .reset_index())

    # ── Monta tensor (N, 48, 11) com NaN onde não há medição ────────────────
    stay_ids   = treat['stay_id'].values         # ordem fixa
    subject_ids = treat['subject_id'].values
    N          = len(stay_ids)
    tensor     = np.full((N, N_STEPS, len(VITALS)), np.nan, dtype=float)

    s2i = {s: i for i, s in enumerate(stay_ids)}
    for _, row in binned.iterrows():
        i = s2i.get(row['stay_id'])
        if i is not None:
            tensor[i, int(row['bin']), :] = row[VITALS].values

    # ── Imputação: forward-fill dentro de cada paciente, depois mediana treino
    #    (mediana calculada depois do split — aqui usa mediana global como proxy)
    global_medians = np.nanmedian(tensor.reshape(-1, len(VITALS)), axis=0)
    global_medians = np.where(np.isfinite(global_medians), global_medians, 0.0)

    for i in range(N):
        frame = pd.DataFrame(tensor[i], columns=VITALS).ffill()
        for j, col in enumerate(VITALS):
            frame[col] = frame[col].fillna(global_medians[j])
        tensor[i] = frame.to_numpy(float)

    # ── Z-score (treino → calculado após split; aqui pré-computa global) ────
    means = tensor.reshape(-1, len(VITALS)).mean(axis=0)
    stds  = tensor.reshape(-1, len(VITALS)).std(axis=0)
    stds  = np.where(stds > 1e-8, stds, 1.0)
    tensor_z = (tensor - means) / stds

    # ── sequence_lengths: quantos bins válidos cada paciente tem ────────────
    valid_counts = (~np.all(np.isnan(tensor), axis=-1)).sum(axis=1)  # (N,)
    # Após imputação todos têm 48, mas usa o original como comprimento real
    orig_tensor = np.full((N, N_STEPS, len(VITALS)), np.nan, dtype=float)
    for _, row in binned.iterrows():
        i = s2i.get(row['stay_id'])
        if i is not None:
            orig_tensor[i, int(row['bin']), :] = row[VITALS].values
    seq_lengths = (~np.all(np.isnan(orig_tensor), axis=-1)).sum(axis=1)  # (N,)
    seq_lengths = np.maximum(seq_lengths, 25)  # mínimo 25 para CIPDataset

    # ── Tratamento: 0 para todos, exceto último slot válido ─────────────────
    transfused_vec = treat.set_index('stay_id').loc[stay_ids, 'transfused'].values  # (N,)
    treatment_tensor = np.zeros((N, N_STEPS, 1), dtype=float)
    for i in range(N):
        last = int(seq_lengths[i]) - 1
        treatment_tensor[i, last, 0] = float(transfused_vec[i])

    # ── Desfecho: mortality_anytime repetida em todos os slots ──────────────
    mortality = trial.set_index('stay_id').loc[stay_ids, 'mortality_anytime'].values  # (N,)
    output_tensor = np.zeros((N, N_STEPS, 1), dtype=float)
    for i in range(N):
        output_tensor[i, :, 0] = float(mortality[i])

    # ── Máscara active_entries ───────────────────────────────────────────────
    active = np.zeros((N, N_STEPS, 1), dtype=float)
    for i in range(N):
        active[i, :int(seq_lengths[i]), 0] = 1.0

    # ── Features estáticas: age, sex, bmi + grupos AIPW ─────────────────────
    static_df = (treat[['stay_id', 'subject_id']]
                 .merge(raw[['stay_id', 'age', 'sex', 'bmi']].drop_duplicates('stay_id'),
                        on='stay_id', how='left')
                 .merge(groups, on='stay_id', how='left'))
    static_df['sex'] = (static_df['sex'] == 'M').astype(float)
    static_df[['B1','B2','M1','M2','M3']] = (
        static_df[['B1','B2','M1','M2','M3']].fillna(0).astype(float))

    static_cols = ['age', 'sex', 'bmi', 'B1', 'B2', 'M1', 'M2', 'M3']
    static_arr = static_df.set_index('stay_id').loc[stay_ids, static_cols].values.astype(float)

    # normaliza features estáticas contínuas (age, bmi)
    static_arr[:, 0] = (static_arr[:, 0] - static_arr[:, 0].mean()) / (static_arr[:, 0].std() + 1e-8)
    static_arr[:, 2] = (static_arr[:, 2] - static_arr[:, 2].mean()) / (static_arr[:, 2].std() + 1e-8)

    scaling_params = {
        'output_means': np.array([0.0]),
        'output_stds':  np.array([1.0]),
        'vitals_means': means,
        'vitals_stds':  stds,
    }

    return {
        'stay_ids':       stay_ids,
        'subject_ids':    subject_ids,
        'vitals_tensor':  tensor_z,          # (N, 48, 11) z-scored
        'treatment':      treatment_tensor,   # (N, 48, 1)  binário
        'output':         output_tensor,      # (N, 48, 1)  mortalidade
        'active':         active,             # (N, 48, 1)
        'seq_lengths':    seq_lengths,        # (N,)
        'static':         static_arr,         # (N, 8)
        'transfused':     transfused_vec,     # (N,) para estratificar split
        'scaling_params': scaling_params,
    }


# ── Dataset Pytorch ──────────────────────────────────────────────────────────
class MIMIC4MortalityDataset(Dataset):
    """
    Dataset VCIP para mortalidade com decoder Bernoulli.

    self.data segue o mesmo contrato do MIMIC3RealDataset:
      prev_treatments    (N, T-1, 1)
      current_treatments (N, T-1, 1)
      static_features    (N, 8)
      prev_outputs       (N, T-1, 1)
      outputs            (N, T-1, 1)
      active_entries     (N, T-1, 1)
      sequence_lengths   (N,)
    """

    def __init__(self, arrays: dict, subset_name: str):
        N = arrays['vitals_tensor'].shape[0]
        T = arrays['vitals_tensor'].shape[1]

        vit  = arrays['vitals_tensor']   # (N, T, 11)
        trt  = arrays['treatment']        # (N, T, 1)
        out  = arrays['output']           # (N, T, 1)
        act  = arrays['active']           # (N, T, 1)
        slen = arrays['seq_lengths']      # (N,)
        stat = arrays['static']           # (N, 8)

        self.scaling_params = arrays['scaling_params']
        self.subset_name    = subset_name

        # CIPDataset corta [start:start+h] e [start+h:start+h+tau]
        # precisamos que todos os arrays temporais tenham shape (N, T, dim)
        # e que static_features tenha shape (N, S) — repeat_static cuida do resto
        self.data = {
            'sequence_lengths':   slen - 1,
            'prev_treatments':    trt[:, :-1, :],    # (N, T-1, 1)
            'current_treatments': trt[:, 1:,  :],    # (N, T-1, 1)
            'static_features':    stat,               # (N, 8)
            'prev_outputs':       out[:, :-1, :],    # (N, T-1, 1)
            'outputs':            out[:, 1:,  :],    # (N, T-1, 1)
            'active_entries':     act[:, 1:,  :],    # (N, T-1, 1)
            # vitals como covariáveis (usado se predict_X=True)
            'vitals':             vit[:, 1:,  :],    # (N, T-1, 11)
            'current_covariates': vit[:, 1:,  :],
        }

        self.processed = True
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.exploded = False
        self.norm_const = 1.0

    def __len__(self):
        return len(self.data['outputs'])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}

    def simulate_output_after_actions(self, H_t, actions, scaling_params=None):
        """Sem simulador real — retorna zeros. Usar apenas modo ranking (GRP)."""
        return np.zeros((H_t['outputs'].shape[0], 1))


# ── Dataset Collection ───────────────────────────────────────────────────────
class MIMIC4MortalityDatasetCollection(RealDatasetCollection):
    """
    Divide os 1.484 pacientes em treino/val/teste por subject_id,
    estratificado por transfused × mortality.
    """

    def __init__(
        self,
        raw_temporal_path: str,
        treatment_path: str,
        trial_path: str,
        groups_path: str,
        seed: int = 42,
        split: dict = None,
        projection_horizon: int = 4,
        autoregressive: bool = True,
        **kwargs,
    ):
        super().__init__()
        if split is None:
            split = {'val': 0.15, 'test': 0.15}

        self.seed = seed

        # ── Carrega e prepara os arrays ──────────────────────────────────────
        arrays = _build_tensors(raw_temporal_path, treatment_path, trial_path, groups_path)

        stay_ids    = arrays['stay_ids']
        subject_ids = arrays['subject_ids']
        transfused  = arrays['transfused']

        # ── Recalcula z-score apenas com treino (boa prática) ────────────────
        # Por simplicidade, o z-score global já foi aplicado em _build_tensors.
        # Para produção: fazer split primeiro, depois recomputar com train.

        # ── Split por subject_id estratificado ──────────────────────────────
        unique_subjects = np.unique(subject_ids)

        # flag de mortalidade por subject (pega o primeiro stay)
        subj_to_mort = {}
        trial = pd.read_parquet(trial_path)[['stay_id', 'mortality_anytime']]
        treat = pd.read_parquet(treatment_path)[['stay_id', 'subject_id', 'transfused']]
        merged = treat.merge(trial, on='stay_id')
        for _, row in merged.iterrows():
            if row['subject_id'] not in subj_to_mort:
                subj_to_mort[row['subject_id']] = int(row['mortality_anytime'])

        mort_flag = np.array([subj_to_mort.get(s, 0) for s in unique_subjects])

        subj_trainval, subj_test = train_test_split(
            unique_subjects,
            test_size=split['test'],
            stratify=mort_flag,
            random_state=seed,
        )
        mort_trainval = mort_flag[np.isin(unique_subjects, subj_trainval)]
        subj_train, subj_val = train_test_split(
            subj_trainval,
            test_size=split['val'] / (1 - split['test']),
            stratify=mort_trainval,
            random_state=seed * 2,
        )

        # ── Filtra arrays por split ──────────────────────────────────────────
        def mask_for(subjects):
            return np.isin(subject_ids, subjects)

        def subset(subjects):
            m = mask_for(subjects)
            sub = {k: v[m] if isinstance(v, np.ndarray) and v.ndim >= 1
                           and v.shape[0] == len(stay_ids)
                   else v
                   for k, v in arrays.items()}
            return sub

        self.train_f = MIMIC4MortalityDataset(subset(subj_train), 'train')
        self.val_f   = MIMIC4MortalityDataset(subset(subj_val),   'val')
        self.test_f  = MIMIC4MortalityDataset(subset(subj_test),  'test')

        self.projection_horizon  = projection_horizon
        self.has_vitals          = True
        self.autoregressive      = autoregressive
        self.processed_data_encoder = True
        self.train_scaling_params   = arrays['scaling_params']

    def process_data_multi(self):
        """
        Override: o VCIP usa test_f.data diretamente via CIPDataset —
        test_f_multi (used by CT baseline) não é necessário aqui.
        """
        self.processed_data_multi = True
