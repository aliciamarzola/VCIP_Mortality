#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ID_COL = "stay_id"
TREAT_COL = "has_transfusion"
CLUSTER_COL = "cluster_kmeans"
MORT_COL = "mortality_anytime"
VM_COL = "vm_time_hours"

WINDOWS = [24, 48, 72]
WEIGHT_CLIP_Q = (0.01, 0.99)
PSCORE_CLIP = (0.01, 0.99)
BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 42
VFD_DAYS = float(os.getenv("VFD_DAYS", "28"))
HB_BINS = (7.0, 8.0)

STABILITY_RATIO_LOW = 0.9
STABILITY_RATIO_HIGH = 1.1
EPS = 1e-9

TOP_N = 20
EXPORT_FULL_EXAM_TABLE = True
MORT_DIFF_EPS = float(os.getenv("MORT_DIFF_EPS", "0.0"))
VM_DIFF_EPS = float(os.getenv("VM_DIFF_EPS", "0.0"))
EXAM_SIGNAL_EPS = float(os.getenv("EXAM_SIGNAL_EPS", "0.0"))
EXAM_DIRECTION_OVERRIDES = os.getenv("EXAM_DIRECTION_OVERRIDES", "")
LABEL_VARIANTS = [
    ("_k2", "k2"),
    ("_k3", "k3"),
    ("_k4", "k4"),
]

IMPROVE_HIGH_PATTERNS = [
    r"^albumin",
    r"^hb",
    r"^hgb",
    r"^hemoglobin",
    r"^hematocrit",
    r"^hct",
    r"^rbc",
    r"^platelet",
    r"^plt",
    r"^spo2",
    r"^sao2",
    r"^o2sat",
    r"^pao2",
    r"^po2",
    r"^pao2_fio2",
    r"^pao2fio2",
    r"^pf_ratio",
    r"^pfratio",
    r"^bicarbonate",
    r"^hco3",
    r"^gcs",
    r"^sbp",
    r"^dbp",
    r"^map",
    r"^mbp",
    r"^meanbp",
]
IMPROVE_LOW_PATTERNS = [
    r"^lactate",
    r"^creatinine",
    r"^bun",
    r"^wbc",
    r"^crp",
    r"^d_dimer",
    r"^ferritin",
    r"^troponin",
    r"^ck",
    r"^ckmb",
    r"^ldh",
    r"^procalcitonin",
    r"^bilirubin",
    r"^inr",
    r"^ptt",
    r"^pt$",
    r"^ast",
    r"^alt",
    r"^fio2",
    r"^temperature",
    r"^temp",
    r"^heartrate",
    r"^hr",
    r"^resp",
    r"^rr",
    r"^sofa",
    r"^apache",
    r"^vasopressor",
    r"^norepinephrine",
    r"^epinephrine",
    r"^dopamine",
    r"^vasopressin",
]


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "src").exists() and (parent / "outputs").exists():
            return parent
    return start


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    if mask.sum() == 0:
        return float("nan")
    w = weights[mask]
    x = values[mask]
    w_sum = np.sum(w)
    if w_sum <= 0:
        return float("nan")
    mean = np.sum(w * x) / w_sum
    return float(np.sum(w * (x - mean) ** 2) / w_sum)


def _smd(
    x_t: np.ndarray,
    x_c: np.ndarray,
    w_t: Optional[np.ndarray] = None,
    w_c: Optional[np.ndarray] = None,
) -> float:
    if x_t.size == 0 or x_c.size == 0:
        return float("nan")
    if w_t is None:
        mean_t = float(np.mean(x_t))
        var_t = float(np.var(x_t, ddof=0))
    else:
        mean_t = _weighted_mean(x_t, w_t)
        var_t = _weighted_var(x_t, w_t)
    if w_c is None:
        mean_c = float(np.mean(x_c))
        var_c = float(np.var(x_c, ddof=0))
    else:
        mean_c = _weighted_mean(x_c, w_c)
        var_c = _weighted_var(x_c, w_c)
    denom = np.sqrt((var_t + var_c) / 2.0) if np.isfinite(var_t) and np.isfinite(var_c) else float("nan")
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float((mean_t - mean_c) / denom)


def _smd_summary(x: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    if x.size == 0:
        return {"smd_mean": float("nan"), "smd_max": float("nan"), "smd_prop_gt_0_1": float("nan")}
    mask_t = t == 1
    mask_c = t == 0
    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return {"smd_mean": float("nan"), "smd_max": float("nan"), "smd_prop_gt_0_1": float("nan")}
    smds = []
    for j in range(x.shape[1]):
        x_t = x[mask_t, j]
        x_c = x[mask_c, j]
        w_t = w[mask_t] if w is not None else None
        w_c = w[mask_c] if w is not None else None
        smds.append(abs(_smd(x_t, x_c, w_t, w_c)))
    smds = np.array(smds, dtype=np.float64)
    return {
        "smd_mean": float(np.nanmean(smds)),
        "smd_max": float(np.nanmax(smds)),
        "smd_prop_gt_0_1": float(np.mean(smds > 0.1)),
    }


def _bootstrap_ipw_diff(
    values_t: np.ndarray,
    weights_t: np.ndarray,
    values_c: np.ndarray,
    weights_c: np.ndarray,
    n_boot: int,
    rng: np.random.RandomState,
) -> Tuple[float, float, float, float]:
    if values_t.size == 0 or values_c.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs = []
    n_t = values_t.size
    n_c = values_c.size
    for _ in range(n_boot):
        idx_t = rng.randint(0, n_t, size=n_t)
        idx_c = rng.randint(0, n_c, size=n_c)
        mean_t = _weighted_mean(values_t[idx_t], weights_t[idx_t])
        mean_c = _weighted_mean(values_c[idx_c], weights_c[idx_c])
        diffs.append(mean_t - mean_c)
    diffs = np.array(diffs, dtype=np.float64)
    lo = float(np.nanpercentile(diffs, 2.5))
    hi = float(np.nanpercentile(diffs, 97.5))
    p = float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return float(np.nanmean(diffs)), lo, hi, p


def _aipw_ate(y: np.ndarray, t: np.ndarray, p: np.ndarray, mu1: np.ndarray, mu0: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    term = mu1 - mu0 + t * (y - mu1) / p - (1 - t) * (y - mu0) / (1 - p)
    return float(np.nanmean(term))


def _fit_outcome_models(x: np.ndarray, t: np.ndarray, y: np.ndarray, is_binary: bool) -> Tuple[np.ndarray, np.ndarray]:
    mu1 = np.full_like(y, np.nan, dtype=np.float64)
    mu0 = np.full_like(y, np.nan, dtype=np.float64)
    overall_mean = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else float("nan")
    for grp, out in [(1, mu1), (0, mu0)]:
        mask = t == grp
        if mask.sum() == 0:
            continue
        y_g = y[mask]
        x_g = x[mask]
        finite = np.isfinite(y_g)
        if finite.sum() == 0:
            out[:] = overall_mean
            continue
        y_g = y_g[finite]
        x_g = x_g[finite]
        if is_binary:
            if np.unique(y_g[~np.isnan(y_g)]).size < 2:
                out[:] = float(np.nanmean(y_g))
                continue
            model = LogisticRegression(max_iter=1000, solver="liblinear")
            model.fit(x_g, y_g)
            out[:] = model.predict_proba(x)[:, 1]
        else:
            model = LinearRegression()
            model.fit(x_g, y_g)
            out[:] = model.predict(x)
    return mu1, mu0


def _nn_match_diff(y: np.ndarray, t: np.ndarray, p: np.ndarray) -> float:
    finite = np.isfinite(y)
    t = t[finite]
    p = p[finite]
    y = y[finite]
    mask_t = t == 1
    mask_c = t == 0
    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return float("nan")
    p_c = p[mask_c]
    y_c = y[mask_c]
    y_t = y[mask_t]
    p_t = p[mask_t]
    matched = []
    for val in p_t:
        idx = int(np.argmin(np.abs(p_c - val)))
        matched.append(y_c[idx])
    return float(np.nanmean(y_t) - np.nanmean(matched))


def _compute_vfd(vm_hours: pd.Series, mortality: pd.Series) -> pd.Series:
    vm_days = vm_hours / 24.0
    vfd = np.maximum(0.0, VFD_DAYS - vm_days)
    vfd = vfd.where(mortality != 1, 0.0)
    return vfd


def _hb_stratum(hb: float) -> str:
    if np.isnan(hb):
        return "hb_unknown"
    if hb < HB_BINS[0]:
        return "hb_lt7"
    if hb < HB_BINS[1]:
        return "hb_7_8"
    return "hb_gt8"


def _fit_propensity(
    df: pd.DataFrame,
    pre_cols: List[str],
    return_model_data: bool = False,
):
    x = df[pre_cols].to_numpy(dtype=np.float64, copy=True)
    y = df[TREAT_COL].to_numpy(dtype=np.int64, copy=True)

    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x)
    x_imputed = x.copy()

    var = x_imputed.var(axis=0)
    keep = var > 0
    x_imputed = x_imputed[:, keep]
    keep_cols = [c for c, ok in zip(pre_cols, keep) if ok]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(x_scaled, y)
    pscore = model.predict_proba(x_scaled)[:, 1]
    pscore = np.clip(pscore, PSCORE_CLIP[0], PSCORE_CLIP[1])

    p_treat = float(y.mean())
    weights = np.where(y == 1, p_treat / pscore, (1.0 - p_treat) / (1.0 - pscore))

    lo, hi = np.quantile(weights, WEIGHT_CLIP_Q)
    weights = np.clip(weights, lo, hi)

    auc = float(roc_auc_score(y, pscore))
    info = {
        "p_treat": p_treat,
        "auc": auc,
        "weights_clip_q": WEIGHT_CLIP_Q,
        "weights_clip_lo": float(lo),
        "weights_clip_hi": float(hi),
    }
    if return_model_data:
        return weights, info, pscore, x_imputed, x_scaled, keep_cols
    return weights, info


def _load_direction_overrides() -> Dict[str, int]:
    if not EXAM_DIRECTION_OVERRIDES:
        return {}
    path = Path(EXAM_DIRECTION_OVERRIDES)
    if not path.exists():
        raise FileNotFoundError(f"EXAM_DIRECTION_OVERRIDES not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides = {}
    for key, value in data.items():
        try:
            direction = int(value)
        except (TypeError, ValueError):
            continue
        if direction not in (-1, 1):
            continue
        overrides[key.lower()] = direction
    return overrides


def _direction_for_feature(feature: str, overrides: Dict[str, int]) -> int:
    key = feature.lower()
    if key in overrides:
        return overrides[key]
    for pattern in IMPROVE_HIGH_PATTERNS:
        if re.match(pattern, key):
            return 1
    for pattern in IMPROVE_LOW_PATTERNS:
        if re.match(pattern, key):
            return -1
    return 0


def _classify_change_direction(delta_diff: float, direction: int) -> str:
    if np.isnan(delta_diff):
        return "sem_comparacao"
    if direction == 0:
        return "sem_direcao"
    if abs(delta_diff) <= EXAM_SIGNAL_EPS:
        return "sem_mudanca"
    if direction > 0:
        return "melhora" if delta_diff > 0 else "piora"
    return "melhora" if delta_diff < 0 else "piora"


def _classify_stability(abs_t: float, abs_c: float) -> str:
    if np.isnan(abs_t) or np.isnan(abs_c):
        return "sem_comparacao"
    if abs_c < EPS and abs_t < EPS:
        return "sem_mudanca"
    if abs_c < EPS and abs_t >= EPS:
        return "piora"
    if abs_t < EPS and abs_c >= EPS:
        return "melhora"
    ratio = abs_t / abs_c if abs_c > 0 else np.inf
    if ratio < STABILITY_RATIO_LOW:
        return "melhora"
    if ratio > STABILITY_RATIO_HIGH:
        return "piora"
    return "sem_mudanca"


def _classify_effect(mort_diff: float, vm_diff: float) -> str:
    if np.isnan(mort_diff) or np.isnan(vm_diff):
        return "sem_comparacao"
    if mort_diff < 0 and vm_diff < 0:
        return "beneficio"
    if mort_diff > 0 and vm_diff > 0:
        return "prejuizo"
    return "sem_comparacao"


def _classify_effect_eps(mort_diff: float, vm_diff: float, exam_signal: float) -> str:
    if np.isnan(mort_diff) or np.isnan(vm_diff) or np.isnan(exam_signal):
        return "sem_comparacao"
    mort_ok = mort_diff <= -MORT_DIFF_EPS
    vm_ok = vm_diff <= -VM_DIFF_EPS
    mort_bad = mort_diff >= MORT_DIFF_EPS
    vm_bad = vm_diff >= VM_DIFF_EPS
    exam_ok = exam_signal >= EXAM_SIGNAL_EPS
    exam_bad = exam_signal <= -EXAM_SIGNAL_EPS
    if mort_ok and vm_ok and exam_ok:
        return "beneficio"
    if mort_bad and vm_bad and exam_bad:
        return "prejuizo"
    return "sem_comparacao"


def _classify_mortality_only(mort_diff: float) -> str:
    if np.isnan(mort_diff):
        return "sem_comparacao"
    if mort_diff < 0:
        return "beneficio"
    if mort_diff > 0:
        return "prejuizo"
    return "sem_comparacao"


def _classify_mortality_eps(mort_diff: float) -> str:
    if np.isnan(mort_diff):
        return "sem_comparacao"
    if mort_diff <= -MORT_DIFF_EPS:
        return "beneficio"
    if mort_diff >= MORT_DIFF_EPS:
        return "prejuizo"
    return "sem_comparacao"


def _classify_vm_only(vm_diff: float) -> str:
    if np.isnan(vm_diff):
        return "sem_comparacao"
    if vm_diff < 0:
        return "beneficio"
    if vm_diff > 0:
        return "prejuizo"
    return "sem_comparacao"


def _classify_vm_eps(vm_diff: float) -> str:
    if np.isnan(vm_diff):
        return "sem_comparacao"
    if vm_diff <= -VM_DIFF_EPS:
        return "beneficio"
    if vm_diff >= VM_DIFF_EPS:
        return "prejuizo"
    return "sem_comparacao"


def _load_features(path: Path) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    schema = pq.read_schema(path)
    pre_cols = [name for name in schema.names if name.endswith("_pre_mean")]
    post_cols = [name for name in schema.names if name.endswith("_post_mean")]
    delta_cols = [name for name in schema.names if name.endswith("_delta")]
    cols = [ID_COL, TREAT_COL] + pre_cols + post_cols + delta_cols
    df = pd.read_parquet(path, columns=cols)
    return df, pre_cols, post_cols, delta_cols


def _label_column_for_suffix(suffix: str) -> str:
    if not suffix:
        return CLUSTER_COL
    if suffix.startswith("_k") and suffix[2:].isdigit():
        return f"{CLUSTER_COL}_k{suffix[2:]}"
    return CLUSTER_COL


def _cluster_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cl in sorted(df[CLUSTER_COL].dropna().unique()):
        dfc = df[df[CLUSTER_COL] == cl]
        if dfc.empty:
            continue

        df0 = dfc[dfc[TREAT_COL] == 0]
        df1 = dfc[dfc[TREAT_COL] == 1]

        mort0 = _weighted_mean(df0[MORT_COL].to_numpy(), df0["ipw"].to_numpy())
        mort1 = _weighted_mean(df1[MORT_COL].to_numpy(), df1["ipw"].to_numpy())
        vm0 = _weighted_mean(df0[VM_COL].to_numpy(), df0["ipw"].to_numpy())
        vm1 = _weighted_mean(df1[VM_COL].to_numpy(), df1["ipw"].to_numpy())

        rows.append(
            {
                CLUSTER_COL: int(cl),
                "n_total": int(len(dfc)),
                "n_transfused": int(len(df1)),
                "n_non_transfused": int(len(df0)),
                "mortality_rate_transfused_ipw": mort1,
                "mortality_rate_non_transfused_ipw": mort0,
                "vm_mean_transfused_ipw": vm1,
                "vm_mean_non_transfused_ipw": vm0,
                "mortality_rate_diff_ipw": mort1 - mort0 if np.isfinite(mort1) and np.isfinite(mort0) else float("nan"),
                "vm_mean_diff_ipw": vm1 - vm0 if np.isfinite(vm1) and np.isfinite(vm0) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _exam_table(
    df: pd.DataFrame,
    pre_cols: List[str],
    post_cols: List[str],
    delta_cols: List[str],
    window: int,
    direction_overrides: Dict[str, int],
) -> pd.DataFrame:
    base_cols = [c[:-9] for c in pre_cols]
    group_cols = [CLUSTER_COL, TREAT_COL]
    agg_cols = pre_cols + post_cols + delta_cols
    grp = df[group_cols + agg_cols].groupby(group_cols, as_index=False).mean()

    rows: List[Dict[str, float]] = []
    for cl in sorted(df[CLUSTER_COL].unique()):
        g0 = grp[(grp[CLUSTER_COL] == cl) & (grp[TREAT_COL] == 0)]
        g1 = grp[(grp[CLUSTER_COL] == cl) & (grp[TREAT_COL] == 1)]
        for feat_base in base_cols:
            pre = f"{feat_base}_pre_mean"
            post = f"{feat_base}_post_mean"
            delta = f"{feat_base}_delta"

            pre0 = g0[pre].iloc[0] if not g0.empty else np.nan
            pre1 = g1[pre].iloc[0] if not g1.empty else np.nan
            post0 = g0[post].iloc[0] if not g0.empty else np.nan
            post1 = g1[post].iloc[0] if not g1.empty else np.nan
            d0 = g0[delta].iloc[0] if not g0.empty else np.nan
            d1 = g1[delta].iloc[0] if not g1.empty else np.nan
            abs_t = abs(d1) if pd.notna(d1) else np.nan
            abs_c = abs(d0) if pd.notna(d0) else np.nan
            change_abs = _classify_stability(abs_t, abs_c)
            ratio = abs_t / abs_c if pd.notna(abs_t) and pd.notna(abs_c) and abs_c > 0 else np.nan
            delta_diff = d1 - d0 if pd.notna(d1) and pd.notna(d0) else np.nan
            direction = _direction_for_feature(feat_base, direction_overrides)
            change_dir = _classify_change_direction(delta_diff, direction)

            rows.append(
                {
                    "window_hours": window,
                    CLUSTER_COL: int(cl),
                    "feature": feat_base,
                    "pre_mean_transfused": pre1,
                    "post_mean_transfused": post1,
                    "delta_transfused": d1,
                    "pre_mean_non_transfused": pre0,
                    "post_mean_non_transfused": post0,
                    "delta_non_transfused": d0,
                    "delta_diff": delta_diff,
                    "abs_delta_ratio": ratio,
                    "change_class": change_dir,
                    "change_class_abs": change_abs,
                    "direction": direction,
                }
            )

    return pd.DataFrame(rows)


def _exam_summary(exam_table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        exam_table.groupby([CLUSTER_COL, "change_class"], as_index=False)
        .size()
        .rename(columns={"size": "n_features"})
    )
    piv = (
        summary.pivot_table(index=CLUSTER_COL, columns="change_class", values="n_features", aggfunc="sum")
        .fillna(0.0)
        .reset_index()
    )
    for col in ["melhora", "piora", "sem_mudanca", "sem_comparacao", "sem_direcao"]:
        if col not in piv.columns:
            piv[col] = 0.0
    piv["n_compared"] = piv["melhora"] + piv["piora"] + piv["sem_mudanca"]
    denom = piv["n_compared"].replace(0, np.nan)
    piv["exam_signal"] = (piv["melhora"] - piv["piora"]) / denom
    return piv


def _top_n_exams(exam_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []
    for cl in sorted(exam_table[CLUSTER_COL].unique()):
        dfc = exam_table[exam_table[CLUSTER_COL] == cl].copy()
        dfc["abs_delta_diff"] = dfc["delta_diff"].abs()
        dfc = dfc.sort_values("abs_delta_diff", ascending=False).head(top_n)
        dfc = dfc.assign(rank=np.arange(1, len(dfc) + 1))
        rows.append(dfc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _phenotype_top_n(df: pd.DataFrame, pre_cols: List[str], window: int, top_n: int) -> pd.DataFrame:
    pre = df[[CLUSTER_COL] + pre_cols].copy()
    overall = pre[pre_cols].mean(axis=0)

    rows = []
    for cl in sorted(pre[CLUSTER_COL].dropna().unique()):
        dfc = pre[pre[CLUSTER_COL] == cl]
        means = dfc[pre_cols].mean(axis=0)
        diffs = (means - overall).abs().sort_values(ascending=False).head(top_n)
        for feat in diffs.index:
            rows.append(
                {
                    "window_hours": window,
                    CLUSTER_COL: int(cl),
                    "feature": feat.replace("_pre_mean", ""),
                    "pre_mean": means[feat],
                    "pre_diff": means[feat] - overall[feat],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    overall_t0 = time.time()
    root = find_repo_root(Path(__file__).resolve())
    outdir = root / "outputs" / "reports" / "clustering"
    outdir.mkdir(parents=True, exist_ok=True)
    inputs_dir = outdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    benefit_dir = outdir / "benefit"
    exams_dir = outdir / "exams"
    phenotypes_dir = outdir / "phenotypes"
    advanced_dir = outdir / "advanced"
    for d in [benefit_dir, exams_dir, phenotypes_dir, advanced_dir]:
        d.mkdir(parents=True, exist_ok=True)

    direction_overrides = _load_direction_overrides()
    if direction_overrides:
        print(f"[info] loaded {len(direction_overrides)} exam direction overrides")

    outcomes_path = root / "outputs_outcomes" / "outcomes_by_stay.csv"
    df_out = pd.read_csv(outcomes_path)
    df_out[MORT_COL] = df_out[MORT_COL].fillna(0).astype(int)
    df_out[VM_COL] = pd.to_numeric(df_out[VM_COL], errors="coerce")

    for label_suffix, label_tag in LABEL_VARIANTS:
        label_t0 = time.time()
        print(f"[progress] label={label_tag} start")
        all_cluster_rows = []
        all_top_rows = []
        all_exam_rows = []
        all_pheno_rows = []
        all_boot_rows = []
        all_diag_rows = []
        all_aipw_rows = []
        all_match_rows = []
        all_vfd_rows = []
        all_hb_rows = []
        rng = np.random.RandomState(BOOTSTRAP_SEED)
        label_col = _label_column_for_suffix(label_suffix)

        for w in WINDOWS:
            window_t0 = time.time()
            feat_name = f"tabular_features_by_stay_with_non_transfused_w{w}.parquet"
            labels_name = f"tabular_clusters_with_non_transfused_w{w}{label_suffix}.csv"
            feat_path = inputs_dir / feat_name
            labels_path = inputs_dir / labels_name
            if not feat_path.exists():
                feat_path = outdir / feat_name
            if not labels_path.exists():
                labels_path = outdir / labels_name
            if not feat_path.exists() or not labels_path.exists():
                print(f"[skip] missing inputs for w{w} ({label_tag})")
                continue

            df_feat, pre_cols, post_cols, delta_cols = _load_features(feat_path)
            df_labels = pd.read_csv(labels_path)
            if label_col not in df_labels.columns:
                print(f"[skip] missing label column {label_col} in {labels_path}")
                continue
            df = df_feat.merge(df_labels[[ID_COL, label_col]], on=ID_COL, how="left")
            df = df.rename(columns={label_col: CLUSTER_COL})
            df = df.merge(df_out[[ID_COL, MORT_COL, VM_COL]], on=ID_COL, how="left")
            df = df[df[CLUSTER_COL].notna()].copy()
            if df.empty:
                print(f"[skip] empty merged data for w{w} ({label_tag})")
                continue
            df[CLUSTER_COL] = df[CLUSTER_COL].astype(int)
            clusters = sorted(df[CLUSTER_COL].dropna().unique())
            print(f"[progress] label={label_tag} window={w}h n_clusters={len(clusters)}")

            (
                weights,
                info,
                pscore,
                x_imputed,
                x_scaled,
                kept_cols,
            ) = _fit_propensity(df, pre_cols, return_model_data=True)
            df["ipw"] = weights
            df["pscore"] = pscore
            df["vfd_28"] = _compute_vfd(df[VM_COL], df[MORT_COL])

            effects = _cluster_effects(df)
            effects["window_hours"] = w
            effects["label_variant"] = label_tag
            effects["propensity_auc"] = info["auc"]
            effects["propensity_p_treat"] = info["p_treat"]
            effects["weights_clip_lo"] = info["weights_clip_lo"]
            effects["weights_clip_hi"] = info["weights_clip_hi"]

            exam_table = _exam_table(df, pre_cols, post_cols, delta_cols, w, direction_overrides)
            exam_summary = _exam_summary(exam_table)

            cluster_summary = effects.merge(exam_summary, on=CLUSTER_COL, how="left")
            cluster_summary["combined_class"] = cluster_summary.apply(
                lambda r: _classify_effect(r["mortality_rate_diff_ipw"], r["vm_mean_diff_ipw"]), axis=1
            )
            cluster_summary["mortality_class_ipw"] = cluster_summary["mortality_rate_diff_ipw"].apply(
                _classify_mortality_only
            )
            cluster_summary["vm_class_ipw"] = cluster_summary["vm_mean_diff_ipw"].apply(_classify_vm_only)
            cluster_summary["combined_class_eps"] = cluster_summary.apply(
                lambda r: _classify_effect_eps(
                    r["mortality_rate_diff_ipw"],
                    r["vm_mean_diff_ipw"],
                    r.get("exam_signal", np.nan),
                ),
                axis=1,
            )
            cluster_summary["mortality_class_eps"] = cluster_summary["mortality_rate_diff_ipw"].apply(
                _classify_mortality_eps
            )
            cluster_summary["vm_class_eps"] = cluster_summary["vm_mean_diff_ipw"].apply(_classify_vm_eps)

            t = df[TREAT_COL].to_numpy(dtype=np.int64, copy=False)
            y_mort = df[MORT_COL].to_numpy(dtype=np.float64, copy=False)
            y_vm = df[VM_COL].to_numpy(dtype=np.float64, copy=False)
            mu1_mort, mu0_mort = _fit_outcome_models(x_scaled, t, y_mort, is_binary=True)
            mu1_vm, mu0_vm = _fit_outcome_models(x_scaled, t, y_vm, is_binary=False)
            hb_series = df.get("hemoglobin_pre_mean")
            hb_strata = hb_series.apply(_hb_stratum) if hb_series is not None else None

            for cl in clusters:
                mask = df[CLUSTER_COL] == cl
                t_c = t[mask]
                if t_c.size == 0:
                    continue
                w_c = weights[mask]
                p_c = pscore[mask]
                x_c = x_imputed[mask]

                smd_unw = _smd_summary(x_c, t_c)
                smd_w = _smd_summary(x_c, t_c, w_c)

                w_q = [float("nan")] * 7
                if w_c.size > 0:
                    q = np.quantile(w_c, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
                    w_q = [float(x) for x in q]

                p_t = p_c[t_c == 1]
                p_n = p_c[t_c == 0]
                p_t_min = float(np.min(p_t)) if p_t.size else float("nan")
                p_t_max = float(np.max(p_t)) if p_t.size else float("nan")
                p_n_min = float(np.min(p_n)) if p_n.size else float("nan")
                p_n_max = float(np.max(p_n)) if p_n.size else float("nan")
                overlap = float(max(0.0, min(p_t_max, p_n_max) - max(p_t_min, p_n_min)))

                all_diag_rows.append(
                    {
                        "window_hours": w,
                        "label_variant": label_tag,
                        "cluster_kmeans": int(cl),
                        "n_total": int(mask.sum()),
                        "n_transfused": int((t_c == 1).sum()),
                        "n_non_transfused": int((t_c == 0).sum()),
                        "n_covariates": int(x_c.shape[1]),
                        "smd_mean_unweighted": smd_unw["smd_mean"],
                        "smd_max_unweighted": smd_unw["smd_max"],
                        "smd_prop_gt_0_1_unweighted": smd_unw["smd_prop_gt_0_1"],
                        "smd_mean_weighted": smd_w["smd_mean"],
                        "smd_max_weighted": smd_w["smd_max"],
                        "smd_prop_gt_0_1_weighted": smd_w["smd_prop_gt_0_1"],
                        "weights_min": w_q[0],
                        "weights_p01": w_q[1],
                        "weights_p05": w_q[2],
                        "weights_p50": w_q[3],
                        "weights_p95": w_q[4],
                        "weights_p99": w_q[5],
                        "weights_max": w_q[6],
                        "pscore_min_t": p_t_min,
                        "pscore_max_t": p_t_max,
                        "pscore_min_c": p_n_min,
                        "pscore_max_c": p_n_max,
                        "pscore_overlap": overlap,
                        "weights_clip_lo": info["weights_clip_lo"],
                        "weights_clip_hi": info["weights_clip_hi"],
                        "pscore_clip_lo": PSCORE_CLIP[0],
                        "pscore_clip_hi": PSCORE_CLIP[1],
                    }
                )

                mask_t = mask & (t == 1)
                mask_n = mask & (t == 0)
                if mask_t.sum() > 0 and mask_n.sum() > 0:
                    mort_boot = _bootstrap_ipw_diff(
                        y_mort[mask_t],
                        weights[mask_t],
                        y_mort[mask_n],
                        weights[mask_n],
                        BOOTSTRAP_N,
                        rng,
                    )
                    vm_boot = _bootstrap_ipw_diff(
                        y_vm[mask_t],
                        weights[mask_t],
                        y_vm[mask_n],
                        weights[mask_n],
                        BOOTSTRAP_N,
                        rng,
                    )
                    all_boot_rows.append(
                        {
                            "window_hours": w,
                            "label_variant": label_tag,
                            "cluster_kmeans": int(cl),
                            "n_transfused": int(mask_t.sum()),
                            "n_non_transfused": int(mask_n.sum()),
                            "mort_diff_mean": mort_boot[0],
                            "mort_ci_lo": mort_boot[1],
                            "mort_ci_hi": mort_boot[2],
                            "mort_pvalue": mort_boot[3],
                            "vm_diff_mean": vm_boot[0],
                            "vm_ci_lo": vm_boot[1],
                            "vm_ci_hi": vm_boot[2],
                            "vm_pvalue": vm_boot[3],
                        }
                    )

                    vfd_t = df.loc[mask_t, "vfd_28"].to_numpy(dtype=np.float64, copy=False)
                    vfd_n = df.loc[mask_n, "vfd_28"].to_numpy(dtype=np.float64, copy=False)
                    vfd_diff = _weighted_mean(vfd_t, weights[mask_t]) - _weighted_mean(vfd_n, weights[mask_n])
                    all_vfd_rows.append(
                        {
                            "window_hours": w,
                            "label_variant": label_tag,
                            "cluster_kmeans": int(cl),
                            "n_transfused": int(mask_t.sum()),
                            "n_non_transfused": int(mask_n.sum()),
                            "vfd_diff_ipw": vfd_diff,
                        }
                    )

                aipw_mort = _aipw_ate(y_mort[mask], t_c, p_c, mu1_mort[mask], mu0_mort[mask])
                aipw_vm = _aipw_ate(y_vm[mask], t_c, p_c, mu1_vm[mask], mu0_vm[mask])
                all_aipw_rows.append(
                    {
                        "window_hours": w,
                        "label_variant": label_tag,
                        "cluster_kmeans": int(cl),
                        "aipw_mort": aipw_mort,
                        "aipw_vm": aipw_vm,
                    }
                )

                match_mort = _nn_match_diff(y_mort[mask], t_c, p_c)
                match_vm = _nn_match_diff(y_vm[mask], t_c, p_c)
                all_match_rows.append(
                    {
                        "window_hours": w,
                        "label_variant": label_tag,
                        "cluster_kmeans": int(cl),
                        "match_mort": match_mort,
                        "match_vm": match_vm,
                    }
                )

                if hb_strata is not None:
                    hb_c = hb_strata[mask]
                    for stratum in hb_c.unique():
                        mask_s = mask & (hb_strata == stratum)
                        mask_ts = mask_s & (t == 1)
                        mask_ns = mask_s & (t == 0)
                        if mask_ts.sum() == 0 or mask_ns.sum() == 0:
                            continue
                        mort_diff = _weighted_mean(y_mort[mask_ts], weights[mask_ts]) - _weighted_mean(
                            y_mort[mask_ns], weights[mask_ns]
                        )
                        vm_diff = _weighted_mean(y_vm[mask_ts], weights[mask_ts]) - _weighted_mean(
                            y_vm[mask_ns], weights[mask_ns]
                        )
                        all_hb_rows.append(
                            {
                                "window_hours": w,
                                "label_variant": label_tag,
                                "cluster_kmeans": int(cl),
                                "hb_stratum": stratum,
                                "n_transfused": int(mask_ts.sum()),
                                "n_non_transfused": int(mask_ns.sum()),
                                "mort_diff_ipw": mort_diff,
                                "vm_diff_ipw": vm_diff,
                            }
                        )
            all_cluster_rows.append(cluster_summary)

            top20 = _top_n_exams(exam_table, TOP_N)
            if not top20.empty:
                top20["label_variant"] = label_tag
                all_top_rows.append(top20)

            if not exam_table.empty:
                exam_table["label_variant"] = label_tag
                all_exam_rows.append(exam_table)

            phenos = _phenotype_top_n(df, pre_cols, w, TOP_N)
            if not phenos.empty:
                phenos["label_variant"] = label_tag
                all_pheno_rows.append(phenos)
            print(f"[progress] label={label_tag} window={w}h done in {time.time() - window_t0:.1f}s")

        if all_cluster_rows:
            cluster_out = pd.concat(all_cluster_rows, ignore_index=True)
            cluster_path = benefit_dir / (
                "tabular_cluster_benefit_summary_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_benefit_summary_all_windows_{label_tag}.csv"
            )
            cluster_out.to_csv(cluster_path, index=False)
            print(f"[save] {cluster_path}")

            summary_cols = [
                "window_hours",
                "label_variant",
                CLUSTER_COL,
                "melhora",
                "piora",
                "sem_mudanca",
                "sem_comparacao",
                "sem_direcao",
                "exam_signal",
            ]
            keep_cols = [c for c in summary_cols if c in cluster_out.columns]
            if keep_cols:
                exam_summary = cluster_out[keep_cols].copy()
                exam_summary_path = exams_dir / (
                    "tabular_cluster_exam_change_summary_all_windows.csv"
                    if label_tag == "best"
                    else f"tabular_cluster_exam_change_summary_all_windows_{label_tag}.csv"
                )
                exam_summary.to_csv(exam_summary_path, index=False)
                print(f"[save] {exam_summary_path}")

        if all_top_rows:
            top_out = pd.concat(all_top_rows, ignore_index=True)
            top_path = exams_dir / (
                "tabular_cluster_top20_exam_changes_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_top20_exam_changes_all_windows_{label_tag}.csv"
            )
            top_out.to_csv(top_path, index=False)
            print(f"[save] {top_path}")

        if all_exam_rows and EXPORT_FULL_EXAM_TABLE:
            exam_out = pd.concat(all_exam_rows, ignore_index=True)
            exam_path = exams_dir / (
                "tabular_cluster_exam_changes_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_exam_changes_all_windows_{label_tag}.csv"
            )
            exam_out.to_csv(exam_path, index=False)
            print(f"[save] {exam_path}")

        if all_pheno_rows:
            pheno_out = pd.concat(all_pheno_rows, ignore_index=True)
            pheno_path = phenotypes_dir / (
                "tabular_cluster_phenotypes_top20_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_phenotypes_top20_all_windows_{label_tag}.csv"
            )
            pheno_out.to_csv(pheno_path, index=False)
            print(f"[save] {pheno_path}")

        if all_boot_rows:
            boot_out = pd.DataFrame(all_boot_rows)
            boot_path = advanced_dir / (
                "tabular_cluster_bootstrap_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_bootstrap_all_windows_{label_tag}.csv"
            )
            boot_out.to_csv(boot_path, index=False)
            print(f"[save] {boot_path}")

        if all_diag_rows:
            diag_out = pd.DataFrame(all_diag_rows)
            diag_path = advanced_dir / (
                "tabular_cluster_ipw_diagnostics_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_ipw_diagnostics_all_windows_{label_tag}.csv"
            )
            diag_out.to_csv(diag_path, index=False)
            print(f"[save] {diag_path}")

        if all_vfd_rows:
            vfd_out = pd.DataFrame(all_vfd_rows)
            vfd_path = advanced_dir / (
                "tabular_cluster_vfd_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_vfd_all_windows_{label_tag}.csv"
            )
            vfd_out.to_csv(vfd_path, index=False)
            print(f"[save] {vfd_path}")

        if all_aipw_rows:
            aipw_out = pd.DataFrame(all_aipw_rows)
            aipw_path = advanced_dir / (
                "tabular_cluster_aipw_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_aipw_all_windows_{label_tag}.csv"
            )
            aipw_out.to_csv(aipw_path, index=False)
            print(f"[save] {aipw_path}")

        if all_match_rows:
            match_out = pd.DataFrame(all_match_rows)
            match_path = advanced_dir / (
                "tabular_cluster_matching_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_matching_all_windows_{label_tag}.csv"
            )
            match_out.to_csv(match_path, index=False)
            print(f"[save] {match_path}")

        if all_hb_rows:
            hb_out = pd.DataFrame(all_hb_rows)
            hb_path = advanced_dir / (
                "tabular_cluster_hb_strata_all_windows.csv"
                if label_tag == "best"
                else f"tabular_cluster_hb_strata_all_windows_{label_tag}.csv"
            )
            hb_out.to_csv(hb_path, index=False)
            print(f"[save] {hb_path}")

        print(f"[progress] label={label_tag} done in {time.time() - label_t0:.1f}s")

    print(f"[done] unified analysis complete in {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
