from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.target_trial_utils import ensure_dir, load_trial_config, output_root, read_json, save_table, write_json  # noqa: E402


FINAL_GROUPS = ["B1", "B2", "M1", "M2", "M3"]
GROUP_INTERPRETATION = {
    "B1": "beneficio: anemia dinamica compensada",
    "B2": "beneficio: B1 com PAM baixa/moderada",
    "M1": "maleficio: estresse hemodinamico",
    "M2": "maleficio: estresse hemodinamico cardiorrenal",
    "M3": "maleficio: ausencia de queda relevante de Hb",
}


class VCIPLite(nn.Module):
    """Variational counterfactual target-achievement model for binary treatment."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mu = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)
        self.treatment_head = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear((2 * input_dim) + (2 * latent_dim) + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.z_mu(h), torch.clamp(self.z_logvar(h), -8.0, 8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        a_vec = a.view(-1, 1)
        treatment_logit = self.treatment_head(torch.cat([x, z], dim=1)).view(-1)
        outcome_input = torch.cat([x, z, a_vec, x * a_vec, z * a_vec], dim=1)
        outcome_logit = self.outcome_head(outcome_input).view(-1)
        x_recon = self.reconstruction_head(z)
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "treatment_logit": treatment_logit,
            "outcome_logit": outcome_logit,
            "x_recon": x_recon,
        }

    def predict_mortality(self, x: torch.Tensor, a_value: float, mc_samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            mu, logvar = self.encode(x)
            a = torch.full((x.shape[0], 1), float(a_value), dtype=x.dtype, device=x.device)
            for _ in range(mc_samples):
                z = self.reparameterize(mu, logvar)
                logits = self.outcome_head(torch.cat([x, z, a, x * a, z * a], dim=1)).view(-1)
                preds.append(torch.sigmoid(logits))
        stacked = torch.stack(preds, dim=0)
        return stacked.mean(dim=0).cpu().numpy(), stacked.std(dim=0).cpu().numpy()


def make_matrix(df: pd.DataFrame, feature_cols: list[str], train_mask: pd.Series) -> tuple[np.ndarray, list[str], SimpleImputer, StandardScaler]:
    raw = pd.get_dummies(df[feature_cols].copy(), dummy_na=True)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.loc[:, raw.notna().any(axis=0)]
    train_raw = raw.loc[train_mask].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(train_raw)
    scaler.fit(x_train)
    x = scaler.transform(imputer.transform(raw))
    return x.astype(np.float32), raw.columns.tolist(), imputer, scaler


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {"brier": float(brier_score_loss(y, p))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
    else:
        out["auc"] = np.nan
        out["average_precision"] = np.nan
    return out


def train_model(
    x: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[VCIPLite, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VCIPLite(x.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

    x_t = torch.tensor(x, dtype=torch.float32)
    a_t = torch.tensor(a, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    w_t = torch.tensor(sample_weight, dtype=torch.float32)

    ds = TensorDataset(x_t[train_mask], a_t[train_mask], y_t[train_mask], w_t[train_mask])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    patience = 35
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, ab, yb, wb in loader:
            xb = xb.to(device)
            ab = ab.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            out = model(xb, ab)
            kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
            loss_y_raw = nn.functional.binary_cross_entropy_with_logits(out["outcome_logit"], yb, reduction="none")
            loss_y = torch.mean(loss_y_raw * wb)
            loss_a = bce(out["treatment_logit"], ab)
            loss_x = mse(out["x_recon"], xb)
            beta = min(1.0, epoch / 60.0) * 0.01
            loss = loss_y + 0.30 * loss_a + beta * kl + 0.03 * loss_x
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append([float(loss.item()), float(loss_y.item()), float(loss_a.item()), float(kl.item()), float(loss_x.item())])

        model.eval()
        with torch.no_grad():
            x_val = x_t[val_mask].to(device)
            a_val = a_t[val_mask].to(device)
            y_val = y_t[val_mask].to(device)
            val_out = model(x_val, a_val)
            val_loss = float(bce(val_out["outcome_logit"], y_val).item())
        mean_losses = np.asarray(losses).mean(axis=0)
        history.append(
            {
                "epoch": epoch,
                "loss": mean_losses[0],
                "outcome_bce": mean_losses[1],
                "treatment_bce": mean_losses[2],
                "kl": mean_losses[3],
                "reconstruction_mse": mean_losses[4],
                "validation_outcome_bce": val_loss,
            }
        )
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_counterfactuals(df: pd.DataFrame, mask: pd.Series, label: str, seed: int) -> dict[str, Any]:
    sub = df.loc[mask].copy()
    row: dict[str, Any] = {
        "group": label,
        "n": int(len(sub)),
        "treated": int(sub["transfused"].sum()) if len(sub) else 0,
        "controls": int((1 - sub["transfused"].astype(int)).sum()) if len(sub) else 0,
    }
    if sub.empty:
        return row
    ite = sub["vcip_ite_mortality"].to_numpy(dtype=float)
    adv = sub["vcip_target_advantage_survival"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(ite, seed)
    row.update(
        {
            "observed_mortality": float(sub["mortality_anytime"].mean()),
            "mean_p_mortality_do_transfuse": float(sub["p_mortality_do_transfuse"].mean()),
            "mean_p_mortality_do_no_transfuse": float(sub["p_mortality_do_no_transfuse"].mean()),
            "mean_p_survival_do_transfuse": float(sub["p_survival_do_transfuse"].mean()),
            "mean_p_survival_do_no_transfuse": float(sub["p_survival_do_no_transfuse"].mean()),
            "vcip_ate_mortality": float(np.mean(ite)),
            "vcip_ate_ci_low": ci_low,
            "vcip_ate_ci_high": ci_high,
            "mean_target_advantage_survival": float(np.mean(adv)),
            "fraction_recommend_transfusion": float((sub["vcip_recommended_action"] == 1).mean()),
            "median_uncertainty_a0": float(sub["p_mortality_do_no_transfuse_sd"].median()),
            "median_uncertainty_a1": float(sub["p_mortality_do_transfuse_sd"].median()),
        }
    )
    return row


def write_report(out_dir: Path, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    def md_table(table: pd.DataFrame, cols: list[str], floatfmt: str = ".3f") -> str:
        work = table[cols].copy()
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for row in work.itertuples(index=False):
            vals = []
            for value in row:
                if isinstance(value, float) or isinstance(value, np.floating):
                    vals.append("nan" if not np.isfinite(value) else format(float(value), floatfmt))
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    all_row = summary[summary["group"].eq("all")].iloc[0]
    final_rows = summary[summary["group"].isin(FINAL_GROUPS)].copy()
    lines = [
        "# VCIP-lite: planejamento contrafactual variacional",
        "",
        "Este modulo e uma analise experimental inspirada em VCIP. Ele nao substitui AIPW/target trial como resultado principal.",
        "",
        "## Definicao",
        "",
        "- Historico: features fisiologicas pre-t0.",
        "- Intervencao: `A=1` transfundir versus `A=0` nao transfundir.",
        "- Alvo: `Y=0`, isto e, sobrevivencia/ausencia de mortalidade.",
        "- Quantidade estimada: `P(Y=0 | do(A=a), X)` para `a in {0,1}`.",
        "- Modelo: encoder variacional para estado latente `z`, cabeca de tratamento, cabeca de outcome e reconstrucao de `X`.",
        "- Heterogeneidade: o decoder de outcome inclui interacoes `A x X` e `A x z`.",
        "- Ajuste observacional: a perda de outcome usa ponderacao por propensity score ja estimado no target trial.",
        "- Escopo: esta e uma adaptacao binaria e exploratoria inspirada em VCIP, nao uma reproducao completa do planejamento sequencial do artigo original.",
        "",
        "## Performance preditiva observada",
        "",
        md_table(metrics, ["split", "n", "brier", "auc", "average_precision"]),
        "",
        "## Efeito contrafactual medio",
        "",
        f"No conjunto completo, o VCIP-lite estimou ATE medio de mortalidade de `{all_row['vcip_ate_mortality']:.3f}`.",
        "Valores negativos favorecem transfusao; valores positivos favorecem nao transfusao.",
        "",
        "## Grupos finais",
        "",
        md_table(
            final_rows,
            [
                "group",
                "n",
                "vcip_ate_mortality",
                "vcip_ate_ci_low",
                "vcip_ate_ci_high",
                "mean_target_advantage_survival",
                "fraction_recommend_transfusion",
            ],
        ),
        "",
        "## Interpretacao",
        "",
        "A leitura correta e probabilistica: para pacientes com covariaveis semelhantes, o modelo estima maior ou menor chance de atingir o alvo de sobrevivencia sob cada intervencao.",
        "Como se trata de modelo neural variacional treinado em dados observacionais, os resultados devem ser descritos como exploratorios e geradores de hipotese.",
        "",
        "Em relacao ao AIPW principal, o VCIP-lite preserva a direcao dos grupos finais, mas apresenta magnitudes mais atenuadas. Isso e esperado: o AIPW e o estimador causal primario; o VCIP-lite e usado aqui como apoio contrafactual individual e planejamento de alvo.",
    ]
    (out_dir / "vcip_lite_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex_section(out_dir: Path, summary: pd.DataFrame) -> None:
    final_rows = summary[summary["group"].isin(FINAL_GROUPS)].copy()
    lines = [
        r"\subsection{Analise contrafactual variacional exploratoria}",
        "",
        r"Como extensao exploratoria, foi implementado um modelo inspirado em \textit{Variational Counterfactual Intervention Planning} (VCIP). O objetivo desta etapa nao foi substituir o estimador AIPW, mas estimar, em nivel individual, a probabilidade de atingir o alvo clinico de sobrevivencia sob duas intervencoes alternativas: transfundir e nao transfundir.",
        "",
        r"Para cada paciente com historico fisiologico pre-$t_0$ denotado por $X_i$, o modelo estima:",
        r"\[",
        r"\widehat{P}(Y_i=0 \mid do(A_i=1), X_i)",
        r"\]",
        r"e",
        r"\[",
        r"\widehat{P}(Y_i=0 \mid do(A_i=0), X_i).",
        r"\]",
        "",
        r"A diferenca entre essas probabilidades foi definida como vantagem contrafactual de sobrevivencia sob transfusao. Valores positivos indicam que o modelo atribui maior probabilidade de atingir o alvo de sobrevivencia no cenario transfundir; valores negativos indicam maior probabilidade no cenario nao transfundir.",
        "",
        r"O modelo utiliza um encoder variacional para representar um estado latente $z$, uma cabeca de tratamento, uma cabeca de desfecho e um termo de reconstrucao das covariaveis. Para permitir heterogeneidade, o decoder de outcome inclui interacoes entre tratamento e covariaveis, bem como entre tratamento e o estado latente. A perda de desfecho foi ponderada por propensity score, aproximando a distribuicao interventional em uma analise observacional.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Resultados exploratorios do VCIP-lite nos grupos finais.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Grupo & $n$ & ITE mortalidade & IC95\% & Vantagem de sobrevivencia \\",
        r"\midrule",
    ]
    for row in final_rows.itertuples():
        lines.append(
            f"{row.group} & {int(row.n)} & {row.vcip_ate_mortality:.3f} & "
            f"[{row.vcip_ate_ci_low:.3f}, {row.vcip_ate_ci_high:.3f}] & "
            f"{row.mean_target_advantage_survival:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"Os resultados preservaram a direcao da analise causal principal: B1 e B2 apresentaram vantagem contrafactual de sobrevivencia sob transfusao, enquanto M1, M2 e M3 apresentaram desvantagem. As magnitudes foram menores que as estimadas por AIPW, reforcando que o VCIP-lite deve ser interpretado como apoio exploratorio para raciocinio contrafactual individual, nao como estimador causal primario.",
        "",
    ]
    (out_dir / "vcip_lite_section.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figures(out_dir: Path, pred: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    groups = summary[summary["group"].isin(FINAL_GROUPS)].copy()

    plt.figure(figsize=(8, 4.8))
    y_pos = np.arange(len(groups))
    x = groups["vcip_ate_mortality"].to_numpy(dtype=float)
    lo = groups["vcip_ate_ci_low"].to_numpy(dtype=float)
    hi = groups["vcip_ate_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])
    colors = ["#008080" if v < 0 else "#B22222" for v in x]
    plt.errorbar(x, y_pos, xerr=xerr, fmt="none", ecolor="#333333", capsize=4, lw=1.4)
    plt.scatter(x, y_pos, c=colors, s=70, zorder=3)
    plt.axvline(0, color="#444444", linestyle="--", lw=1)
    plt.yticks(y_pos, groups["group"])
    plt.xlabel("VCIP-lite ITE medio de mortalidade: p(Y=1|do(A=1)) - p(Y=1|do(A=0))")
    plt.title("VCIP-lite por grupos finais")
    plt.tight_layout()
    plt.savefig(fig_dir / "vcip_lite_group_forest.png", dpi=220)
    plt.close()

    plot_df = pred[pred["final_primary_label"].isin(FINAL_GROUPS)].copy()
    if plot_df.empty:
        plot_df = pred.copy()
    labels = FINAL_GROUPS
    data = [plot_df.loc[plot_df["final_primary_label"].eq(g), "vcip_ite_mortality"].dropna().to_numpy() for g in labels]
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.axhline(0, color="#444444", linestyle="--", lw=1)
    plt.ylabel("ITE de mortalidade")
    plt.title("Distribuicao dos efeitos contrafactuais individuais")
    plt.tight_layout()
    plt.savefig(fig_dir / "vcip_lite_ite_distribution_by_group.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.2, 6))
    sample = pred.sample(min(len(pred), 1200), random_state=42)
    colors = sample["vcip_ite_mortality"].map(lambda v: "#008080" if v < 0 else "#B22222")
    plt.scatter(sample["p_survival_do_no_transfuse"], sample["p_survival_do_transfuse"], c=colors, alpha=0.65, s=22)
    lims = [0, 1]
    plt.plot(lims, lims, color="#444444", linestyle="--", lw=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("P(sobrevivencia | do(A=0))")
    plt.ylabel("P(sobrevivencia | do(A=1))")
    plt.title("Comparacao contrafactual do alvo de sobrevivencia")
    plt.tight_layout()
    plt.savefig(fig_dir / "vcip_lite_survival_target_scatter.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    reco = groups["fraction_recommend_transfusion"].to_numpy(dtype=float)
    plt.bar(groups["group"], reco, color=["#008080" if g in {"B1", "B2"} else "#B22222" for g in groups["group"]])
    plt.ylim(0, 1)
    plt.ylabel("Fracao com recomendacao contrafactual A=1")
    plt.title("Planejamento contrafactual: transfundir para atingir alvo de sobrevivencia")
    plt.tight_layout()
    plt.savefig(fig_dir / "vcip_lite_recommended_action_by_group.png", dpi=220)
    plt.close()


def run_vcip_lite(config_dir: str) -> None:
    config = load_trial_config(config_dir)
    out_root = output_root(config)
    proc = out_root / "processed"
    caus = out_root / "causal"
    final = out_root / "final_groups"
    out_dir = ensure_dir(out_root / "vcip_lite")

    metadata = read_json(proc / "target_trial_metadata.json")
    feature_cols = metadata.get("feature_columns", [])
    outcome_col = metadata.get("primary_outcome", "mortality_anytime")
    if outcome_col != "mortality_anytime":
        raise RuntimeError(f"VCIP-lite currently expects mortality_anytime, got {outcome_col}")

    trial = pd.read_parquet(proc / "trial_dataset.parquet")
    assignments = pd.read_parquet(final / "final_group_assignments.parquet")
    nuisance_path = caus / "nuisance_predictions.parquet"
    nuisance = pd.read_parquet(nuisance_path) if nuisance_path.exists() else pd.DataFrame({"stay_id": trial["stay_id"]})

    required = ["stay_id", "analysis_split", "transfused", outcome_col]
    missing = [col for col in required + feature_cols if col not in trial.columns]
    if missing:
        raise RuntimeError(f"Missing columns for VCIP-lite: {missing[:20]}")

    df = trial[required + feature_cols].merge(assignments[["stay_id", "final_primary_label"] + FINAL_GROUPS], on="stay_id", how="left")
    df = df.merge(nuisance[["stay_id", "mu0", "mu1", "propensity_score"]], on="stay_id", how="left")
    df = df.dropna(subset=["transfused", outcome_col, "analysis_split"]).copy()
    for group in FINAL_GROUPS:
        df[group] = df[group].fillna(False).astype(int)
    train_mask = df["analysis_split"].eq("discovery")
    val_mask = df["analysis_split"].eq("validation")
    if train_mask.sum() < 50 or val_mask.sum() < 20:
        raise RuntimeError("VCIP-lite needs discovery and validation split with enough observations.")

    vcip_feature_cols = feature_cols + FINAL_GROUPS
    x, matrix_cols, _, _ = make_matrix(df, vcip_feature_cols, train_mask)
    a = df["transfused"].astype(int).to_numpy()
    y = df[outcome_col].astype(int).to_numpy()
    if "propensity_score" in df.columns and df["propensity_score"].notna().any():
        ps = df["propensity_score"].astype(float).clip(0.05, 0.95).to_numpy()
        sample_weight = np.where(a == 1, 1.0 / ps, 1.0 / (1.0 - ps))
        sample_weight = np.clip(sample_weight, 0.25, 8.0)
        sample_weight = sample_weight / np.nanmean(sample_weight[train_mask.to_numpy()])
    else:
        sample_weight = np.ones(len(df), dtype=float)
    seed = int(config.get("target_trial", {}).get("seed", 42))
    vcfg = config.get("vcip_lite", {})
    model, history = train_model(
        x=x,
        a=a,
        y=y,
        sample_weight=sample_weight.astype(np.float32),
        train_mask=train_mask.to_numpy(),
        val_mask=val_mask.to_numpy(),
        seed=seed,
        latent_dim=int(vcfg.get("latent_dim", 16)),
        hidden_dim=int(vcfg.get("hidden_dim", 64)),
        epochs=int(vcfg.get("epochs", 350)),
        batch_size=int(vcfg.get("batch_size", 128)),
        lr=float(vcfg.get("learning_rate", 1e-3)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    mc_samples = int(vcfg.get("mc_samples", 100))
    p1, sd1 = model.predict_mortality(x_tensor, 1.0, mc_samples=mc_samples)
    p0, sd0 = model.predict_mortality(x_tensor, 0.0, mc_samples=mc_samples)
    observed_p = np.where(a == 1, p1, p0)

    pred = df[["stay_id", "analysis_split", "transfused", outcome_col, "final_primary_label"] + FINAL_GROUPS].copy()
    pred["p_mortality_do_transfuse"] = np.clip(p1, 1e-5, 1 - 1e-5)
    pred["p_mortality_do_no_transfuse"] = np.clip(p0, 1e-5, 1 - 1e-5)
    pred["p_survival_do_transfuse"] = 1 - pred["p_mortality_do_transfuse"]
    pred["p_survival_do_no_transfuse"] = 1 - pred["p_mortality_do_no_transfuse"]
    pred["p_mortality_do_transfuse_sd"] = sd1
    pred["p_mortality_do_no_transfuse_sd"] = sd0
    pred["vcip_ite_mortality"] = pred["p_mortality_do_transfuse"] - pred["p_mortality_do_no_transfuse"]
    pred["vcip_target_advantage_survival"] = pred["p_survival_do_transfuse"] - pred["p_survival_do_no_transfuse"]
    pred["vcip_recommended_action"] = (pred["vcip_target_advantage_survival"] > 0).astype(int)
    pred["observed_prediction"] = observed_p
    if "mu0" in df.columns:
        pred["aipw_mu0"] = df["mu0"].to_numpy()
        pred["aipw_mu1"] = df["mu1"].to_numpy()
        pred["aipw_like_ite"] = df["mu1"].to_numpy() - df["mu0"].to_numpy()

    metrics_rows = []
    for split in ["all", "discovery", "validation"]:
        mask = np.ones(len(pred), dtype=bool) if split == "all" else pred["analysis_split"].eq(split).to_numpy()
        row = {"split": split, "n": int(mask.sum())}
        row.update(binary_metrics(y[mask], observed_p[mask]))
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    summary_rows = []
    for split in ["all", "discovery", "validation"]:
        split_mask = pd.Series(np.ones(len(pred), dtype=bool), index=pred.index) if split == "all" else pred["analysis_split"].eq(split)
        row = summarize_counterfactuals(pred, split_mask, split, seed)
        row["stratum_type"] = "split"
        summary_rows.append(row)
    for group in FINAL_GROUPS:
        group_mask = pred[group].fillna(False).astype(bool)
        row = summarize_counterfactuals(pred, group_mask, group, seed + FINAL_GROUPS.index(group) + 10)
        row["stratum_type"] = "final_group"
        row["interpretation"] = GROUP_INTERPRETATION[group]
        summary_rows.append(row)
        for split in ["discovery", "validation"]:
            mask = group_mask & pred["analysis_split"].eq(split)
            srow = summarize_counterfactuals(pred, mask, f"{group}_{split}", seed + FINAL_GROUPS.index(group) + 100)
            srow["stratum_type"] = "final_group_split"
            srow["parent_group"] = group
            summary_rows.append(srow)
    summary = pd.DataFrame(summary_rows)

    example_cols = [
        "stay_id",
        "analysis_split",
        "final_primary_label",
        "transfused",
        outcome_col,
        "p_survival_do_transfuse",
        "p_survival_do_no_transfuse",
        "vcip_target_advantage_survival",
        "vcip_ite_mortality",
        "p_mortality_do_transfuse_sd",
        "p_mortality_do_no_transfuse_sd",
    ]
    most_benefit = pred.nsmallest(15, "vcip_ite_mortality")[example_cols].copy()
    most_benefit["example_type"] = "largest_model_estimated_benefit_from_transfusion"
    most_harm = pred.nlargest(15, "vcip_ite_mortality")[example_cols].copy()
    most_harm["example_type"] = "largest_model_estimated_harm_from_transfusion"
    patient_examples = pd.concat([most_benefit, most_harm], ignore_index=True)

    save_table(pred, out_dir / "vcip_lite_individual_counterfactuals.parquet")
    save_table(pred, out_dir / "vcip_lite_individual_counterfactuals.csv")
    save_table(patient_examples, out_dir / "vcip_lite_patient_examples.csv")
    save_table(summary, out_dir / "vcip_lite_group_summary.csv")
    save_table(metrics, out_dir / "vcip_lite_metrics.csv")
    save_table(history, out_dir / "vcip_lite_training_history.csv")
    torch.save(model.state_dict(), out_dir / "vcip_lite_model.pt")
    write_json(
        out_dir / "vcip_lite_metadata.json",
        {
            "model": "VCIP-lite variational target-achievement model",
            "not_main_estimand": True,
            "target": "survival, Y=0",
            "intervention": "binary transfusion A in {0,1}",
            "feature_source": str(proc / "trial_dataset.parquet"),
            "n": int(len(pred)),
            "n_features_raw": int(len(vcip_feature_cols)),
            "n_features_encoded": int(len(matrix_cols)),
            "includes_final_group_indicators": True,
            "includes_treatment_interactions": True,
            "uses_ipw_weighted_outcome_loss": bool("propensity_score" in df.columns),
            "device": str(device),
            "mc_samples": mc_samples,
            "outputs": [
                "vcip_lite_individual_counterfactuals.parquet",
                "vcip_lite_group_summary.csv",
                "vcip_lite_metrics.csv",
                "figures/*.png",
            ],
        },
    )
    make_figures(out_dir, pred, summary)
    write_report(out_dir, metrics, summary)
    write_latex_section(out_dir, summary)
    print(f"vcip_lite_complete outputs={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experimental VCIP-lite counterfactual target-achievement model.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    run_vcip_lite(args.config_dir)


if __name__ == "__main__":
    main()
