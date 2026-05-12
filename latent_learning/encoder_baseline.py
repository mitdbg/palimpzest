"""
Minimal baseline encoder for the paper dataset.

Input features: (benchmark features) per operator 
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from palimpzest.constants import MODEL_CARDS, Model

# ---------------------------------------------------------------------------
# Model lookup helpers
# ---------------------------------------------------------------------------

NAME_TO_MODEL: dict[str, Model] = {m.name: m for m in Model}

def mmlu_score(model_name: str) -> float:
    model = NAME_TO_MODEL[model_name]
    return MODEL_CARDS[model.value]["overall"]

NUM_OP = 3
OP_DIM = 9
INPUT_DIM = NUM_OP * OP_DIM
HIDDEN_DIM = 128
LATENT_DIM = 32


# ---------------------------------------------------------------------------
# Plan label parsing
# ---------------------------------------------------------------------------

def parse_plan_label(label: str) -> list[str]:
    parsed = ast.literal_eval(label)
    if isinstance(parsed, str):
        parsed = (parsed,)
    return list(parsed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join("latent_learning/data/legal/legal_results_grouped.csv")
        # csv_path = os.path.join("new_paper_results.csv")
    df = pd.read_csv(csv_path, usecols=["plan_label", "classifier_mean_quality"])
    # df = pd.read_csv(csv_path, usecols=["plan_label", "quality"])
    # df = df.dropna(subset=["quality"]) #some models failed to complete
    # df = df[~df["plan_label"].str.startswith("('GPT_5_NANO'")]
    # df = df[~df["plan_label"].str.startswith("('GPT_4_1'")] # for extra cleaning only
    # df = df[~df["plan_label"].str.endswith("'GPT_4_1_NANO')")]
    return df


def build_dataset(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X : FloatTensor of shape (N, num_op*op_dim)
        y : FloatTensor of shape (N,)
    """
    df_benchmarks = pd.read_csv("LLM_benchmark/data/benchmarks_features.csv")
    df_benchmarks.drop(columns=["mmlupro_overall", "cnn_rouge1_f1", "cnn_rouge2_f1", "cnn_rougeL_f1", "cnn_bertscore_f1"], inplace=True)
    df_benchmarks.set_index("model", inplace=True)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        models = parse_plan_label(row["plan_label"])
        feat_0 = df_benchmarks.loc[models[0]]
        feat_1 = df_benchmarks.loc[models[1]]
        feat_2 = df_benchmarks.loc[models[2]]
        X_list.append(np.concatenate([feat_0.values, feat_1.values, feat_2.values]))
        y_list.append(float(row["classifier_mean_quality"]))
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)  # (N, num_op*op_dim)
    y = torch.tensor(y_list,           dtype=torch.float32)  # (N,)
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BaselinePlanEncoder(nn.Module):
    """
    Minimal baseline: maps (summarizer, classifier) → latent → quality.

    Input x: (batch, op-dim)
    """

    def __init__(
        self,
        input_dim:  int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.plan_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim)  →  latent: (batch, latent_dim)"""
        return self.plan_proj(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent  = self.encode(x)
        quality = self.regressor(latent).squeeze(-1)
        return latent, quality


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def contrastive_loss(
    latents:   torch.Tensor,
    qualities: torch.Tensor,
    margin:    float = 0.5,
    debug:     bool  = False,
) -> torch.Tensor:
    """
    similar   → minimise ||z_i - z_j||²
    dissimilar → hinge up to margin
    """
    diff    = latents.unsqueeze(1) - latents.unsqueeze(0)   # (N, N, latent_dim)
    dist_sq = (diff ** 2).sum(dim=-1)                        # (N, N)

    same_quality = (qualities.unsqueeze(1) == qualities.unsqueeze(0)).float()
    eye          = torch.eye(latents.size(0), device=latents.device)
    similar      = same_quality      * (1 - eye)
    dissimilar   = (1 - same_quality) * (1 - eye)

    loss_sim = (similar    * dist_sq).sum()
    loss_dis = (dissimilar * F.relu(margin - dist_sq)).sum()

    if debug:
        print(f"  Contrastive: sim={loss_sim.item():.4f}  dis={loss_dis.item():.4f}")

    n_pairs = (1 - eye).sum().clamp(min=1)
    return (loss_sim + loss_dis) / n_pairs


def soft_contrastive_loss(
    latents:     torch.Tensor,
    qualities:   torch.Tensor,
    sigma_q:     float = 0.05,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Soft supervised contrastive loss.

    Quality similarity (Gaussian kernel):
        s_ij = exp(-(qi - qj)^2 / (2 * sigma_q^2))

    Row-normalised weights per anchor i:
        w_ij = s_ij / sum_{m≠i} s_im

    Loss:
        L = -(1/N) * sum_i  sum_{j≠i}  w_ij * log(
                exp(sim(zi, zj) / τ) / sum_{k≠i} exp(sim(zi, zk) / τ) )

    where sim(zi, zj) is cosine similarity.
    """
    N = latents.size(0)
    z       = F.normalize(latents, dim=-1)   # (N, D)
    cos_sim = z @ z.T                        # (N, N)

    qdiff = qualities.unsqueeze(1) - qualities.unsqueeze(0)       # (N, N)
    s     = torch.exp(-(qdiff ** 2) / (2 * sigma_q ** 2))         # (N, N) ∈ (0, 1]

    eye = torch.eye(N, device=latents.device).bool()
    s   = s.masked_fill(eye, 0.0)
    w   = s / s.sum(dim=1, keepdim=True).clamp(min=1e-8)          # (N, N)

    scaled          = cos_sim / temperature
    scaled          = scaled.masked_fill(eye, float('-inf'))
    log_softmax_sim = scaled - torch.logsumexp(scaled, dim=1, keepdim=True)  # (N, N)

    loss_matrix = (w * log_softmax_sim).masked_fill(eye, 0.0)
    return -loss_matrix.sum() / N


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_epochs:    int        = 500,
    lr:          float      = 1e-3,
    alpha:       float      = 1.0,
    beta:        float      = 1.0,
    sigma_q:     float      = 0.05,
    temperature: float      = 0.1,
    #margin: float
    log_every:   int        = 10,
    batch_size:  int | None = None,
) -> BaselinePlanEncoder:
    print("Loading data...")
    df = load_data()
    print(f"  Rows: {len(df)}")

    X, y = build_dataset(df)
    print(f"  X shape: {X.shape}")

    N         = X.size(0)
    use_batch = batch_size is not None and batch_size < N

    torch.manual_seed(42)
    model     = BaselinePlanEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_info = f", batch_size={batch_size}" if use_batch else ""
    print(f"\nTraining for {n_epochs} epochs  "
          f"(alpha={alpha}, beta={beta}, σ_q={sigma_q}, τ={temperature}, lr={lr}{batch_info})\n")
    print(f"{'Epoch':>6}  {'Reg Loss':>10}  {'Cont Loss':>10}  {'Total':>10}")
    print("-" * 44)

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()

        if use_batch:
            perm           = torch.randperm(N)
            X_shuf, y_shuf = X[perm], y[perm]
            epoch_reg = epoch_cont = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                Xb, yb      = X_shuf[start : start + batch_size], y_shuf[start : start + batch_size]
                optimizer.zero_grad()
                latents, preds = model(Xb)
                reg_loss  = F.mse_loss(preds, yb)
                # cont_loss = contrastive_loss(latents, yb, margin)
                cont_loss = soft_contrastive_loss(latents, yb, sigma_q=sigma_q, temperature=temperature)
                total     = alpha * reg_loss + beta * cont_loss
                total.backward()
                optimizer.step()
                epoch_reg  += reg_loss.item()
                epoch_cont += cont_loss.item()
                n_batches  += 1
            reg_loss_val  = epoch_reg  / n_batches
            cont_loss_val = epoch_cont / n_batches
        else:
            optimizer.zero_grad()
            latents, preds = model(X)
            reg_loss  = F.mse_loss(preds, y)
            cont_loss = soft_contrastive_loss(latents, y, sigma_q=sigma_q, temperature=temperature)
            total     = alpha * reg_loss + beta * cont_loss
            total.backward()
            optimizer.step()
            reg_loss_val  = reg_loss.item()
            cont_loss_val = cont_loss.item()

        total_val = alpha * reg_loss_val + beta * cont_loss_val
        loss_history.append({
            "epoch":      epoch,
            "reg_loss":   reg_loss_val,
            "cont_loss":  cont_loss_val,
            "total_loss": total_val,
        })

        if epoch % log_every == 0 or epoch == 1:
            print(f"{epoch:>6}  {reg_loss_val:>10.4f}  {cont_loss_val:>10.4f}  {total_val:>10.4f}")

    print("\nTraining complete.")

    data_dir  = "latent_learning/data/legal"
    data_name = f"baseline_({INPUT_DIM},{HIDDEN_DIM},{LATENT_DIM})_{ALPHA}_{BETA}_{TEMPERATURE}_{SIGMA_Q}"
    weights_path = os.path.join(data_dir, f"{data_name}_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

    loss_csv_path = os.path.join(data_dir, f"{data_name}_loss_history.csv")
    pd.DataFrame(loss_history).to_csv(loss_csv_path, index=False)
    print(f"Loss history saved to {loss_csv_path}")

    return model


if __name__ == "__main__":
    NUM_EPOCHS   = 10000
    LOG_EVERY    = 500
    ALPHA        = 1.0
    BETA         = 0.0
    SIGMA_Q      = 0.05
    TEMPERATURE  = 0.1
    BATCH_SIZE   = 50
    trained_model = train(
        n_epochs=NUM_EPOCHS,
        alpha=ALPHA,
        beta=BETA,
        sigma_q=SIGMA_Q,
        temperature=TEMPERATURE,
        log_every=LOG_EVERY,
        batch_size=BATCH_SIZE,
    )
