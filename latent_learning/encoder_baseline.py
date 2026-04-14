"""
encoder_onehot.py
=================
Minimal baseline encoder for the paper dataset.

Input features: [mmlu_summarizer/100, mmlu_classifier/100]  — shape (2,)
Goal: learn a latent space where plans with the same quality cluster together
      (contrastive loss → 0) using only the two MMLU scores as input.

The ground-truth pattern: high summarizer MMLU score → quality 1.0.
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


# ---------------------------------------------------------------------------
# Paper pipeline constants
# ---------------------------------------------------------------------------

PAPER_OP_KEYS = ["summarizer", "classifier"]
N_OPS     = len(PAPER_OP_KEYS)   # 2
INPUT_DIM = N_OPS                # 2 — one MMLU score per operator
HIDDEN_DIM = 128
LATENT_DIM = 64


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

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_paper_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "new_paper_results.csv")
    df = pd.read_csv(csv_path, usecols=["plan_label", "quality"])
    df = df.dropna(subset=["quality"])
    df = df[~df["plan_label"].str.startswith("('GPT_5_NANO'")]
    df = df[~df["plan_label"].str.startswith("('GPT_4_1'")]
    df = df[~df["plan_label"].str.endswith("'GPT_4_1_NANO')")]
    return df


def build_dataset(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X : FloatTensor of shape (N, 2)  — [mmlu_sum/100, mmlu_cls/100]
        y : FloatTensor of shape (N,)
    """
    X_list, y_list = [], []
    for _, row in df.iterrows():
        models = parse_plan_label(row["plan_label"])
        feat = np.array(
            [mmlu_score(models[0]) / 100.0,
             mmlu_score(models[1]) / 100.0],
            dtype=np.float32,
        )
        X_list.append(feat)
        y_list.append(float(row["quality"]))
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)  # (N, 2)
    y = torch.tensor(y_list,           dtype=torch.float32)  # (N,)
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BaselinePlanEncoder(nn.Module):
    """
    Minimal baseline: maps (mmlu_summarizer, mmlu_classifier) → latent → quality.

    Input x: (batch, 2)
        x[:, 0] = summarizer MMLU / 100
        x[:, 1] = classifier MMLU / 100
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
        """x: (batch, 2)  →  latent: (batch, latent_dim)"""
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_epochs:  int   = 500,
    margin:    float = 0.5,
    lr:        float = 1e-3,
    alpha:     float = 1.0,
    beta:      float = 1.0,
    log_every: int   = 10,
) -> BaselinePlanEncoder:
    print("Loading paper data...")
    paper_df = load_paper_data()
    print(f"  Paper rows: {len(paper_df)}")

    X, y = build_dataset(paper_df)
    y_simp = torch.where(y < 1.0, torch.tensor(0.33), y)
    print(f"  X shape: {X.shape}")

    torch.manual_seed(42)
    model     = BaselinePlanEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining for {n_epochs} epochs  (margin={margin}, alpha={alpha}, beta={beta}, lr={lr})\n")
    print(f"{'Epoch':>6}  {'Reg Loss':>10}  {'Cont Loss':>10}  {'Total Loss':>10}")
    print("-" * 44)

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        latents, preds = model(X)
        reg_loss  = F.mse_loss(preds, y_simp)
        cont_loss = contrastive_loss(latents, y_simp, margin=margin, debug=(epoch % log_every == 0))
        total     = alpha * reg_loss + beta * cont_loss

        total.backward()
        optimizer.step()

        loss_history.append({
            "epoch":      epoch,
            "reg_loss":   reg_loss.item(),
            "cont_loss":  cont_loss.item(),
            "total_loss": total.item(),
        })

        if epoch % log_every == 0 or epoch == 1:
            print(f"{epoch:>6}  {reg_loss.item():>10.4f}  {cont_loss.item():>10.4f}  {total.item():>10.4f}")

    print("\nTraining complete.")

    weights_path = os.path.join(DATA_DIR, f"extra_clean_(128,64)_baseline_encoder_weights_{n_epochs}_{margin}_{alpha}_{beta}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

    loss_csv_path = os.path.join(DATA_DIR, f"extra_clean_(128,64)_baseline_loss_history_{n_epochs}_{margin}_{alpha}_{beta}.csv")
    pd.DataFrame(loss_history).to_csv(loss_csv_path, index=False)
    print(f"Loss history saved to {loss_csv_path}")

    return model


if __name__ == "__main__":
    num_epochs    = 8000
    log_every     = 200
    custom_margin = 10.0
    alpha         = 1.0
    beta          = 1.0
    trained_model = train(
        n_epochs=num_epochs,
        margin=custom_margin,
        alpha=alpha,
        beta=beta,
        log_every=log_every,
    )
