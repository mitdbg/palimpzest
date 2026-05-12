"""
encoder.py
==========
Encodes physical plans into a latent space for quality prediction.

Architecture:
  - Per-operator input feature x_i = [MMLU-Pro score(s)] ++ text_embedding(operator_prompt)
  - Plan encoder: aggregates per-operator features into a plan embedding
  - Regression head: predicts quality from plan embedding
  - Loss: contrastive (similar-quality plans → similar embeddings) + MSE regression

Data:
  - cleaned_paper_results.csv  : 2-operator pipeline (summarizer → classifier)
  - new_email_results_small.csv: 4-operator pipeline (subject, sender → fraud, internal)

plan_label formats:
  Paper : "('GPT_5_MINI', 'GPT_4_1_NANO')"   — (summarizer_model, classifier_model)
  Email : "('GPT_5_MINI', 'GPT_5_NANO', 'GPT_4_1_NANO', 'GPT_5_MINI')"
          — (subject_model, sender_model, fraud_model, internal_model)
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI

from palimpzest.constants import MODEL_CARDS, Model

# ---------------------------------------------------------------------------
# Add palimpzest to path
# # ---------------------------------------------------------------------------
# REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


NAME_TO_MODEL: dict[str, Model] = {m.name: m for m in Model}
def mmlu_score(model_name: str) -> float:
    """Return the MMLU-Pro 'overall' score for a model given its enum name."""
    model = NAME_TO_MODEL[model_name]
    return MODEL_CARDS[model.value]["overall"]

PAPER_OPERATOR_PROMPTS: dict[str, str] = {
    "summarizer": "The summary of the research paper.",
    "classifier": "'True' if the research paper uses f1 score to measure performance, 'False' otherwise",
}

EMAIL_OPERATOR_PROMPTS: dict[str, str] = {
    "subject_extractor": "The subject of the email",
    "sender_extractor":  "The email address of the email's sender", 
    "fraud_classifier":   "'True' if the email refers to a fraudulent scheme "
            "(i.e., 'Raptor', 'Deathstar', 'Chewco', and/or 'Fat Boy'), 'False' otherwise",
    "internal_classifier":"'True' if the email is not quoting from a news article or an article "
            "written by someone outside of Enron, 'False' otherwise",
}

# Ordered list of operator keys matching the position in plan_label tuples
PAPER_OP_KEYS  = ["summarizer", "classifier"]
EMAIL_OP_KEYS  = ["subject_extractor", "sender_extractor", "fraud_classifier", "internal_classifier"]

def parse_plan_label(label: str) -> list[str]:
    """
    Parse a plan_label string into an ordered list of model name strings.

    Both paper and email labels are stored as Python tuple literals, e.g.
        "('GPT_5_MINI', 'GPT_4_1_NANO')"
    Returns a list of model name strings in operator order.
    """
    parsed = ast.literal_eval(label)
    if isinstance(parsed, str):
        parsed = (parsed,)
    return list(parsed)


# ---------------------------------------------------------------------------
# Text embedding via OpenAI text-embedding-3-small (dim=1536)
# ---------------------------------------------------------------------------

_OPENAI_CLIENT: OpenAI | None = None

def get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()  # reads OPENAI_API_KEY from env
    return _OPENAI_CLIENT
def embed_text(text: str) -> np.ndarray:
    """Return a 1536-dim L2-normalised embedding using text-embedding-3-small."""
    client = get_openai_client()
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    emb = emb / np.linalg.norm(emb)  # L2 normalise
    return emb


# ---------------------------------------------------------------------------
# Pre-computed prompt embeddings (6 total: 2 paper + 4 email)
# Each prompt text is fixed per dataset/task.
# Call precompute_prompt_embeddings() before building any dataset.
# ---------------------------------------------------------------------------

_PROMPT_EMBEDDINGS: dict[str, np.ndarray] = {}

def precompute_prompt_embeddings() -> None:
    """
    Embed each unique operator prompt exactly once (6 API calls total).
    Results are cached in _PROMPT_EMBEDDINGS keyed by op_key.
    """
    all_prompts = {**PAPER_OPERATOR_PROMPTS, **EMAIL_OPERATOR_PROMPTS}
    for op_key, text in all_prompts.items():
        if op_key not in _PROMPT_EMBEDDINGS:
            _PROMPT_EMBEDDINGS[op_key] = embed_text(text)
    print(f"Pre-computed {len(_PROMPT_EMBEDDINGS)} prompt embeddings.")


# ---------------------------------------------------------------------------
# Per-operator feature construction
# ---------------------------------------------------------------------------

def operator_feature(model_name: str, op_key: str) -> np.ndarray:
    """
    Build the input feature vector for a single operator slot.

    x_i = [mmlu_score / 100]  ++  cached_text_embedding(op_key)
         = shape (1 + 1536,)

    Requires precompute_prompt_embeddings() to have been called first.
    """
    score = mmlu_score(model_name) / 100.0
    score_vec = np.array([score], dtype=np.float32)
    prompt_emb = _PROMPT_EMBEDDINGS[op_key]              # (1536,) — reused
    return np.concatenate([score_vec, prompt_emb])       # (1537,)


# ---------------------------------------------------------------------------
# Plan feature construction
# ---------------------------------------------------------------------------

def plan_features(model_names: list[str], op_keys: list[str]) -> np.ndarray:
    """
    Build the concatenated feature matrix for a full plan.

    Returns shape (n_operators, 1537) — one row per operator.
    """
    assert len(model_names) == len(op_keys), (
        f"model count {len(model_names)} ≠ op count {len(op_keys)}"
    )
    rows = [operator_feature(m, k) for m, k in zip(model_names, op_keys)]
    return np.stack(rows, axis=0)  # (n_ops, 1537)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_paper_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "latent_learning_data/new_paper_results.csv")
    df = pd.read_csv(csv_path, usecols=["plan_label", "quality"])
    df = df.dropna(subset=["quality"])
    # drop outliers
    df = df[~df["plan_label"].str.startswith("('GPT_5_NANO'")]
    # df = df[~df["plan_label"].str.startswith("('GPT_4_1'")]
    df = df[~df["plan_label"].str.endswith("'GPT_4_1_NANO')")]
    return df


def load_email_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "new_email_results_small.csv")
    df = pd.read_csv(csv_path, usecols=["plan_label", "quality"])
    df = df.dropna(subset=["quality"])
    return df


# ---------------------------------------------------------------------------
# Build feature tensors for training
# ---------------------------------------------------------------------------

def build_dataset(
    df: pd.DataFrame,
    op_keys: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X : FloatTensor of shape (N, n_ops, 1537)
        y : FloatTensor of shape (N,)

    Requires precompute_prompt_embeddings() to have been called first.
    """
    X_list, y_list = [], []
    for _, row in df.iterrows():
        model_names = parse_plan_label(row["plan_label"])
        feat = plan_features(model_names, op_keys)  # (n_ops, 1537)
        X_list.append(feat)
        y_list.append(float(row["quality"]))
    X = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)  # (N, n_ops, 1537)
    y = torch.tensor(y_list, dtype=torch.float32)                     # (N,)
    return X, y


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

PROMPT_DIM  = 1536  # text-embedding-3-small output dimension
OP_FEAT_DIM = 1537  # 1 (mmlu) + 1536 (prompt embedding) — kept for feature construction
HIDDEN_DIM  = 128
LATENT_DIM  = 64



class PlanEncoder(nn.Module):
    """
    Encodes a plan into a latent embedding using a prompt-guided weighting
    of per-operator MMLU scores.

    Architecture
    ------------
    Input x: (batch, n_ops, 1537)
        x[:, :, 0]   = MMLU score / 100  — the signal to aggregate
        x[:, :, 1:]  = prompt embedding  — used to learn operator importance

    For each operator slot i:
        weight_vec_i = weight_net(prompt_emb_i)   # (batch, n_ops, hidden_dim)

    Plan embedding (latent):
        z = sum_i( mmlu_i * weight_vec_i )         # (batch, hidden_dim)
          projected to latent_dim

    Variable-length plans are supported: the weighted sum over n_ops works
    for any n_ops. Zero-padded operator slots (mmlu=0, prompt=0) contribute
    nothing to the sum and are harmless.
    """

    def __init__(
        self,
        prompt_dim: int = PROMPT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        # maps prompt embedding → weight vector (one per operator slot)
        self.weight_vecs = None  # for debugging only
        self.weight_net = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # project weighted sum to latent space
        self.plan_proj = nn.Sequential(
            # nn.ReLU(), #added to introduce nonlinearity before projection
            nn.Linear(hidden_dim, latent_dim),
            # nn.Tanh(),  # bound latent space to [-1, 1]
            nn.ReLU(),
        )
        # regression head
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_ops, 1537)
        returns latent: (batch, latent_dim)
        """
        mmlu   = x[:, :, :1]    # (batch, n_ops, 1)    — MMLU scores
        prompt = x[:, :, 1:]    # (batch, n_ops, 1536)  — prompt embeddings

        # learn a weight vector per operator slot from its prompt embedding
        self.weight_vecs = self.weight_net(prompt)          # (batch, n_ops, hidden_dim)

        # weighted sum: each operator contributes mmlu_i * weight_vec_i
        plan_repr = (mmlu * self.weight_vecs).sum(dim=1)    # (batch, hidden_dim)

        latent = self.plan_proj(plan_repr)             # (batch, latent_dim)
        return latent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent : (batch, latent_dim)
            quality: (batch,)
        """
        latent  = self.encode(x)
        quality = self.regressor(latent).squeeze(-1)   # (batch,)
        return latent, quality


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def contrastive_loss(
    latents: torch.Tensor,
    qualities: torch.Tensor,
    task_ids: torch.Tensor,
    margin: float = 0.5,
    debug: bool = False,
) -> torch.Tensor:
    """
    Pairwise contrastive loss restricted to same-task pairs.

    Args:
        latents  : (N, latent_dim)
        qualities: (N,) quality scores
        task_ids : (N,) integer task label — pairs from different tasks are ignored
        margin   : hinge margin for dissimilar pairs

    Similarity is defined as exact quality equality (quality_i == quality_j).

    similar   → minimise ||z_i - z_j||²
    dissimilar → maximise it (hinge up to margin)
    """
    # z = F.normalize(latents, dim=-1)  # (N, latent_dim)
    z = latents  # try no normalization

    # pairwise squared Euclidean distances
    diff = z.unsqueeze(1) - z.unsqueeze(0)           # (N, N, latent_dim)
    dist_sq = (diff ** 2).sum(dim=-1)                 # (N, N)

    # same-task mask: only consider pairs where task_ids match
    same_task = (task_ids.unsqueeze(1) == task_ids.unsqueeze(0)).float()  # (N, N)

    # similarity label: exact same quality (within floating-point equality)
    same_quality = (qualities.unsqueeze(1) == qualities.unsqueeze(0)).float()  # (N, N)
    similar    = same_quality * same_task
    dissimilar = (1.0 - same_quality) * same_task

    # mask diagonal
    eye = torch.eye(latents.size(0), device=latents.device)
    similar    = similar    * (1 - eye)
    dissimilar = dissimilar * (1 - eye)

    # debug: look at paper and email separately
    paper_task = (task_ids.unsqueeze(1) == 0).float()
    email_task = (task_ids.unsqueeze(1) == 1).float()
    paper_same = same_quality * paper_task * (1 - eye)
    paper_dis = (1.0 - same_quality) * paper_task * (1 - eye)
    email_same = same_quality * email_task * (1 - eye)
    email_dis = (1.0 - same_quality) * email_task * (1 - eye)

    loss_paper_sim  = (paper_same    * dist_sq).sum()
    loss_paper_dis  = (paper_dis    * F.relu(margin - dist_sq)).sum()
    loss_email_sim  = (email_same    * dist_sq).sum()
    loss_email_dis  = (email_dis    * F.relu(margin - dist_sq)).sum()
    if debug:
        print(f"Contrastive loss (paper): {loss_paper_sim:.4f} (similar) + {loss_paper_dis:.4f} (dissimilar)")
        # print(f"Contrastive loss (email): {loss_email_sim.item():.4f} (similar) + {loss_email_dis.item():.4f} (dissimilar)")

    loss_sim  = (similar    * dist_sq).sum()
    loss_dis  = (dissimilar * F.relu(margin - dist_sq)).sum()
    if debug:
        print(f"Contrastive loss: {loss_sim.item():.4f} (similar) + {loss_dis.item():.4f} (dissimilar)")

    n_pairs = (same_task * (1 - eye)).sum().clamp(min=1)
    return (loss_sim + loss_dis) / n_pairs

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_epochs: int = 500,
    margin: float = 0.5,
    lr: float = 1e-3,
    alpha: float = 1.0,   # weight for regression loss
    beta: float = 1.0,    # weight for contrastive loss
    log_every: int = 10,
) -> PlanEncoder:
    """
    Train PlanEncoder on the combined paper + email datasets.
    All data is used as a single full-batch at each step (no mini-batching),
    which is fine given the small dataset sizes (~81 paper + ~32 email rows).
    """
    print("Loading data...")
    paper_df = load_paper_data()
    email_df = load_email_data()
    print(f"  Paper rows: {len(paper_df)}, Email rows: {len(email_df)}")

    print("Pre-computing 6 prompt embeddings (one API call each)...")
    precompute_prompt_embeddings()

    print("Building feature tensors...")
    X_paper, y_paper = build_dataset(paper_df, PAPER_OP_KEYS)
    y_paper_simp = torch.where(y_paper < 1.0, 0.33, y_paper)  # treat 1/3 and 2/3 as the same for contrastive loss
    X_email, y_email = build_dataset(email_df, EMAIL_OP_KEYS)
    print(f"  X_paper: {X_paper.shape}, X_email: {X_email.shape}")

    # Pad op dimension so paper (n_ops=2) and email (n_ops=4) can share a batch.
    # Paper plans are padded with zeros along the operator axis.
    n_ops_max = max(X_paper.shape[1], X_email.shape[1])
    if X_paper.shape[1] < n_ops_max:
        pad = torch.zeros(X_paper.shape[0], n_ops_max - X_paper.shape[1], X_paper.shape[2])
        X_paper = torch.cat([X_paper, pad], dim=1)

    # task ids: 0 = paper, 1 = email
    task_ids = torch.cat([
        torch.zeros(len(y_paper), dtype=torch.long),
        torch.ones(len(y_email),  dtype=torch.long),
    ])

    # combine into one batch
    X_all = torch.cat([X_paper, X_email], dim=0)   # (N_paper + N_email, n_ops_max, 1537)
    y_all = torch.cat([y_paper, y_email],  dim=0)   # (N_paper + N_email,)

    torch.manual_seed(42)
    model = PlanEncoder(prompt_dim=PROMPT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining for {n_epochs} epochs  (margin = {margin}, alpha={alpha}, beta={beta}, lr={lr})\n")
    print(f"{'Epoch':>6}  {'Reg Loss':>10}  {'Cont Loss':>10}  {'Total Loss':>10}")
    print("-" * 44)

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        if epoch % log_every == 0:
            print(f"summary weight norm: {model.weight_vecs[0, 0, :].norm(dim=-1).item():.4f}, ")
            print(f"classifier weight norm: {model.weight_vecs[0, 1, :].norm(dim=-1).item():.4f}")
            print("weight_vec cos similarity:", F.cosine_similarity(model.weight_vecs[0, 0, :], model.weight_vecs[0, 1, :], dim=-1).item())
        optimizer.zero_grad()

        # latents, preds = model(X_all)
        paper_latents, paper_preds = model(X_paper)

        # reg_loss  = F.mse_loss(preds, y_all)
        # cont_loss = contrastive_loss(latents, y_all, task_ids, margin = margin, debug = epoch%log_every==0)
        reg_loss  = F.mse_loss(paper_preds, y_paper_simp)
        cont_loss = contrastive_loss(paper_latents, y_paper_simp, task_ids[:len(y_paper_simp)], margin = margin, debug = epoch%log_every==0)
        total     = alpha * reg_loss + beta * cont_loss

        total.backward()
        optimizer.step()

        loss_history.append({
            "epoch": epoch,
            "reg_loss": reg_loss.item(),
            "cont_loss": cont_loss.item(),
            "total_loss": total.item(),
        })

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"{epoch:>6}  {reg_loss.item():>10.4f}  "
                f"{cont_loss.item():>10.4f}  {total.item():>10.4f}"
            )

    print("\nTraining complete.")

    save_path = os.path.join(DATA_DIR, f"unnormalized_encoder_weights_{n_epochs}_{margin}_{alpha}_{beta}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Weights saved to {save_path}")

    loss_csv_path = os.path.join(DATA_DIR, f"unnormalized_loss_history_{n_epochs}_{margin}_{alpha}_{beta}.csv")
    pd.DataFrame(loss_history).to_csv(loss_csv_path, index=False)
    print(f"Loss history saved to {loss_csv_path}")

    return model


if __name__ == "__main__":
    num_epochs  = 40000
    log_every   = 1000
    custom_margin = 10.0
    alpha = 1.0
    beta = 1.0
    trained_model = train(n_epochs=num_epochs, margin=custom_margin, alpha=alpha, beta=beta, log_every=log_every)
