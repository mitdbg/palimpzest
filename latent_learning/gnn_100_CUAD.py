"""
gnn_cuad.py
===========
GNN encoder with quality regression head for the 100_CUAD dataset.
(All physical plans are single chains, so the message passing loop still works.)

Node features = [benchmark_features | one_hot_op_type(3) | task_difficulty(1)]
  one_hot_op_type : [is_filter, is_map, is_aggregate]
  task_difficulty : looked up from TASK_DIFFICULTY by provision / field name

Contrastive loss: soft contrastive loss applied only within plans that share
the same logical query_plan string (same provisions + plan type, e.g. 'FFM').

Decoder architecture is retained in the model class but the forward pass only
performs encoding + regression (decode() is never called during training).

Message passing (T iterations, topological order):
  Aggregate : attention(source_hidden) → message
  Update    : GRUCell(message, h_prev) → h_new
Readout:
  "MPool"   : element-wise max over all node hidden states → MLP → quality
  "CLSOnly" : last node hidden state → MLP → quality
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CUAD_TRAIN_CSV = "latent_learning/data/100_CUAD/100_CUAD_train.csv"
CUAD_TEST_CSV  = "latent_learning/data/100_CUAD/100_CUAD_test.csv"
BENCHMARK_CSV  = "LLM_benchmark/data/benchmarks_features.csv"
DROP_COLS      = ["mmlupro_overall", "cnn_rouge1_f1", "cnn_rouge2_f1", "cnn_rougeL_f1", "cnn_bertscore_f1"]

# ---------------------------------------------------------------------------
# Task difficulty scores (provision / field name → score in [0, 1])
# Fill in values before training; missing keys default to 0.0.
# ---------------------------------------------------------------------------

TASK_DIFFICULTY: dict[str, float] = {
    "Agreement Date": 0.15,
    "Governing Law": 0.20,
    "Insurance": 0.25,
    "Audit Rights": 0.30,
    "License Grant": 0.35,
    "Revenue/Profit Sharing": 0.45,
    "Anti-Assignment": 0.50,
    "Minimum Commitment": 0.55,
    "Expiration Date": 0.60,
    "Cap on Liability": 0.65,
    "Post-Termination Services": 0.70,
    "Exclusivity": 0.75,
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _parse_query_plan(label: str) -> tuple[tuple[str, ...], str]:
    """Return (provisions_tuple, plan_type_str) from a query_plan cell."""
    parsed     = ast.literal_eval(label)
    provisions = tuple(parsed[0])
    plan_type  = str(parsed[1])      # e.g. 'FFM', 'FFA'
    return provisions, plan_type


def _parse_physical_plan(label: str) -> list[str]:
    parsed = ast.literal_eval(label)
    return [parsed] if isinstance(parsed, str) else list(parsed)


def _task_features(
    plan_type: str,
    provisions: tuple[str, ...],
    filter_order: int,
) -> list[np.ndarray]:
    """
    One 4-dim vector per node: [is_filter, is_map, is_aggregate, difficulty].
    filter_order=1 swaps the first two provisions (both are filter nodes).
    """
    prov_order = list(provisions)
    if filter_order == 1 and len(prov_order) >= 2:
        prov_order[0], prov_order[1] = prov_order[1], prov_order[0]

    feats = []
    for i, op in enumerate(plan_type):
        if   op == 'F': one_hot = [1.0, 0.0, 0.0]
        elif op == 'M': one_hot = [0.0, 1.0, 0.0]
        elif op == 'A': one_hot = [0.0, 0.0, 1.0]
        else:           one_hot = [0.0, 0.0, 0.0]
        difficulty = TASK_DIFFICULTY.get(prov_order[i], 0.0)
        feats.append(np.array(one_hot + [difficulty], dtype=np.float32))
    return feats


def build_dataset(
    csv_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a 100_CUAD CSV and build tensors.

    Returns
    -------
    X              : FloatTensor (N, n_nodes, node_feat_dim)
                     node_feat_dim = benchmark_dim + 4
    y              : FloatTensor (N,)
    query_plan_ids : LongTensor  (N,)  integer ID per unique logical query_plan
    """
    df_bench = pd.read_csv(BENCHMARK_CSV)
    df_bench.drop(columns=DROP_COLS, inplace=True)
    df_bench.set_index("model", inplace=True)

    df = pd.read_csv(csv_path).dropna(subset=["quality"])

    unique_plans = {qp: idx for idx, qp in enumerate(df["query_plan"].unique())}
    print(unique_plans)

    X_list, y_list, qp_id_list = [], [], []
    for _, row in df.iterrows():
        provisions, plan_type = _parse_query_plan(row["query_plan"])
        models       = _parse_physical_plan(row["physical_plan"])
        filter_order = int(row["filter_order"])

        task_feats  = _task_features(plan_type, provisions, filter_order)
        bench_feats = [df_bench.loc[m].values.astype(np.float32) for m in models]

        node_feats = [
            np.concatenate([bench_feats[i], task_feats[i]])
            for i in range(len(models))
        ]
        X_list.append(np.stack(node_feats))
        y_list.append(float(row["quality"]))
        qp_id_list.append(unique_plans[row["query_plan"]])

    X      = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y      = torch.tensor(y_list,            dtype=torch.float32)
    qp_ids = torch.tensor(qp_id_list,        dtype=torch.long)
    print(f"Dataset ({csv_path}): {X.shape[0]} plans, {X.shape[1]} nodes, "
          f"feat_dim={X.shape[2]} (bench + 4 task)")
    return X, y, qp_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GNN_Decoder(nn.Module):
    """
    GNN over a variable-length chain: node_0 → node_1 → … → node_{n-1}.

    The decoder sub-network is retained but decode() is not called in forward().
    forward() returns only (latent, quality_pred).

    Parameters
    ----------
    node_feat_dim    : raw node feature dimension (benchmark + 4 task features)
    hidden_dim       : hidden state / message dimension
    n_message_passes : number of message-passing iterations T
    readout          : "MPool" or "CLSOnly"
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int,
        n_message_passes: int,
        readout: str,
    ):
        super().__init__()
        self.n_message_passes = n_message_passes
        self.readout_mode     = readout
        self.hidden_dim       = hidden_dim
        self.node_feat_dim    = node_feat_dim

        # 1. Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_encode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.agg_encode     = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.uppdate_encode = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # 2. Regression head
        self.reg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 3. Decoder (retained, not called in forward)
        self.start_state_decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_decode = nn.Sequential(
            nn.Linear(hidden_dim + node_feat_dim, hidden_dim + node_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim + node_feat_dim, hidden_dim + node_feat_dim),
        )
        self.agg_decode = nn.MultiheadAttention(
            num_heads=1,
            embed_dim=hidden_dim,
            kdim=hidden_dim + node_feat_dim,
            vdim=hidden_dim + node_feat_dim,
            batch_first=True,
        )
        self.update_decode = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.pred_decode   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim),
        )

    def decode(
        self,
        latent: torch.Tensor,
        num_nodes: int,
        x_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        latent  : [B, hidden_dim]
        x_true  : [B, N, node_feat_dim]  (optional teacher forcing)
        returns : [B, N, node_feat_dim]
        """
        B = latent.size(0)
        h = self.start_state_decode(latent)

        pred_x, hidden_states = [], []
        for i in range(num_nodes):
            if i == 0:
                context = torch.zeros(B, self.hidden_dim, device=latent.device)
            else:
                prev_h     = torch.stack(hidden_states, dim=1)
                prev_x     = x_true[:, :i] if x_true is not None else torch.stack(pred_x, dim=1)
                message    = self.message_decode(torch.cat([prev_h, prev_x], dim=-1))
                query      = h.unsqueeze(1)
                context, _ = self.agg_decode(query=query, key=message, value=message)
                context    = context.squeeze(1)
            h   = self.update_decode(context, h)
            x_i = self.pred_decode(h)
            pred_x.append(x_i)
            hidden_states.append(h)

        return torch.stack(pred_x, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x       : (B, n_nodes, node_feat_dim)
        returns : (latent, quality_pred)
            latent       : (B, hidden_dim)
            quality_pred : (B,) in [0, 1]
        """
        B, n_nodes, _ = x.shape
        h       = self.node_encoder(x)                      # (B, n_nodes, H)
        h_nodes = [h[:, i] for i in range(n_nodes)]

        for _ in range(self.n_message_passes):
            new_h = []
            for i in range(n_nodes):
                if i == 0:
                    inp = torch.zeros(B, self.hidden_dim, device=x.device)
                    new_h.append(self.uppdate_encode(inp, h_nodes[0]))
                else:
                    msg      = self.message_encode(new_h[i - 1])
                    q        = h_nodes[i].unsqueeze(1)
                    kv       = msg.unsqueeze(1)
                    agg, _   = self.agg_encode(q, kv, kv)
                    agg      = agg.squeeze(1)
                    new_h.append(self.uppdate_encode(agg, h_nodes[i]))
            h_nodes = new_h

        all_h = torch.stack(h_nodes, dim=1)                 # (B, n_nodes, H)
        if self.readout_mode == "MPool":
            latent = all_h.max(dim=1).values
        else:  # CLSOnly
            latent = h_nodes[-1]

        quality_pred = self.reg(latent).squeeze(-1)
        return latent, quality_pred


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def soft_contrastive_loss(
    latents: torch.Tensor,
    qualities: torch.Tensor,
    query_plan_ids: torch.Tensor,
    sigma_q: float = 0.05,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Soft supervised contrastive loss restricted to pairs that share the same
    logical query_plan (identified by integer query_plan_ids).

    Quality similarity (Gaussian kernel):
        s_ij = exp(-(qi - qj)^2 / (2 * sigma_q^2))

    Row-normalised weights (over same-plan, non-self pairs):
        w_ij = s_ij / sum_{m: same_plan, m≠i} s_im

    Loss:
        L = -(1/N) sum_i sum_{j: same_plan, j≠i}
                w_ij * log( exp(sim(zi,zj)/τ) / sum_{k: same_plan, k≠i} exp(sim(zi,zk)/τ) )

    Anchors with no valid partner (unique query_plan in the batch) contribute 0.
    """
    N = latents.size(0)
    z       = F.normalize(latents, dim=-1)
    cos_sim = z @ z.T                                           # (N, N)

    qdiff = qualities.unsqueeze(1) - qualities.unsqueeze(0)
    s     = torch.exp(-(qdiff ** 2) / (2 * sigma_q ** 2))      # (N, N)

    eye       = torch.eye(N, device=latents.device).bool()
    same_plan = query_plan_ids.unsqueeze(1) == query_plan_ids.unsqueeze(0)
    valid     = same_plan & ~eye                                # (N, N)

    s = s.masked_fill(~valid, 0.0)
    w = s / s.sum(dim=1, keepdim=True).clamp(min=1e-8)         # (N, N)

    scaled = cos_sim / temperature
    scaled = scaled.masked_fill(~valid, float('-inf'))

    # Guard against rows with no valid pairs (all -inf → logsumexp = -inf → NaN)
    has_valid       = valid.any(dim=1)                          # (N,)
    safe_scaled     = scaled.clone()
    safe_scaled[~has_valid] = 0.0                              # placeholder row, won't contribute
    log_softmax_sim = safe_scaled - torch.logsumexp(safe_scaled, dim=1, keepdim=True)

    loss_matrix = w * log_softmax_sim
    loss_matrix = loss_matrix.masked_fill(~valid, 0.0)
    loss_matrix = loss_matrix * has_valid.float().unsqueeze(1)
    return -loss_matrix.sum() / N


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    readout: str,
    hidden_dim: int,
    n_message_passes: int,
    n_epochs: int,
    alpha: float,
    beta: float,
    sigma_q: float = 0.05,
    temperature: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 50,
    log_every: int = 50,
    seed: int = 42,
) -> GNN_Decoder:
    """
    Parameters
    ----------
    readout          : "MPool" or "CLSOnly"
    hidden_dim       : hidden state dimension
    n_message_passes : number of message-passing iterations T
    n_epochs         : training epochs
    alpha            : weight for regression (MSE) loss
    beta             : weight for contrastive loss
    sigma_q          : Gaussian kernel width for quality similarity
    temperature      : softmax temperature τ
    lr               : Adam learning rate
    batch_size       : mini-batch size
    log_every        : print interval (epochs)
    seed             : random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n=== CUAD-GNN  readout={readout}  T={n_message_passes}  hidden={hidden_dim}  "
          f"epochs={n_epochs}  α={alpha}  β={beta}  τ={temperature}  σ_q={sigma_q} ===")

    X_train, y_train, qp_train = build_dataset(CUAD_TRAIN_CSV)
    X_test,  y_test,  qp_test  = build_dataset(CUAD_TEST_CSV)

    N      = X_train.shape[0]
    loader = DataLoader(
        TensorDataset(X_train, y_train, qp_train),
        batch_size=min(batch_size, N),
        shuffle=True,
    )

    node_feat_dim = X_train.shape[2]
    model     = GNN_Decoder(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        n_message_passes=n_message_passes,
        readout=readout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Train: {N} plans   Test: {X_test.shape[0]} plans")
    print(f"{'Epoch':>6}  {'Reg':>9}  {'Cont':>9}  {'Total':>9}  "
          f"{'T_Reg':>9}  {'T_Cont':>9}  {'T_Total':>9}")
    print("-" * 80)

    best_total   = float("inf")
    best_epoch   = 0
    best_state   = None
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_reg = epoch_cont = 0.0

        for xb, yb, qpb in loader:
            optimizer.zero_grad()
            l, p      = model(xb)
            reg_loss  = F.mse_loss(p, yb)
            cont_loss = soft_contrastive_loss(l, yb, qpb, sigma_q=sigma_q, temperature=temperature)
            loss      = alpha * reg_loss + beta * cont_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_reg  += reg_loss.item()
            epoch_cont += cont_loss.item()

        epoch_reg  /= len(loader)
        epoch_cont /= len(loader)
        epoch_total = alpha * epoch_reg + beta * epoch_cont

        model.eval()
        with torch.no_grad():
            t_l, t_p = model(X_test)
            t_reg    = F.mse_loss(t_p, y_test).item()
            t_cont   = soft_contrastive_loss(t_l, y_test, qp_test, sigma_q=sigma_q, temperature=temperature).item()
            t_total  = alpha * t_reg + beta * t_cont

        if epoch_total < best_total:
            best_total = epoch_total
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0 or epoch == 1:
            star = "  *" if epoch == best_epoch else ""
            print(f"{epoch:>6}  {epoch_reg:>9.4f}  {epoch_cont:>9.4f}  {epoch_total:>9.4f}  "
                  f"{t_reg:>9.4f}  {t_cont:>9.4f}  {t_total:>9.4f}" + star)

        loss_history.append({
            "epoch":      epoch,
            "reg_loss":   epoch_reg,
            "cont_loss":  epoch_cont,
            "total_loss": epoch_total,
            "test_reg":   t_reg,
            "test_cont":  t_cont,
            "test_total": t_total,
        })

    data_dir  = "latent_learning/data/100_CUAD"
    data_name = (f"gnn_{readout}_T{n_message_passes}"
                 f"_({hidden_dim})_({alpha},{beta})_({temperature},{sigma_q})")
    save_path = os.path.join(data_dir, f"{data_name}_epoch{best_epoch}-{n_epochs}_weights.pt")
    torch.save(best_state, save_path)
    print(f"\nBest epoch: {best_epoch}/{n_epochs}  (total loss {best_total:.4f})")
    print(f"Weights saved → {save_path}")

    loss_csv_path = os.path.join(data_dir, f"{data_name}_{n_epochs}_loss_history.csv")
    pd.DataFrame(loss_history).to_csv(loss_csv_path, index=False)
    print(f"Loss history saved → {loss_csv_path}")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_EPOCHS         = 0
    HIDDEN_DIM       = 32
    N_MESSAGE_PASSES = 3
    LR               = 1e-3
    ALPHA            = 1.0
    BETA             = 1.0
    TEMPERATURE      = 0.5
    SIGMA_Q          = 0.05
    LOG_EVERY        = 500
    READOUT_MODE     = "MPool"   # "MPool" or "CLSOnly"

    model = train(
        readout=READOUT_MODE,
        hidden_dim=HIDDEN_DIM,
        n_message_passes=N_MESSAGE_PASSES,
        n_epochs=N_EPOCHS,
        lr=LR,
        alpha=ALPHA,
        beta=BETA,
        temperature=TEMPERATURE,
        sigma_q=SIGMA_Q,
        log_every=LOG_EVERY,
        batch_size=50,
    )
