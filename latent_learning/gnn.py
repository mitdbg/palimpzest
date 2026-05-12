"""
gnn.py
======
GNN encoder-decoder with quality regression head
(currently only supports linear plans, e.g. A->B->C)

Contrastive learning modes:
  "cross"  : compare pairs across both plan types in a shared latent space
  "within" : compare pairs only within the same plan type

Message passing (T iterations, topological order):
  Aggregate : attention(source_hidden) → message
  Update    : GRUCell(message, h_prev) → h_new
  Nodes are processed left to right so each node's updated state is immediately
  available as input for the next node in the same iteration.
Readout (two variants):
  "MPool"   : element-wise max over all node hidden states → MLP → quality
  "CLSOnly" : last node (classifier) hidden state → MLP → quality
"""

from __future__ import annotations

import ast
import os
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEGAL_CSV_ESC_TRAIN = "latent_learning/data/legal/legal_results_grouped_train.csv"
LEGAL_CSV_ESC_TEST  = "latent_learning/data/legal/legal_results_grouped_test.csv"
LEGAL_CSV_ESC       = "latent_learning/data/legal/legal_results_grouped.csv"
LEGAL_CSV_EC        = "latent_learning/data/legal/legal_results_ec_grouped.csv"
PAPER_CSV_TRAIN     = "latent_learning/data/paper/paper_results_grouped_train.csv"
PAPER_CSV_TEST      = "latent_learning/data/paper/paper_results_grouped_test.csv"

BENCHMARK_CSV = "LLM_benchmark/data/benchmarks_features.csv"
DROP_COLS     = ["mmlupro_overall", "cnn_rouge1_f1", "cnn_rouge2_f1", "cnn_rougeL_f1", "cnn_bertscore_f1"]

# Plan type constants
PLAN_ESC = 0  # extractor → summarizer → classifier (3 nodes)
PLAN_EC  = 1  # extractor → classifier (2 nodes)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_plan_label(label: str) -> list[str]:
    parsed = ast.literal_eval(label)
    if isinstance(parsed, str):
        return [parsed]
    return list(parsed)


def build_dataset(
    csv_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load plan-grouped results CSV and build (X, y) tensors.

    Returns:
        X : FloatTensor (N, n_nodes, node_feat_dim)
        y : FloatTensor (N,)
    """
    df_benchmarks = pd.read_csv(BENCHMARK_CSV)
    df_benchmarks.drop(columns=DROP_COLS, inplace=True)
    df_benchmarks.set_index("model", inplace=True)
    all_models = [
        "GPT_5",
        "GPT_5_MINI",
        "o4_MINI",
        "GPT_4_1",
        "GPT_5_NANO",
        "GPT_4_1_MINI",
        "GPT_4o",
        "GPT_4o_MINI",
        "GPT_4_1_NANO",
    ]  # decreasing mmlupro_overall score
    model_to_idx = {m: i for i, m in enumerate(all_models)}

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["classifier_mean_quality"])

    X_list, y_list, one_hot_list = [], [], []
    for _, row in df.iterrows():
        models = parse_plan_label(row["plan_label"])
        plan_one_hot = torch.zeros(len(all_models)*len(models))
        for i, m in enumerate(models):
            plan_one_hot[i*len(all_models)+model_to_idx[m]]=1
        feats = [df_benchmarks.loc[m].values.astype(np.float32) for m in models]
        X_list.append(np.stack(feats))   # (n_nodes, feat_dim)
        y_list.append(float(row["classifier_mean_quality"]))
        one_hot_list.append(plan_one_hot)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    one_hot = torch.tensor(np.stack(one_hot_list), dtype=torch.float32)
    print(f"Dataset ({csv_path}): {X.shape[0]} plans, {X.shape[1]} nodes, feat_dim={X.shape[2]}")
    return X, y #, one_hot


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GNN_Decoder(nn.Module):
    """
    GNN over a variable-length chain: node_0 → node_1 → … → node_{n-1}.

    Parameters
    ----------
    node_feat_dim    : raw benchmark feature dimension
    hidden_dim       : hidden state / message dimension
    n_message_passes : number of message-passing iterations T
    readout          : "MPool" or "CLSOnly"
    """

    def __init__(
        self,
        node_feat_dim,
        hidden_dim,
        n_message_passes,
        readout
    ):
        super().__init__()
        self.n_message_passes = n_message_passes
        self.readout_mode = readout
        self.hidden_dim = hidden_dim
        self.node_feat_dim = node_feat_dim

        #1. Encoder
        # Project raw benchmark features into the hidden space
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        # Message function: MLP that maps a source node's hidden state to a message
        self.message_encode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Aggregate function: attention over incoming messages
        self.agg_encode = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)

        # Update function: shared GRU cell
        self.uppdate_encode = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # 2. Regression head to predict plan quality
        self.reg = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), 1),
            nn.Sigmoid(),
        )


        # 3. Decoder: sequentially reconstruct benchmark features following topological order
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
        self.pred_decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim),
        )

    def decode(self, latent, num_nodes, x_true=None):
        """
        latent  : [B, hidden_dim]
        x_true  : [B, N, node_feat_dim] — ground truth for teacher forcing (optional)
        returns x_reconstructed : [B, N, node_feat_dim]
        """
        B = latent.size(0)
        h = self.start_state_decode(latent)  # [B, H]

        pred_x = []
        hidden_states = []

        for i in range(num_nodes):
            if i == 0:
                context = torch.zeros(B, self.hidden_dim, device=latent.device)
            else:
                prev_h = torch.stack(hidden_states, dim=1)  # [B, i, H]
                prev_x = x_true[:, :i] if x_true is not None else torch.stack(pred_x, dim=1)
                message = self.message_decode(torch.cat([prev_h, prev_x], dim=-1))  # [B, i, H+D]
                query = h.unsqueeze(1)
                context, _ = self.agg_decode(query=query, key=message, value=message)
                context = context.squeeze(1)

            h = self.update_decode(context, h)
            x_i = self.pred_decode(h)
            pred_x.append(x_i)
            hidden_states.append(h)

        return torch.stack(pred_x, dim=1)  # [B, N, D]

    def forward(self, x: torch.Tensor, teacher_forcing: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x               : (B, n_nodes, node_feat_dim)  — n_nodes can be 2 (EC) or 3 (ESC)
        teacher_forcing : if True, feed ground truth into the decoder at each step
        returns : (latent, quality, x_reconstructed)
            latent          : (B, hidden_dim)
            quality         : (B,) in [0, 1]
            x_reconstructed : (B, n_nodes, node_feat_dim)
        """
        B, n_nodes, _ = x.shape
        h = self.node_encoder(x)                        # (B, n_nodes, H)
        h_nodes = [h[:, i] for i in range(n_nodes)]    # list of (B, H)

        for _ in range(self.n_message_passes):
            new_h = []
            for i in range(n_nodes):
                if i == 0:
                    # Source node: no incoming edge, GRU input is zero
                    inp = torch.zeros(B, self.hidden_dim, device=x.device)
                    new_h.append(self.uppdate_encode(inp, h_nodes[0]))
                else:
                    # Use the already-updated predecessor state (left-to-right sequential)
                    msg = self.message_encode(new_h[i - 1])
                    q   = h_nodes[i].unsqueeze(1)   # [B, 1, H]
                    kv  = msg.unsqueeze(1)           # [B, 1, H]
                    agg, _ = self.agg_encode(q, kv, kv)
                    agg = agg.squeeze(1)             # [B, H]
                    new_h.append(self.uppdate_encode(agg, h_nodes[i]))
            h_nodes = new_h

        all_h = torch.stack(h_nodes, dim=1)   # (B, n_nodes, H)
        if self.readout_mode == "MPool":
            latent = all_h.max(dim=1).values   # (B, H)
        elif self.readout_mode == "CLSOnly":
            latent = h_nodes[-1]               # last node = classifier

        quality_pred = self.reg(latent).squeeze(-1)
        x_reconstructed = self.decode(latent, num_nodes=n_nodes, x_true=x if teacher_forcing else None)
        return latent, quality_pred, x_reconstructed


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def soft_contrastive_loss(
    latents: torch.Tensor,
    qualities: torch.Tensor,
    plan_types: torch.Tensor | None = None,
    mode: str = "cross",
    sigma_q: float = 0.05,
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

    Parameters
    ----------
    plan_types : LongTensor (N,)
    mode       : "cross"  — compare all pairs regardless of plan type
                 "within" — exclude pairs from different plan types (requires plan_types)
    """
    N = latents.size(0)
    z       = F.normalize(latents, dim=-1)   # (N, D)
    cos_sim = z @ z.T                        # (N, N)

    qdiff = qualities.unsqueeze(1) - qualities.unsqueeze(0)       # (N, N)
    s     = torch.exp(-(qdiff ** 2) / (2 * sigma_q ** 2))         # (N, N)

    eye = torch.eye(N, device=latents.device).bool()
    s   = s.masked_fill(eye, 0.0)

    same_type = None
    if mode == "within" and plan_types is not None:
        same_type = plan_types.unsqueeze(1) == plan_types.unsqueeze(0)  # (N, N)
        s = s.masked_fill(~same_type, 0.0)

    w = s / s.sum(dim=1, keepdim=True).clamp(min=1e-8)   # (N, N)

    scaled = cos_sim / temperature
    scaled = scaled.masked_fill(eye, float('-inf'))
    if mode == "within":
        # Exclude cross-type plans from the softmax denominator
        scaled = scaled.masked_fill(~same_type, float('-inf'))
    log_softmax_sim = scaled - torch.logsumexp(scaled, dim=1, keepdim=True)  # (N, N)

    loss_matrix = w * log_softmax_sim
    loss_matrix = loss_matrix.masked_fill(eye, 0.0)
    if mode == "within":
        loss_matrix = loss_matrix.masked_fill(~same_type, 0.0)
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
    gamma: float,
    contrastive_mode: str,
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
    contrastive_mode : "cross"  — compare pairs across plan types in shared latent space
                       "within" — compare pairs only within the same plan type
    sigma_q          : Gaussian kernel width for quality similarity
    temperature      : softmax temperature τ for contrastive loss
    lr               : Adam learning rate
    batch_size       : mini-batch size per plan type
    log_every        : print interval (epochs)
    seed             : random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n=== LegalGNN  readout={readout}  T={n_message_passes}  hidden={hidden_dim}  "
          f"epochs={n_epochs}  α={alpha}  β={beta}  γ={gamma} "
          f"contrast={contrastive_mode}  τ={temperature}  σ_q={sigma_q} ===")

    # X_esc_train, y_esc_train, onehot_esc_train = build_legal_dataset(LEGAL_CSV_ESC_TRAIN)
    # X_esc_test,  y_esc_test, onehot_esc_test  = build_legal_dataset(LEGAL_CSV_ESC_TEST)
    X_esc,        y_esc        = build_dataset(LEGAL_CSV_ESC)
    # X_ec,        y_ec        = build_legal_dataset(LEGAL_CSV_EC)
    # X_paper_train, y_paper_train = build_dataset(PAPER_CSV_TRAIN)
    # X_paper_test, y_paper_test = build_dataset(PAPER_CSV_TEST)

    N = X_esc.shape[0]

    loader = DataLoader(
        TensorDataset(X_esc, y_esc),
        batch_size=min(batch_size, N),
        shuffle=True,
    )
    # ec_loader = DataLoader(
    #     TensorDataset(X_ec, y_ec),
    #     batch_size=min(batch_size, N_ec),
    #     shuffle=True,
    # )

    # The shorter loader is cycled so every batch from the longer one has a partner
    # if len(esc_loader) >= len(ec_loader):
    #     n_steps = len(esc_loader)
    #     def make_pairs(): return zip(esc_loader, cycle(ec_loader))
    # else:
    #     n_steps = len(ec_loader)
    #     def make_pairs(): return zip(cycle(esc_loader), ec_loader)

    node_feat_dim = X_esc.shape[2]
    model     = GNN_Decoder(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim,
                             n_message_passes=n_message_passes, readout=readout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"ESC: {N} plans") # steps/epoch: {n_steps}
    print(f"{'Epoch':>6}  {'Reg':>9}  {'Cont':>9}  {'Decode':>9} {'Total':>9}  "
        #   f"{'T_Reg':>9}  {'T_Cont':>9} {'T_Decode':>9} {'T_Total':>9}"
        )
    print("-" * 72)

    best_total = float("inf")
    best_epoch = 0
    best_state = None
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_reg = epoch_cont = epoch_decoder = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()

            # # two-plan loss
            # reg_loss = (len(p_esc)*F.mse_loss(p_esc, yb_esc) +
            #             len(p_ec)*F.mse_loss(p_ec, yb_ec))/(len(p_esc)+len(p_ec))

            # all_latents = torch.cat([l_esc, l_ec], dim=0)
            # all_y       = torch.cat([yb_esc, yb_ec], dim=0)
            # pt          = torch.cat([
            #     torch.full((len(yb_esc),), PLAN_ESC, dtype=torch.long),
            #     torch.full((len(yb_ec),),  PLAN_EC,  dtype=torch.long),
            # ])
            # if contrastive_mode == "cross":
            #     cont_loss = contrastive_loss(
            #         all_latents, all_y, plan_types=pt, mode="cross",
            #         sigma_q=sigma_q, temperature=temperature,
            #     )
            # elif contrastive_mode == "within":
            #     cont_loss = contrastive_loss(
            #         all_latents, all_y, plan_types=pt, mode="within",
            #         sigma_q=sigma_q, temperature=temperature,
            #     )

            # single plan loss
            l, p, recon  = model(xb)
            reg_loss = F.mse_loss(p, yb)
            cont_loss = soft_contrastive_loss(l, yb, sigma_q=sigma_q, temperature=temperature)
            decode_loss = F.mse_loss(recon, xb)
            loss = alpha * reg_loss + beta * 0 + gamma * decode_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_reg  += reg_loss.item()
            epoch_cont += cont_loss.item()
            epoch_decoder += decode_loss.item()

        epoch_reg  /= len(loader) #n_steps
        epoch_cont /= len(loader) #n_steps
        epoch_decoder /= len(loader)
        epoch_total = alpha * epoch_reg + beta * epoch_cont + gamma * epoch_decoder

        # Evaluation on test set
        # model.eval()
        # with torch.no_grad():
        #     t_l, t_p, t_recon  = model(X_paper_test)

        #     t_reg = F.mse_loss(t_p, y_paper_test).item()
        #     t_cont = contrastive_loss(t_l, y_paper_test, sigma_q=sigma_q, temperature=temperature).item()
        #     t_decoder = F.mse_loss(t_recon, X_paper_test).item()

        #     t_total = alpha * t_reg + beta * t_cont + gamma * t_decoder

        if epoch_total < best_total:
            best_total = epoch_total
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0 or epoch == 1:
            print(f"{epoch:>6}  {epoch_reg:>9.4f}  {epoch_cont:>9.4f} {epoch_decoder:>9.6f} {epoch_total:>9.4f}  "
                #   f"{t_reg:>9.4f}  {t_cont:>9.4f}  {t_decoder:>9.6f} {t_total:>9.4f}"
                  + ("  *" if epoch == best_epoch else ""))

        loss_history.append({
            "epoch":           epoch,
            "reg_loss":        epoch_reg,
            "cont_loss":       epoch_cont,
            "decoder_loss":    epoch_decoder,
            "total_loss":      epoch_total,
            # "test_reg_loss":   t_reg,
            # "test_cont_loss":  t_cont,
            # "test_decoder_loss":t_decoder,
            # "test_total_loss": t_total,
        })

    # Save weights and loss curves
    data_dir  = "latent_learning/data/legal"
    data_name = (f"gnnDecoder_{readout}_T{n_message_passes}"
                 f"_({hidden_dim})_({alpha},{beta},{gamma})_({temperature},{sigma_q})")
    save_path = os.path.join(data_dir, f"{data_name}_epoch{best_epoch}-{n_epochs}_weights.pt")
    torch.save(best_state, save_path)
    print(f"\nBest epoch: {best_epoch}/{n_epochs}  (total loss {best_total:.4f})")
    print(f"Weights saved → {save_path}")

    loss_csv_path = os.path.join(data_dir, f"{data_name}_{n_epochs}_loss_history.csv")
    pd.DataFrame(loss_history).to_csv(loss_csv_path, index=False)
    print(f"Loss history saved → {loss_csv_path}")


# ---------------------------------------------------------------------------
# Entry point: train and compare both readout variants
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_EPOCHS         = 5000
    HIDDEN_DIM       = 32
    N_MESSAGE_PASSES = 3
    LR               = 1e-3
    ALPHA            = 1.0
    BETA             = 0.0
    GAMMA            = 1.0
    TEMPERATURE      = 0.5
    SIGMA_Q          = 0.05
    LOG_EVERY        = 500
    READOUT_MODE     = "MPool"   # "MPool" or "CLSOnly"
    CONTRASTIVE_MODE = "cross"   # "cross" or "within"

    model = train(
        readout=READOUT_MODE,
        hidden_dim=HIDDEN_DIM,
        n_message_passes=N_MESSAGE_PASSES,
        n_epochs=N_EPOCHS,
        lr=LR,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        contrastive_mode=CONTRASTIVE_MODE,
        temperature=TEMPERATURE,
        sigma_q=SIGMA_Q,
        log_every=LOG_EVERY,
        batch_size=50,
    )
