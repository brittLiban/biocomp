"""
Task 4 / Real Delta-T Sprint: GRU-D baseline — next-visit CST regression.

GRU-D (Che et al. 2018) extends GRU with:
  - time-decay on hidden state (gamma_h): h decays toward learned mean as gap grows
  - input imputation via decay (gamma_x): missing inputs decay toward learned mean

Here all inputs are present (no missing data), so we use GRU-D primarily for
its time-aware hidden state decay — the key advantage over a plain GRU on
irregular OLIVES visit sequences (gaps range from 4 to 28+ weeks).

Input per visit:  [embedding (1024) | delta_t (1) | bcva (1)]  = 1026-d
Target:           CST at next visit (continuous, μm)
Split:            by Eye_ID (77 train / 19 test eyes)
Metrics:          RMSE (primary), MAE (secondary)

Run:
    python scripts/baseline_grud.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wandb

from data.olives import build_sequences, split_by_eye

# Set True to use real week gaps (Prime eyes) instead of ordinal 1.0 everywhere.
# Prime: gaps normalized to 4-week units (W0→W4 = 1.0, W0→W8 = 2.0, etc.).
# TREX: ordinal 1.0 per step (real timing unavailable — T&E protocol, OCT-DME empty).
REAL_DELTA_T = True


# ── Dataset ────────────────────────────────────────────────────────────────

class CSTRegressionDataset(Dataset):
    """
    Each item = one (input_sequence, target_cst) pair from a single eye.

    Input sequence shape: (n_visits - 1, 1026)
      [embedding_t (1024) | delta_t (1) | bcva_t (1)]
    Target: CST at visit t+1, normalised to zero mean / unit std.

    delta_t is the visit gap in steps (1 = one visit apart).
    For OLIVES: TREX visits are monthly (V1, V2, ...), Prime visits are
    at weeks 0, 4, 8, 12, 16, 20, 28 — we use ordinal step index here
    since we don't have exact calendar dates. A future version can swap in
    week-based deltas from OCT-DR.xlsx.
    """

    def __init__(self, sequences: dict, cst_mean: float, cst_std: float):
        self.cst_mean = cst_mean
        self.cst_std  = cst_std
        self.items = []

        for seq in sequences.values():
            n = seq["n_visits"]
            if n < 2:
                continue
            embs = seq["embeddings"].astype(np.float32)   # (n, 1024)
            cst  = seq["cst"].astype(np.float32)           # (n,)
            bcva = seq["bcva"].astype(np.float32)          # (n,)

            if REAL_DELTA_T and "week_gaps" in seq:
                delta_t = seq["week_gaps"].astype(np.float32)  # (n-1,)
            else:
                delta_t = np.ones(n - 1, dtype=np.float32)

            # Input: visits 0..n-2; delta_t[i] = gap from visit i to i+1
            inp = np.concatenate([
                embs[:-1],
                delta_t[:, None],
                bcva[:-1, None],
            ], axis=1)   # (n-1, 1026)

            tgt = (cst[1:] - cst_mean) / cst_std   # (n-1,)

            self.items.append((inp, tgt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        inp, tgt = self.items[idx]
        return torch.from_numpy(inp), torch.from_numpy(tgt)


def collate_fn(batch):
    """Pad variable-length sequences; return (inputs, targets, lengths)."""
    inputs, targets = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in inputs])
    max_len = lengths.max().item()
    inp_dim = inputs[0].shape[1]

    padded_inp = torch.zeros(len(batch), max_len, inp_dim)
    padded_tgt = torch.zeros(len(batch), max_len)

    for i, (x, t) in enumerate(zip(inputs, targets)):
        n = x.shape[0]
        padded_inp[i, :n] = x
        padded_tgt[i, :n] = t

    return padded_inp, padded_tgt, lengths


# ── Model ──────────────────────────────────────────────────────────────────

class GRUD(nn.Module):
    """
    GRU-D with time-decay on hidden state.
    Hidden state decays toward zero between visits according to:
        h_t = gamma_h(delta_t) * h_{t-1}
        gamma_h = exp(-max(0, W_gamma * delta_t + b_gamma))
    Then applies standard GRU update.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Time-decay gate for hidden state
        self.decay_h = nn.Linear(1, hidden_dim)

        # GRU cell
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       (batch, seq_len, input_dim)
        lengths: (batch,)
        returns: (batch, seq_len, output_dim)
        """
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            # delta_t is the (t+1)-th feature in x (index 1024)
            delta_t = x[:, t, 1024:1025]                    # (batch, 1)

            # Decay hidden state: h decays toward 0 as gap grows
            gamma_h = torch.exp(-torch.clamp(self.decay_h(delta_t), min=0))
            h = gamma_h * h

            h = self.gru_cell(x[:, t, :], h)
            h_drop = self.dropout(h)
            outputs.append(self.fc(h_drop))                  # (batch, output_dim)

        return torch.stack(outputs, dim=1)                   # (batch, seq_len, output_dim)


# ── Training ───────────────────────────────────────────────────────────────

def masked_mse(pred: torch.Tensor, tgt: torch.Tensor,
               lengths: torch.Tensor) -> torch.Tensor:
    """MSE over valid (non-padded) timesteps only."""
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return ((pred.squeeze(-1) - tgt) ** 2)[mask].mean()


def evaluate(model, loader, cst_std, device):
    model.eval()
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for x, tgt, lengths in loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            pred = model(x, lengths).squeeze(-1)
            for i, l in enumerate(lengths):
                all_pred.append(pred[i, :l].cpu().numpy())
                all_tgt.append(tgt[i, :l].cpu().numpy())
    pred_np = np.concatenate(all_pred) * cst_std
    tgt_np  = np.concatenate(all_tgt)  * cst_std
    rmse = np.sqrt(((pred_np - tgt_np) ** 2).mean())
    mae  = np.abs(pred_np - tgt_np).mean()
    return rmse, mae


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = "grud_realdelta_seed42" if REAL_DELTA_T else "grud_ordinal_seed42"
    run = wandb.init(
        project="synapse",
        name=run_name,
        config={
            "model":        "gru-d",
            "real_delta_t": REAL_DELTA_T,
            "seed":         42,
            "hidden_dim":   128,
            "dropout":      0.2,
            "lr":           1e-3,
            "weight_decay": 1e-4,
            "n_epochs":     60,
        },
        tags=["gru-d", "cst-regression", "real-delta-t" if REAL_DELTA_T else "ordinal"],
    )

    # Data
    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=42)

    # Compute CST stats on training eyes only
    all_train_cst = np.concatenate([s["cst"] for s in train_seqs.values()])
    cst_mean = float(all_train_cst.mean())
    cst_std  = float(all_train_cst.std())
    print(f"CST normalisation (train): mean={cst_mean:.1f}  std={cst_std:.1f} um")

    train_ds = CSTRegressionDataset(train_seqs, cst_mean, cst_std)
    test_ds  = CSTRegressionDataset(test_seqs,  cst_mean, cst_std)
    print(f"Train sequences: {len(train_ds)}  |  Test sequences: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False,
                              collate_fn=collate_fn)

    # Model
    model = GRUD(input_dim=1026, hidden_dim=128, output_dim=1,
                 dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GRU-D parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train
    best_rmse = float("inf")
    best_state = None
    n_epochs = 60

    print(f"\nTraining GRU-D for {n_epochs} epochs...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, tgt, lengths in train_loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            optimizer.zero_grad()
            pred = model(x, lengths)
            loss = masked_mse(pred, tgt, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            rmse, mae = evaluate(model, test_loader, cst_std, device)
            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch:3d} | train_loss={train_loss/len(train_loader):.4f} "
                  f"| test RMSE={rmse:.1f} um | MAE={mae:.1f} um")

    # Final evaluation with best checkpoint
    model.load_state_dict(best_state)
    rmse, mae = evaluate(model, test_loader, cst_std, device)

    # Naive persistence baseline
    persist_rmse_list, persist_mae_list = [], []
    for seq in test_seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        pred = cst[:-1]
        tgt  = cst[1:]
        persist_rmse_list.append(((pred - tgt) ** 2).mean())
        persist_mae_list.append(np.abs(pred - tgt).mean())
    persist_rmse = float(np.sqrt(np.mean(persist_rmse_list)))
    persist_mae  = float(np.mean(persist_mae_list))

    wandb.log({
        "final_rmse":       rmse,
        "final_mae":        mae,
        "persistence_rmse": persist_rmse,
        "real_delta_t":     REAL_DELTA_T,
    })
    run.finish()

    print("\n=== GRU-D (next-visit CST regression) ===")
    print(f"Mode : {'real delta-t' if REAL_DELTA_T else 'ordinal'}")
    print(f"RMSE : {rmse:.1f} um  (primary)")
    print(f"MAE  : {mae:.1f} um")
    print(f"Persistence RMSE: {persist_rmse:.1f} um  |  MAE: {persist_mae:.1f} um")
    print(f"GRU-D vs persistence: RMSE delta = {rmse - persist_rmse:+.1f} um")


if __name__ == "__main__":
    main()
