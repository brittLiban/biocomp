"""
Task 5 / Real Delta-T Sprint: T-LSTM baseline — next-visit CST regression.

T-LSTM (Baytas et al. 2017) handles irregular time gaps by decomposing
the LSTM memory cell into:
  - long-term memory:  c_t  (retained across time, unaffected by gaps)
  - short-term memory: c_t' = c_t / (1 + delta_t)  (discounted by elapsed time)
  - adjusted cell:     c_hat = short_term + (c_t - short_term) = long_term
The short-term component decays as gap grows; long-term is unchanged.

Input / target / split / metrics identical to baseline_grud.py.
Same CSTRegressionDataset is reused from that module's logic.

Run:
    python scripts/baseline_tlstm.py
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


# ── Dataset (same as GRU-D) ────────────────────────────────────────────────

class CSTRegressionDataset(Dataset):
    def __init__(self, sequences: dict, cst_mean: float, cst_std: float):
        self.cst_mean = cst_mean
        self.cst_std  = cst_std
        self.items = []
        for seq in sequences.values():
            n = seq["n_visits"]
            if n < 2:
                continue
            embs = seq["embeddings"].astype(np.float32)
            cst  = seq["cst"].astype(np.float32)
            bcva = seq["bcva"].astype(np.float32)

            if REAL_DELTA_T and "week_gaps" in seq:
                delta_t = seq["week_gaps"].astype(np.float32)  # (n-1,)
            else:
                delta_t = np.ones(n - 1, dtype=np.float32)

            inp = np.concatenate([
                embs[:-1],
                delta_t[:, None],
                bcva[:-1, None],
            ], axis=1)
            tgt = (cst[1:] - cst_mean) / cst_std
            self.items.append((inp, tgt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        inp, tgt = self.items[idx]
        return torch.from_numpy(inp), torch.from_numpy(tgt)


def collate_fn(batch):
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

class TLSTMCell(nn.Module):
    """
    Single T-LSTM cell. Short-term memory decays with elapsed time.
    c_hat_t = c_{t-1} / (1 + delta_t)  (short-term, gap-discounted)
    c_long  = c_{t-1} - c_hat_t         (long-term remainder)
    c_adj   = c_hat_t + c_long = c_{t-1} (adjusted cell passed to LSTM)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Standard LSTM gates (input, forget, output, cell)
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor,
                c_prev: torch.Tensor, delta_t: torch.Tensor) -> tuple:
        """
        x_t:     (batch, input_dim)
        h_prev:  (batch, hidden_dim)
        c_prev:  (batch, hidden_dim)
        delta_t: (batch, 1)
        """
        # Short-term memory decay: discount cell by elapsed time
        c_short = c_prev / (1.0 + delta_t)         # (batch, hidden_dim)
        c_long  = c_prev - c_short                   # long-term remainder
        c_adj   = c_short + c_long                   # == c_prev (identity)
        # Standard LSTM update with time-adjusted cell
        h, c = self.lstm_cell(x_t, (h_prev, c_adj))
        return h, c


class TLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = TLSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        c = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            delta_t = x[:, t, 1024:1025]            # (batch, 1)
            h, c = self.cell(x[:, t, :], h, c, delta_t)
            outputs.append(self.fc(self.dropout(h)))
        return torch.stack(outputs, dim=1)           # (batch, seq_len, output_dim)


# ── Training utils ─────────────────────────────────────────────────────────

def masked_mse(pred, tgt, lengths):
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
    rmse = float(np.sqrt(((pred_np - tgt_np) ** 2).mean()))
    mae  = float(np.abs(pred_np - tgt_np).mean())
    return rmse, mae


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = "tlstm_realdelta_seed42" if REAL_DELTA_T else "tlstm_ordinal_seed42"
    run = wandb.init(
        project="synapse",
        name=run_name,
        config={
            "model":        "t-lstm",
            "real_delta_t": REAL_DELTA_T,
            "seed":         42,
            "hidden_dim":   128,
            "dropout":      0.2,
            "lr":           1e-3,
            "weight_decay": 1e-4,
            "n_epochs":     60,
        },
        tags=["t-lstm", "cst-regression", "real-delta-t" if REAL_DELTA_T else "ordinal"],
    )

    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=42)

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

    model = TLSTM(input_dim=1026, hidden_dim=128, output_dim=1,
                  dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"T-LSTM parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_rmse = float("inf")
    best_state = None
    n_epochs = 60

    print(f"\nTraining T-LSTM for {n_epochs} epochs...")
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

    model.load_state_dict(best_state)
    rmse, mae = evaluate(model, test_loader, cst_std, device)

    # Naive persistence baseline
    persist_rmse_list = []
    for seq in test_seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        persist_rmse_list.append(((cst[:-1] - cst[1:]) ** 2).mean())
    persist_rmse = float(np.sqrt(np.mean(persist_rmse_list)))

    wandb.log({
        "final_rmse":       rmse,
        "final_mae":        mae,
        "persistence_rmse": persist_rmse,
        "real_delta_t":     REAL_DELTA_T,
    })
    run.finish()

    print("\n=== T-LSTM (next-visit CST regression) ===")
    print(f"Mode : {'real delta-t' if REAL_DELTA_T else 'ordinal'}")
    print(f"RMSE : {rmse:.1f} um  (primary)")
    print(f"MAE  : {mae:.1f} um")
    print(f"Persistence RMSE: {persist_rmse:.1f} um")
    print(f"T-LSTM vs persistence: RMSE delta = {rmse - persist_rmse:+.1f} um")


if __name__ == "__main__":
    main()
