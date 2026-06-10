"""
Retraining with proper train/val/test split for checkpoint selection.

Fixes test-set leakage: checkpoint selection now uses validation RMSE.
Test set is touched exactly once, at the very end with the locked checkpoint.

Split (96 eyes total):
  - 19 test eyes:  split_by_eye(seqs, test_frac=0.2, seed=42)  — UNCHANGED vs prior runs
  - 60 train eyes: split_by_eye(trainval, test_frac=0.22, seed=42) takes first ~17 as val
  - 17 val eyes:   same call above

Runs all 6 combinations: GRU-D, T-LSTM, ODE-RNN × ordinal + real-dt
Logs each to W&B (project=synapse) with "_valsplit" in the run name.
Prints a summary table of new test RMSE numbers at the end.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

from data.olives import build_sequences, split_by_eye
from dynamics.latent_ode import LatentODE


# ── Shared dataset ─────────────────────────────────────────────────────────

class CSTRegressionDataset(Dataset):
    def __init__(self, sequences: dict, cst_mean: float, cst_std: float,
                 real_delta_t: bool = False):
        self.cst_mean = cst_mean
        self.cst_std  = cst_std
        self.items    = []
        for seq in sequences.values():
            n = seq["n_visits"]
            if n < 2:
                continue
            embs = seq["embeddings"].astype(np.float32)
            cst  = seq["cst"].astype(np.float32)
            bcva = seq["bcva"].astype(np.float32)
            if real_delta_t and "week_gaps" in seq:
                delta_t = seq["week_gaps"].astype(np.float32)
            else:
                delta_t = np.ones(n - 1, dtype=np.float32)
            inp = np.concatenate([embs[:-1], delta_t[:, None], bcva[:-1, None]], axis=1)
            tgt = (cst[1:] - cst_mean) / cst_std
            self.items.append((inp, tgt))

    def __len__(self): return len(self.items)
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


# ── Models ─────────────────────────────────────────────────────────────────

class GRUD(nn.Module):
    def __init__(self, input_dim=1026, hidden_dim=128, output_dim=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay_h  = nn.Linear(1, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            delta_t = x[:, t, 1024:1025]
            gamma_h = torch.exp(-torch.clamp(self.decay_h(delta_t), min=0))
            h = gamma_h * h
            h = self.gru_cell(x[:, t, :], h)
            outputs.append(self.fc(self.dropout(h)))
        return torch.stack(outputs, dim=1)


class TLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, x_t, h_prev, c_prev, delta_t):
        c_short = c_prev / (1.0 + delta_t)
        c_long  = c_prev - c_short
        c_adj   = c_short + c_long
        return self.lstm_cell(x_t, (h_prev, c_adj))


class TLSTM(nn.Module):
    def __init__(self, input_dim=1026, hidden_dim=128, output_dim=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell       = TLSTMCell(input_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        c = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            delta_t = x[:, t, 1024:1025]
            h, c = self.cell(x[:, t, :], h, c, delta_t)
            outputs.append(self.fc(self.dropout(h)))
        return torch.stack(outputs, dim=1)


# ── Shared training utils ──────────────────────────────────────────────────

def masked_mse(pred, tgt, lengths):
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return ((pred.squeeze(-1) - tgt) ** 2)[mask].mean()


def evaluate_recurrent(model, loader, cst_std, device):
    """Returns (ts_wt_rmse, mae, eye_wt_rmse)."""
    model.eval()
    all_pred, all_tgt = [], []
    eye_sq_errs = []   # one mean-sq-error per eye (for eye-weighted RMSE)
    with torch.no_grad():
        for x, tgt, lengths in loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            pred = model(x, lengths).squeeze(-1)
            for i, l in enumerate(lengths):
                p = pred[i, :l].cpu().numpy() * cst_std
                t = tgt[i, :l].cpu().numpy() * cst_std
                all_pred.append(p)
                all_tgt.append(t)
                eye_sq_errs.append(float(((p - t) ** 2).mean()))
    pred_np = np.concatenate(all_pred)
    tgt_np  = np.concatenate(all_tgt)
    ts_wt_rmse  = float(np.sqrt(((pred_np - tgt_np) ** 2).mean()))
    mae         = float(np.abs(pred_np - tgt_np).mean())
    eye_wt_rmse = float(np.sqrt(np.mean(eye_sq_errs)))
    return ts_wt_rmse, mae, eye_wt_rmse


def evaluate_ode(model, loader, cst_std, device, real_delta_t):
    """Returns (ts_wt_rmse, mae, eye_wt_rmse)."""
    model.eval()
    all_pred, all_tgt = [], []
    eye_sq_errs = []
    with torch.no_grad():
        for x, tgt, lengths in loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            delta_t_seq = x[:, :, 1024] if real_delta_t else None
            pred = model(x[:, :, :1024], lengths, delta_t_seq=delta_t_seq).squeeze(-1)
            for i, l in enumerate(lengths):
                p = pred[i, :l].cpu().numpy() * cst_std
                t = tgt[i, :l].cpu().numpy() * cst_std
                all_pred.append(p)
                all_tgt.append(t)
                eye_sq_errs.append(float(((p - t) ** 2).mean()))
    pred_np = np.concatenate(all_pred)
    tgt_np  = np.concatenate(all_tgt)
    ts_wt_rmse  = float(np.sqrt(((pred_np - tgt_np) ** 2).mean()))
    mae         = float(np.abs(pred_np - tgt_np).mean())
    eye_wt_rmse = float(np.sqrt(np.mean(eye_sq_errs)))
    return ts_wt_rmse, mae, eye_wt_rmse


def persistence_rmse_eye_wt(seqs):
    """Eye-weighted persistence RMSE (each patient counts equally)."""
    per_eye = []
    for seq in seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        per_eye.append(((cst[:-1] - cst[1:]) ** 2).mean())
    return float(np.sqrt(np.mean(per_eye)))


def persistence_rmse_ts_wt(seqs):
    """Timestep-weighted persistence RMSE (each visit counts equally)."""
    sq_err = []
    for seq in seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        sq_err.extend(((cst[:-1] - cst[1:]) ** 2).tolist())
    return float(np.sqrt(np.mean(sq_err)))


# ── Training loop — recurrent models ──────────────────────────────────────

def train_recurrent(model_cls, model_name, real_delta_t,
                    train_seqs, val_seqs, test_seqs, cst_mean, cst_std,
                    device, n_epochs=60, hidden_dim=128, dropout=0.2,
                    lr=1e-3, wd=1e-4, batch_size=16, sched_step=20):
    timing = "realdelta" if real_delta_t else "ordinal"
    run_name = f"{model_name}_{timing}_valsplit_seed42"
    print(f"\n{'='*60}")
    print(f"  Training {run_name}")
    print(f"{'='*60}")

    run = wandb.init(
        project="synapse",
        name=run_name,
        config={"model": model_name, "real_delta_t": real_delta_t,
                "seed": 42, "hidden_dim": hidden_dim, "dropout": dropout,
                "lr": lr, "weight_decay": wd, "n_epochs": n_epochs,
                "val_split": True, "train_eyes": len(train_seqs),
                "val_eyes": len(val_seqs), "test_eyes": len(test_seqs)},
        tags=[model_name, "cst-regression", "valsplit",
              "real-delta-t" if real_delta_t else "ordinal"],
    )

    train_ds = CSTRegressionDataset(train_seqs, cst_mean, cst_std, real_delta_t)
    val_ds   = CSTRegressionDataset(val_seqs,   cst_mean, cst_std, real_delta_t)
    test_ds  = CSTRegressionDataset(test_seqs,  cst_mean, cst_std, real_delta_t)
    print(f"  Train: {len(train_ds)} eyes  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = model_cls(input_dim=1026, hidden_dim=hidden_dim, output_dim=1, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=0.5)

    best_val_rmse = float("inf")
    best_state    = None

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
            val_rmse, val_mae, _ = evaluate_recurrent(model, val_loader, cst_std, device)
            is_best = val_rmse < best_val_rmse
            if is_best:
                best_val_rmse = val_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wandb.log({"val_rmse": val_rmse, "val_mae": val_mae,
                       "train_loss": train_loss / len(train_loader), "epoch": epoch})
            print(f"  Epoch {epoch:3d} | loss={train_loss/len(train_loader):.4f} "
                  f"| val RMSE={val_rmse:.1f} um {'<-- best' if is_best else ''}")

    # Final test evaluation — exactly once, locked checkpoint
    model.load_state_dict(best_state)
    test_rmse, test_mae, test_eyewt = evaluate_recurrent(model, test_loader, cst_std, device)
    wandb.log({"test_rmse": test_rmse, "test_mae": test_mae,
               "test_rmse_eyewt": test_eyewt, "best_val_rmse": best_val_rmse})
    run.finish()
    print(f"  FINAL TEST RMSE ts-wt: {test_rmse:.2f} um  eye-wt: {test_eyewt:.2f} um  MAE: {test_mae:.2f} um")
    return test_rmse, test_mae, test_eyewt


# ── Training loop — ODE-RNN ────────────────────────────────────────────────

def train_ode(real_delta_t, train_seqs, val_seqs, test_seqs, cst_mean, cst_std,
              device, n_epochs=100, latent_dim=32, ode_hidden=64, dropout=0.2,
              lr=1e-3, wd=1e-4, batch_size=16, sched_step=30, rtol=1e-3, atol=1e-4):
    timing   = "realdelta" if real_delta_t else "ordinal"
    run_name = f"ode_{timing}_valsplit_seed42"
    print(f"\n{'='*60}")
    print(f"  Training {run_name}")
    print(f"{'='*60}")

    run = wandb.init(
        project="synapse",
        name=run_name,
        config={"model": "ode-rnn", "real_delta_t": real_delta_t,
                "seed": 42, "latent_dim": latent_dim, "ode_hidden": ode_hidden,
                "dropout": dropout, "lr": lr, "weight_decay": wd,
                "n_epochs": n_epochs, "val_split": True,
                "train_eyes": len(train_seqs), "val_eyes": len(val_seqs),
                "test_eyes": len(test_seqs)},
        tags=["ode-rnn", "cst-regression", "valsplit",
              "real-delta-t" if real_delta_t else "ordinal"],
    )

    train_ds = CSTRegressionDataset(train_seqs, cst_mean, cst_std, real_delta_t)
    val_ds   = CSTRegressionDataset(val_seqs,   cst_mean, cst_std, real_delta_t)
    test_ds  = CSTRegressionDataset(test_seqs,  cst_mean, cst_std, real_delta_t)
    print(f"  Train: {len(train_ds)} eyes  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LatentODE(emb_dim=1024, latent_dim=latent_dim, ode_hidden=ode_hidden,
                      dropout=dropout, rtol=rtol, atol=atol).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=0.5)

    best_val_rmse = float("inf")
    best_state    = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, tgt, lengths in train_loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            optimizer.zero_grad()
            delta_t_seq = x[:, :, 1024] if real_delta_t else None
            pred = model(x[:, :, :1024], lengths, delta_t_seq=delta_t_seq)
            loss = masked_mse(pred, tgt, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            val_rmse, val_mae, _ = evaluate_ode(model, val_loader, cst_std, device, real_delta_t)
            is_best = val_rmse < best_val_rmse
            if is_best:
                best_val_rmse = val_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wandb.log({"val_rmse": val_rmse, "val_mae": val_mae,
                       "train_loss": train_loss / len(train_loader), "epoch": epoch})
            print(f"  Epoch {epoch:3d} | loss={train_loss/len(train_loader):.4f} "
                  f"| val RMSE={val_rmse:.1f} um {'<-- best' if is_best else ''}")

    # Final test evaluation — exactly once, locked checkpoint
    model.load_state_dict(best_state)
    test_rmse, test_mae, test_eyewt = evaluate_ode(model, test_loader, cst_std, device, real_delta_t)
    wandb.log({"test_rmse": test_rmse, "test_mae": test_mae,
               "test_rmse_eyewt": test_eyewt, "best_val_rmse": best_val_rmse})
    run.finish()
    print(f"  FINAL TEST RMSE ts-wt: {test_rmse:.2f} um  eye-wt: {test_eyewt:.2f} um  MAE: {test_mae:.2f} um")
    return test_rmse, test_mae, test_eyewt


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data and build splits — test eyes IDENTICAL to all prior runs
    seqs = build_sequences()
    trainval_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=42)   # 77 + 19
    train_seqs, val_seqs     = split_by_eye(trainval_seqs, test_frac=0.22, seed=42)  # 60 + 17
    print(f"\nSplit: {len(train_seqs)} train  {len(val_seqs)} val  {len(test_seqs)} test")

    # CST normalisation from TRAIN only (not trainval)
    all_train_cst = np.concatenate([s["cst"] for s in train_seqs.values()])
    cst_mean = float(all_train_cst.mean())
    cst_std  = float(all_train_cst.std())
    print(f"CST normalisation (train only): mean={cst_mean:.1f}  std={cst_std:.1f} um")

    # Persistence baseline on test set (unaffected by checkpoint selection)
    p_eyewt = persistence_rmse_eye_wt(test_seqs)
    p_tswt  = persistence_rmse_ts_wt(test_seqs)
    print(f"Persistence RMSE: eye-wt={p_eyewt:.2f} um  ts-wt={p_tswt:.2f} um")

    results = {}

    # ── GRU-D ──
    for rdt in [False, True]:
        rmse, mae, eyewt = train_recurrent(
            GRUD, "grud", rdt,
            train_seqs, val_seqs, test_seqs, cst_mean, cst_std, device)
        results[f"grud_{'realdelta' if rdt else 'ordinal'}"] = (rmse, mae, eyewt)

    # ── T-LSTM ──
    for rdt in [False, True]:
        rmse, mae, eyewt = train_recurrent(
            TLSTM, "tlstm", rdt,
            train_seqs, val_seqs, test_seqs, cst_mean, cst_std, device)
        results[f"tlstm_{'realdelta' if rdt else 'ordinal'}"] = (rmse, mae, eyewt)

    # ── ODE-RNN ──
    for rdt in [False, True]:
        rmse, mae, eyewt = train_ode(
            rdt, train_seqs, val_seqs, test_seqs, cst_mean, cst_std, device)
        results[f"ode_{'realdelta' if rdt else 'ordinal'}"] = (rmse, mae, eyewt)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VALIDATION-SPLIT RESULTS — FULL METRIC TABLE")
    print("=" * 80)
    print(f"  {'Model':<20} {'Ord ts-wt':>10} {'Ord eye-wt':>11} {'Rdt ts-wt':>10} {'D ts-wt':>8} {'Rdt MAE':>9}")
    print(f"  {'-'*20} {'-'*10} {'-'*11} {'-'*10} {'-'*8} {'-'*9}")
    print(f"  {'Persistence (eye-wt)':<20} {'91.74':>10} {'91.74':>11}")
    print(f"  {'Persistence (ts-wt)':<20} {'55.04':>10}")
    for model_name in ["grud", "tlstm", "ode"]:
        ord_rmse, ord_mae, ord_eyewt = results[f"{model_name}_ordinal"]
        rdt_rmse, rdt_mae, rdt_eyewt = results[f"{model_name}_realdelta"]
        delta = rdt_rmse - ord_rmse
        label = {"grud": "GRU-D", "tlstm": "T-LSTM", "ode": "ODE-RNN"}[model_name]
        print(f"  {label:<20} {ord_rmse:>10.2f} {ord_eyewt:>11.2f} {rdt_rmse:>10.2f} {delta:>+8.2f} {rdt_mae:>9.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
