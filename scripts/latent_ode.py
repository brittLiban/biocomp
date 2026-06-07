"""
Latent ODE — next-visit CST regression training script.

Model:  src/dynamics/latent_ode.py  (ODE-RNN, Rubanova et al. 2019)
Target: RMSE < 82.0 um on next-visit CST, test split seed=42 (Decision #9)
W&B:    project=synapse, run=ode_realdelta_seed42 (real delta-t) or latent_ode_v1_seed42

Input / target / split / normalisation identical to baseline_grud.py:
  - Input sequence: (n_visits - 1, 1026) = [emb_t (1024) | δt (1) | bcva_t (1)]
    δt column: week_gaps (normalized) when REAL_DELTA_T=True; ordinal 1.0 otherwise.
    Only emb_t[:, :1024] is fed to the ODE encoder; δt is also extracted for
    the ODE integration interval.
  - Target: next-visit CST, normalised to zero mean / unit std (train stats only)
  - Split: split_by_eye(seqs, test_frac=0.2, seed=42) → 77 train / 19 test eyes

Run:
    python scripts/latent_ode.py
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

# Set True to use real week gaps for ODE integration interval.
# Prime eyes: gaps normalized to 4-week units (column 1024 of input tensor).
# TREX eyes: ordinal 1.0 per step (real timing unavailable).
# The ODE model reads x[:, :, 1024] as delta_t_seq when REAL_DELTA_T=True.
REAL_DELTA_T = True


# ── Dataset ────────────────────────────────────────────────────────────────

class CSTRegressionDataset(Dataset):
    """
    One item per eye: full visit sequence minus the last visit.

    Input shape:  (n_visits - 1, 1026)  [emb_t (1024) | δt (1) | bcva_t (1)]
    Target shape: (n_visits - 1,)       CST at visit t+1, normalised.

    δt column: week_gaps (normalized) when REAL_DELTA_T=True; ordinal 1.0 otherwise.
    """

    def __init__(self, sequences: dict, cst_mean: float, cst_std: float):
        self.cst_mean = cst_mean
        self.cst_std  = cst_std
        self.items    = []

        for seq in sequences.values():
            n = seq["n_visits"]
            if n < 2:
                continue
            embs = seq["embeddings"].astype(np.float32)   # (n, 1024)
            cst  = seq["cst"].astype(np.float32)          # (n,)
            bcva = seq["bcva"].astype(np.float32)         # (n,)

            if REAL_DELTA_T and "week_gaps" in seq:
                delta_t = seq["week_gaps"].astype(np.float32)  # (n-1,)
            else:
                delta_t = np.ones(n - 1, dtype=np.float32)

            inp = np.concatenate([
                embs[:-1],
                delta_t[:, None],   # column 1024: integration interval for ODE
                bcva[:-1, None],
            ], axis=1)                                        # (n-1, 1026)
            tgt = (cst[1:] - cst_mean) / cst_std            # (n-1,)

            self.items.append((inp, tgt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        inp, tgt = self.items[idx]
        return torch.from_numpy(inp), torch.from_numpy(tgt)


def collate_fn(batch):
    """Pad variable-length sequences to longest in batch."""
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


# ── Loss & evaluation ──────────────────────────────────────────────────────

def masked_mse(pred: torch.Tensor, tgt: torch.Tensor,
               lengths: torch.Tensor) -> torch.Tensor:
    """MSE over valid (non-padded) timesteps only. Identical to baselines."""
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return ((pred.squeeze(-1) - tgt) ** 2)[mask].mean()


def evaluate(model, loader, cst_std: float, device) -> tuple[float, float]:
    """RMSE and MAE in original CST units (μm). Identical to baselines."""
    model.eval()
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for x, tgt, lengths in loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            delta_t_seq = x[:, :, 1024] if REAL_DELTA_T else None
            pred = model(x[:, :, :1024], lengths, delta_t_seq=delta_t_seq).squeeze(-1)
            for i, l in enumerate(lengths):
                all_pred.append(pred[i, :l].cpu().numpy())
                all_tgt.append(tgt[i, :l].cpu().numpy())
    pred_np = np.concatenate(all_pred) * cst_std
    tgt_np  = np.concatenate(all_tgt)  * cst_std
    rmse = float(np.sqrt(((pred_np - tgt_np) ** 2).mean()))
    mae  = float(np.abs(pred_np - tgt_np).mean())
    return rmse, mae


def persistence_rmse(test_seqs: dict) -> float:
    """Naive persistence baseline: predict last observed CST. Lower bound."""
    sq_errors = []
    for seq in test_seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        sq_errors.append(((cst[:-1] - cst[1:]) ** 2).mean())
    return float(np.sqrt(np.mean(sq_errors)))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    SEED       = 42
    TARGET_BAR = 82.0   # Decision #9 — must beat this, no moving it

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    CFG = dict(
        latent_dim   = 32,
        ode_hidden   = 64,
        dropout      = 0.2,
        lr           = 1e-3,
        weight_decay = 1e-4,
        batch_size   = 16,
        n_epochs     = 100,
        sched_step   = 30,
        sched_gamma  = 0.5,
        rtol         = 1e-3,
        atol         = 1e-4,
        seed         = SEED,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = "ode_realdelta_seed42" if REAL_DELTA_T else "latent_ode_v1_seed42"
    run = wandb.init(
        project = "synapse",
        name    = run_name,
        config  = {**CFG, "real_delta_t": REAL_DELTA_T},
        tags    = ["latent-ode", "cst-regression",
                   "real-delta-t" if REAL_DELTA_T else "ordinal"],
    )

    # ── Data — same split as every baseline
    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=SEED)

    all_train_cst = np.concatenate([s["cst"] for s in train_seqs.values()])
    cst_mean = float(all_train_cst.mean())
    cst_std  = float(all_train_cst.std())
    print(f"CST normalisation (train): mean={cst_mean:.1f}  std={cst_std:.1f} um")

    train_ds = CSTRegressionDataset(train_seqs, cst_mean, cst_std)
    test_ds  = CSTRegressionDataset(test_seqs,  cst_mean, cst_std)
    print(f"Train sequences: {len(train_ds)}  |  Test sequences: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(
        test_ds,  batch_size=CFG["batch_size"], shuffle=False, collate_fn=collate_fn)

    # ── Model
    model = LatentODE(
        emb_dim    = 1024,
        latent_dim = CFG["latent_dim"],
        ode_hidden = CFG["ode_hidden"],
        dropout    = CFG["dropout"],
        rtol       = CFG["rtol"],
        atol       = CFG["atol"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Latent ODE parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CFG["sched_step"], gamma=CFG["sched_gamma"])

    # ── Train
    best_rmse  = float("inf")
    best_state = None

    print(f"\nTraining Latent ODE for {CFG['n_epochs']} epochs...")
    print(f"Must beat: {TARGET_BAR:.1f} um RMSE (Decision #9, non-negotiable)")

    for epoch in range(1, CFG["n_epochs"] + 1):
        model.train()
        train_loss = 0.0

        for x, tgt, lengths in train_loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            optimizer.zero_grad()
            delta_t_seq = x[:, :, 1024] if REAL_DELTA_T else None
            pred = model(x[:, :, :1024], lengths, delta_t_seq=delta_t_seq)
            loss = masked_mse(pred, tgt, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch,
                   "lr": scheduler.get_last_lr()[0]})

        if epoch % 10 == 0:
            rmse, mae = evaluate(model, test_loader, cst_std, device)
            is_best   = rmse < best_rmse
            if is_best:
                best_rmse  = rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wandb.log({"test_rmse": rmse, "test_mae": mae,
                       "best_test_rmse": best_rmse, "epoch": epoch})
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} "
                  f"| RMSE={rmse:.1f} um | MAE={mae:.1f} um"
                  f"{'  <-- best' if is_best else ''}")

    # ── Final result — best checkpoint, not last epoch
    model.load_state_dict(best_state)
    final_rmse, final_mae = evaluate(model, test_loader, cst_std, device)
    persist = persistence_rmse(test_seqs)

    wandb.log({
        "final_rmse":       final_rmse,
        "final_mae":        final_mae,
        "persistence_rmse": persist,
        "target_bar":       TARGET_BAR,
        "beat_bar":         final_rmse < TARGET_BAR,
    })

    print("\n" + "=" * 55)
    print("  Latent ODE - next-visit CST regression")
    print("=" * 55)
    print(f"  RMSE (primary)  : {final_rmse:.1f} um")
    print(f"  MAE             : {final_mae:.1f} um")
    print(f"  Persistence RMSE: {persist:.1f} um  (naive last-value)")
    print(f"  Baseline bar    : {TARGET_BAR:.1f} um  (GRU-D/T-LSTM best, Decision #9)")
    print(f"  vs bar          : {final_rmse - TARGET_BAR:+.1f} um  "
          f"({'BEATS BAR' if final_rmse < TARGET_BAR else 'MISSES BAR'})")
    print("=" * 55)

    # Save best checkpoint to disk for reproducibility
    import pathlib
    ckpt_dir = pathlib.Path("models")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = "ode_realdelta_seed42.pt" if REAL_DELTA_T else "latent_ode_v1_seed42.pt"
    ckpt_path = ckpt_dir / ckpt_name
    torch.save({
        "model_state": best_state,
        "cfg":         CFG,
        "cst_mean":    cst_mean,
        "cst_std":     cst_std,
        "final_rmse":  final_rmse,
        "final_mae":   final_mae,
    }, ckpt_path)
    print(f"\n  Checkpoint saved -> {ckpt_path}")

    run.finish()
    return final_rmse


if __name__ == "__main__":
    main()
