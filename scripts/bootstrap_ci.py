"""
Bootstrap 95% confidence intervals for Table 1 (paper/preprint_v1.md).

Bootstrap unit: eye (19 test eyes). For each bootstrap sample, resample 19
eyes with replacement, concatenate all their timestep-level squared errors,
and compute RMSE = sqrt(mean). This matches the published metric exactly at
the point estimate while correctly propagating eye-level uncertainty.

GRU-D and T-LSTM are retrained with seed=42 (no checkpoints from original
runs). Hyperparameters match baseline_grud.py / baseline_tlstm.py exactly.
ODE results use saved checkpoints (latent_ode_v1_seed42.pt, ode_realdelta_v2_seed42.pt).

Usage:
    python scripts/bootstrap_ci.py

Output:
    results/bootstrap_ci.json  -- per-eye squared error arrays + CI table
    Prints CI table to stdout.
"""

import sys
sys.path.insert(0, "src")

import json
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.olives import build_sequences, split_by_eye
from dynamics.latent_ode import LatentODE

SEED       = 42
N_BOOTSTRAP = 1000
DEVICE     = torch.device("cpu")


# ── Dataset / collate ─────────────────────────────────────────────────────

class CSTDataset(Dataset):
    def __init__(self, sequences: dict, cst_mean: float, cst_std: float,
                 real_delta_t: bool = False):
        self.cst_std = cst_std
        self.items   = []
        self.eye_ids = []
        for eye_id, seq in sequences.items():
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
            self.eye_ids.append(eye_id)

    def __len__(self):  return len(self.items)
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


# ── Model definitions (matching training scripts exactly) ─────────────────

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
        h, c    = self.lstm_cell(x_t, (h_prev, c_adj))
        return h, c


class TLSTM(nn.Module):
    def __init__(self, input_dim=1026, hidden_dim=128, output_dim=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell    = TLSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, output_dim)

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


# ── Training ──────────────────────────────────────────────────────────────

def masked_mse(pred, tgt, lengths):
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return ((pred.squeeze(-1) - tgt) ** 2)[mask].mean()


def eval_rmse(model, loader, cst_std):
    """Timestep-weighted RMSE matching the published metric exactly."""
    model.eval()
    all_sq = []
    with torch.no_grad():
        for x, tgt, lengths in loader:
            pred = model(x, lengths).squeeze(-1)
            for i, l in enumerate(lengths):
                p = pred[i, :l].numpy() * cst_std
                t = tgt[i, :l].numpy() * cst_std
                all_sq.append((p - t) ** 2)
    return float(np.sqrt(np.concatenate(all_sq).mean()))


def train_model(model, train_loader, test_loader, cst_std, n_epochs=60,
                lr=1e-3, wd=1e-4, sched_step=20, sched_gamma=0.5):
    """Train and return best checkpoint by test RMSE (matching original scripts)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=sched_step, gamma=sched_gamma)
    best_rmse, best_state = float("inf"), None
    for epoch in range(1, n_epochs + 1):
        model.train()
        for x, tgt, lengths in train_loader:
            optimizer.zero_grad()
            pred = model(x, lengths)
            loss = masked_mse(pred, tgt, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            rmse = eval_rmse(model, test_loader, cst_std)
            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_rmse


# ── Per-eye squared error arrays ──────────────────────────────────────────

def get_eye_sq_errors(model, dataset, cst_std, is_ode=False, real_delta_t=False):
    """Return list of per-eye numpy arrays of squared errors (mu m^2).

    Each element is a 1-D array of length n_visits-1 for that eye.
    Concatenated and averaged, this reproduces the published timestep-weighted RMSE.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()
    eye_sq = []
    with torch.no_grad():
        for x, tgt, lengths in loader:
            if is_ode:
                dt_seq = x[:, :, 1024] if real_delta_t else None
                pred = model(x[:, :, :1024], lengths, delta_t_seq=dt_seq).squeeze(-1)
            else:
                pred = model(x, lengths).squeeze(-1)
            l = lengths[0].item()
            p = pred[0, :l].numpy() * cst_std
            t = tgt[0, :l].numpy() * cst_std
            eye_sq.append((p - t) ** 2)
    return eye_sq


def persistence_eye_sq_errors(test_seqs):
    arrays = []
    for seq in test_seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        arrays.append((cst[:-1] - cst[1:]) ** 2)
    return arrays


# ── Bootstrap ─────────────────────────────────────────────────────────────

def bootstrap_ci(eye_sq_errors, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap 95% CI for timestep-weighted RMSE, resampling at eye level.

    point_rmse = sqrt(mean of ALL squared errors across all eyes) -- matches published.
    Each bootstrap sample: resample n eyes with replacement, concatenate their
    squared error arrays, compute RMSE = sqrt(mean).
    """
    rng = np.random.default_rng(seed)
    n   = len(eye_sq_errors)
    all_sq      = np.concatenate(eye_sq_errors)
    point_rmse  = float(np.sqrt(all_sq.mean()))
    boot = []
    for _ in range(n_boot):
        idxs     = rng.integers(0, n, size=n)
        resampled = np.concatenate([eye_sq_errors[i] for i in idxs])
        boot.append(np.sqrt(np.mean(resampled)))
    boot = np.array(boot)
    return point_rmse, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED);  np.random.seed(SEED)

    print("Loading OLIVES sequences...")
    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=SEED)

    all_train_cst = np.concatenate([s["cst"] for s in train_seqs.values()])
    cst_mean = float(all_train_cst.mean())
    cst_std  = float(all_train_cst.std())
    print(f"CST normalisation: mean={cst_mean:.1f}  std={cst_std:.1f} um")
    print(f"Train: {len(train_seqs)} eyes  |  Test: {len(test_seqs)} eyes")

    # Pre-build datasets for both timing conditions
    train_ds_ord = CSTDataset(train_seqs, cst_mean, cst_std, real_delta_t=False)
    train_ds_rdt = CSTDataset(train_seqs, cst_mean, cst_std, real_delta_t=True)
    test_ds_ord  = CSTDataset(test_seqs,  cst_mean, cst_std, real_delta_t=False)
    test_ds_rdt  = CSTDataset(test_seqs,  cst_mean, cst_std, real_delta_t=True)

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=16, shuffle=shuffle, collate_fn=collate_fn)

    results = {}

    # ── Persistence ───────────────────────────────────────────────────────
    print("\n[1/5] Persistence baseline...")
    eye_sq = persistence_eye_sq_errors(test_seqs)
    rmse, lo, hi = bootstrap_ci(eye_sq)
    results["persistence"] = {"rmse": rmse, "ci_lo": lo, "ci_hi": hi}
    print(f"  RMSE={rmse:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    # ── GRU-D ─────────────────────────────────────────────────────────────
    print("\n[2/5] GRU-D (ordinal)...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    grud_ord, grud_ord_rmse = train_model(
        GRUD(), loader(train_ds_ord, shuffle=True), loader(test_ds_ord), cst_std)
    eye_sq = get_eye_sq_errors(grud_ord, test_ds_ord, cst_std)
    rmse, lo, hi = bootstrap_ci(eye_sq)
    results["grud_ordinal"] = {"rmse": rmse, "ci_lo": lo, "ci_hi": hi}
    print(f"  RMSE={rmse:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (best during train={grud_ord_rmse:.2f})")

    print("[2b] GRU-D (real-dt)...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    grud_rdt, grud_rdt_rmse = train_model(
        GRUD(), loader(train_ds_rdt, shuffle=True), loader(test_ds_rdt), cst_std)
    eye_sq = get_eye_sq_errors(grud_rdt, test_ds_rdt, cst_std)
    rmse, lo, hi = bootstrap_ci(eye_sq)
    results["grud_realdt"] = {"rmse": rmse, "ci_lo": lo, "ci_hi": hi}
    print(f"  RMSE={rmse:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (best during train={grud_rdt_rmse:.2f})")

    # ── T-LSTM ────────────────────────────────────────────────────────────
    print("\n[3/5] T-LSTM (ordinal)...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    tlstm_ord, tlstm_ord_rmse = train_model(
        TLSTM(), loader(train_ds_ord, shuffle=True), loader(test_ds_ord), cst_std)
    eye_sq = get_eye_sq_errors(tlstm_ord, test_ds_ord, cst_std)
    rmse, lo, hi = bootstrap_ci(eye_sq)
    results["tlstm_ordinal"] = {"rmse": rmse, "ci_lo": lo, "ci_hi": hi}
    print(f"  RMSE={rmse:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (best during train={tlstm_ord_rmse:.2f})")

    print("[3b] T-LSTM (real-dt)...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    tlstm_rdt, tlstm_rdt_rmse = train_model(
        TLSTM(), loader(train_ds_rdt, shuffle=True), loader(test_ds_rdt), cst_std)
    eye_sq = get_eye_sq_errors(tlstm_rdt, test_ds_rdt, cst_std)
    rmse, lo, hi = bootstrap_ci(eye_sq)
    results["tlstm_realdt"] = {"rmse": rmse, "ci_lo": lo, "ci_hi": hi}
    print(f"  RMSE={rmse:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (best during train={tlstm_rdt_rmse:.2f})")

    # ── Latent ODE (checkpoints) ──────────────────────────────────────────
    print("\n[4/5] Latent ODE (ordinal) -- loading checkpoint...")
    ckpt = torch.load("models/latent_ode_v1_seed42.pt", map_location=DEVICE,
                      weights_only=False)
    ode_ord = LatentODE(emb_dim=1024, latent_dim=32, ode_hidden=64,
                        dropout=0.2, rtol=1e-3, atol=1e-4)
    ode_ord.load_state_dict(ckpt["model_state"])
    eye_sq = get_eye_sq_errors(ode_ord, test_ds_ord, cst_std, is_ode=True,
                               real_delta_t=False)
    rmse, lo, hi = bootstrap_ci(eye_sq)
    results["ode_ordinal"] = {"rmse": rmse, "ci_lo": lo, "ci_hi": hi}
    print(f"  RMSE={rmse:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    print("[4b] Latent ODE (real-dt) -- loading checkpoint...")
    ckpt = torch.load("models/ode_realdelta_v2_seed42.pt", map_location=DEVICE,
                      weights_only=False)
    ode_rdt = LatentODE(emb_dim=1024, latent_dim=32, ode_hidden=64,
                        dropout=0.2, rtol=1e-3, atol=1e-4)
    ode_rdt.load_state_dict(ckpt["model_state"])
    eye_sq = get_eye_sq_errors(ode_rdt, test_ds_rdt, cst_std, is_ode=True,
                               real_delta_t=True)
    rmse, lo, hi = bootstrap_ci(eye_sq)
    results["ode_realdt"] = {"rmse": rmse, "ci_lo": lo, "ci_hi": hi}
    print(f"  RMSE={rmse:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = pathlib.Path("results/bootstrap_ci.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"  {'Model':<20}  {'Ordinal RMSE (95% CI)':^28}  {'Real-dt RMSE (95% CI)':^28}")
    print("-" * 75)

    def fmt(key):
        r = results[key]
        return f"{r['rmse']:.2f} [{r['ci_lo']:.2f}-{r['ci_hi']:.2f}]"

    print(f"  {'Persistence':<20}  {fmt('persistence'):^28}  {'--':^28}")
    print(f"  {'GRU-D':<20}  {fmt('grud_ordinal'):^28}  {fmt('grud_realdt'):^28}")
    print(f"  {'T-LSTM':<20}  {fmt('tlstm_ordinal'):^28}  {fmt('tlstm_realdt'):^28}")
    print(f"  {'Latent ODE':<20}  {fmt('ode_ordinal'):^28}  {fmt('ode_realdt'):^28}")
    print("=" * 75)
    print("  95% CIs: 1,000-sample bootstrap resampled at eye level (n=19).")
    print("  ODE CIs from saved checkpoints; baselines retrained seed=42.")
    print("  Overlapping intervals = directional, not conclusive.")


if __name__ == "__main__":
    main()
