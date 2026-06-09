"""
Metric inconsistency diagnostic.

Computes both eye-weighted and timestep-weighted RMSE for all models,
per-eye RMSE distributions, and the timing experiment under both metrics.

No paper edits. Report only.

Usage:
    python scripts/diagnose_metrics.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.olives import build_sequences, split_by_eye
from dynamics.latent_ode import LatentODE

SEED   = 42
DEVICE = torch.device("cpu")


# ── Dataset / collate (identical to bootstrap_ci.py) ─────────────────────

class CSTDataset(Dataset):
    def __init__(self, sequences, cst_mean, cst_std, real_delta_t=False):
        self.cst_std = cst_std
        self.items   = []
        self.eye_ids = []
        for eye_id, seq in sequences.items():
            n = seq["n_visits"]
            if n < 2:
                continue
            embs    = seq["embeddings"].astype(np.float32)
            cst     = seq["cst"].astype(np.float32)
            bcva    = seq["bcva"].astype(np.float32)
            delta_t = (seq["week_gaps"].astype(np.float32)
                       if real_delta_t and "week_gaps" in seq
                       else np.ones(n - 1, dtype=np.float32))
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


# ── Model definitions ─────────────────────────────────────────────────────

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
        h, c    = self.lstm_cell(x_t, (h_prev, c_short + c_long))
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


def train_model(model, train_loader, test_loader, cst_std,
                n_epochs=60, lr=1e-3, wd=1e-4, sched_step=20, sched_gamma=0.5):
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
            # timestep-weighted RMSE for checkpoint selection (matches original scripts)
            rmse = timestep_rmse_from_loader(model, test_loader, cst_std,
                                             is_ode=False)
            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


# ── Per-eye prediction extraction ─────────────────────────────────────────

def get_per_eye_predictions(model, dataset, cst_std, is_ode=False, real_delta_t=False):
    """
    Returns list of (n_visits-1,) arrays: per-eye squared errors in um^2.
    Also returns per-eye visit counts.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()
    per_eye_sq  = []   # list of arrays, one per eye
    per_eye_n   = []   # number of predictions per eye
    with torch.no_grad():
        for x, tgt, lengths in loader:
            if is_ode:
                dt_seq = x[:, :, 1024] if real_delta_t else None
                pred = model(x[:, :, :1024], lengths, delta_t_seq=dt_seq).squeeze(-1)
            else:
                pred = model(x, lengths).squeeze(-1)
            l = lengths[0].item()
            p = pred[0, :l].numpy() * cst_std
            t = tgt[0,  :l].numpy() * cst_std
            sq = (p - t) ** 2
            per_eye_sq.append(sq)
            per_eye_n.append(l)
    return per_eye_sq, per_eye_n


def persistence_per_eye_sq(test_seqs):
    arrays = []
    counts = []
    for seq in test_seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        sq = (cst[:-1] - cst[1:]) ** 2
        arrays.append(sq)
        counts.append(len(sq))
    return arrays, counts


# ── RMSE metrics ──────────────────────────────────────────────────────────

def timestep_rmse(per_eye_sq):
    """sqrt(mean over all timesteps) — matches published model evaluation."""
    return float(np.sqrt(np.concatenate(per_eye_sq).mean()))


def eye_rmse(per_eye_sq):
    """sqrt(mean of per-eye MSEs) — matches published persistence evaluation."""
    per_eye_mse = [sq.mean() for sq in per_eye_sq]
    return float(np.sqrt(np.mean(per_eye_mse)))


def per_eye_rmse(per_eye_sq):
    """RMSE for each individual eye."""
    return [float(np.sqrt(sq.mean())) for sq in per_eye_sq]


def timestep_rmse_from_loader(model, loader, cst_std, is_ode=False, real_delta_t=False):
    model.eval()
    all_sq = []
    with torch.no_grad():
        for x, tgt, lengths in loader:
            if is_ode:
                dt_seq = x[:, :, 1024] if real_delta_t else None
                pred = model(x[:, :, :1024], lengths, delta_t_seq=dt_seq).squeeze(-1)
            else:
                pred = model(x, lengths).squeeze(-1)
            for i, l in enumerate(lengths):
                p = pred[i, :l].numpy() * cst_std
                t = tgt[i, :l].numpy() * cst_std
                all_sq.append((p - t) ** 2)
    return float(np.sqrt(np.concatenate(all_sq).mean()))


def loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=16, shuffle=shuffle, collate_fn=collate_fn)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED);  np.random.seed(SEED)

    print("Loading OLIVES sequences...")
    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=SEED)

    all_train_cst = np.concatenate([s["cst"] for s in train_seqs.values()])
    cst_mean = float(all_train_cst.mean())
    cst_std  = float(all_train_cst.std())
    print(f"CST: mean={cst_mean:.1f}  std={cst_std:.1f} um  |  "
          f"train={len(train_seqs)} eyes  test={len(test_seqs)} eyes\n")

    train_ds_ord = CSTDataset(train_seqs, cst_mean, cst_std, real_delta_t=False)
    train_ds_rdt = CSTDataset(train_seqs, cst_mean, cst_std, real_delta_t=True)
    test_ds_ord  = CSTDataset(test_seqs,  cst_mean, cst_std, real_delta_t=False)
    test_ds_rdt  = CSTDataset(test_seqs,  cst_mean, cst_std, real_delta_t=True)

    # ── Persistence ───────────────────────────────────────────────────────
    print("Computing persistence predictions...")
    p_sq, p_n = persistence_per_eye_sq(test_seqs)
    persist_per_eye = per_eye_rmse(p_sq)
    persist_ts  = timestep_rmse(p_sq)
    persist_eye = eye_rmse(p_sq)

    # ── GRU-D (ordinal) ───────────────────────────────────────────────────
    print("Training GRU-D ordinal...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    grud_ord = train_model(GRUD(),
                           loader(train_ds_ord, shuffle=True),
                           loader(test_ds_ord), cst_std)
    grud_ord_sq, _ = get_per_eye_predictions(grud_ord, test_ds_ord, cst_std)
    grud_ord_per_eye = per_eye_rmse(grud_ord_sq)
    grud_ord_ts  = timestep_rmse(grud_ord_sq)
    grud_ord_eye = eye_rmse(grud_ord_sq)

    # ── GRU-D (real-dt) ───────────────────────────────────────────────────
    print("Training GRU-D real-dt...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    grud_rdt = train_model(GRUD(),
                           loader(train_ds_rdt, shuffle=True),
                           loader(test_ds_rdt), cst_std)
    grud_rdt_sq, _ = get_per_eye_predictions(grud_rdt, test_ds_rdt, cst_std)
    grud_rdt_ts  = timestep_rmse(grud_rdt_sq)
    grud_rdt_eye = eye_rmse(grud_rdt_sq)

    # ── T-LSTM (ordinal) ──────────────────────────────────────────────────
    print("Training T-LSTM ordinal...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    tlstm_ord = train_model(TLSTM(),
                            loader(train_ds_ord, shuffle=True),
                            loader(test_ds_ord), cst_std)
    tlstm_ord_sq, _ = get_per_eye_predictions(tlstm_ord, test_ds_ord, cst_std)
    tlstm_ord_per_eye = per_eye_rmse(tlstm_ord_sq)
    tlstm_ord_ts  = timestep_rmse(tlstm_ord_sq)
    tlstm_ord_eye = eye_rmse(tlstm_ord_sq)

    # ── T-LSTM (real-dt) ──────────────────────────────────────────────────
    print("Training T-LSTM real-dt...")
    torch.manual_seed(SEED);  np.random.seed(SEED)
    tlstm_rdt = train_model(TLSTM(),
                            loader(train_ds_rdt, shuffle=True),
                            loader(test_ds_rdt), cst_std)
    tlstm_rdt_sq, _ = get_per_eye_predictions(tlstm_rdt, test_ds_rdt, cst_std)
    tlstm_rdt_ts  = timestep_rmse(tlstm_rdt_sq)
    tlstm_rdt_eye = eye_rmse(tlstm_rdt_sq)

    # ── ODE (ordinal, checkpoint) ─────────────────────────────────────────
    print("Loading ODE ordinal checkpoint...")
    ckpt = torch.load("models/latent_ode_v1_seed42.pt", map_location=DEVICE,
                      weights_only=False)
    ode_ord = LatentODE(emb_dim=1024, latent_dim=32, ode_hidden=64,
                        dropout=0.2, rtol=1e-3, atol=1e-4)
    ode_ord.load_state_dict(ckpt["model_state"])
    ode_ord_sq, _ = get_per_eye_predictions(ode_ord, test_ds_ord, cst_std,
                                            is_ode=True, real_delta_t=False)
    ode_ord_per_eye = per_eye_rmse(ode_ord_sq)
    ode_ord_ts  = timestep_rmse(ode_ord_sq)
    ode_ord_eye = eye_rmse(ode_ord_sq)

    # ── ODE (real-dt, checkpoint) ─────────────────────────────────────────
    print("Loading ODE real-dt checkpoint...")
    ckpt = torch.load("models/ode_realdelta_v2_seed42.pt", map_location=DEVICE,
                      weights_only=False)
    ode_rdt = LatentODE(emb_dim=1024, latent_dim=32, ode_hidden=64,
                        dropout=0.2, rtol=1e-3, atol=1e-4)
    ode_rdt.load_state_dict(ckpt["model_state"])
    ode_rdt_sq, _ = get_per_eye_predictions(ode_rdt, test_ds_rdt, cst_std,
                                            is_ode=True, real_delta_t=True)
    ode_rdt_ts  = timestep_rmse(ode_rdt_sq)
    ode_rdt_eye = eye_rmse(ode_rdt_sq)

    # ─────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC 1: Per-eye RMSE for persistence, GRU-D ordinal,
    #               T-LSTM ordinal, ODE ordinal
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("DIAGNOSTIC 1 — Per-eye RMSE (um) for 19 test eyes (ordinal time)")
    print("=" * 75)
    n_visits_test = [len(sq) for sq in p_sq]
    header = f"  {'Eye':>3}  {'Visits':>6}  {'Persist':>8}  {'GRU-D':>8}  {'T-LSTM':>8}  {'ODE':>8}  {'Best model':>10}"
    print(header)
    print("-" * 75)
    n_model_beats_persist = 0
    for i in range(len(persist_per_eye)):
        nv  = n_visits_test[i]
        prs = persist_per_eye[i]
        gru = grud_ord_per_eye[i]
        tls = tlstm_ord_per_eye[i]
        ode = ode_ord_per_eye[i]
        best_model = min(gru, tls, ode)
        beats = "*" if best_model < prs else " "
        if best_model < prs:
            n_model_beats_persist += 1
        print(f"  {i+1:>3}  {nv:>6}  {prs:>8.1f}  {gru:>8.1f}  {tls:>8.1f}  {ode:>8.1f}  {beats}")
    print(f"\n  Best model beats persistence on {n_model_beats_persist}/19 eyes ({100*n_model_beats_persist/19:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC 2 & 3: Eye-weighted vs timestep-weighted RMSE
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("DIAGNOSTIC 2 & 3 — Eye-weighted vs Timestep-weighted RMSE (um)")
    print("=" * 75)
    print(f"  {'Model':<20}  {'Eye-weighted':>14}  {'Timestep-weighted':>18}  {'Difference':>12}")
    print("-" * 75)

    rows = [
        ("Persistence",        persist_eye,   persist_ts),
        ("GRU-D (ordinal)",    grud_ord_eye,  grud_ord_ts),
        ("T-LSTM (ordinal)",   tlstm_ord_eye, tlstm_ord_ts),
        ("ODE (ordinal)",      ode_ord_eye,   ode_ord_ts),
    ]
    for name, ew, tw in rows:
        diff = ew - tw
        print(f"  {name:<20}  {ew:>14.2f}  {tw:>18.2f}  {diff:>+12.2f}")

    print(f"\n  Published W&B numbers (timestep-weighted from original scripts):")
    print(f"    Persistence: 91.7 (eye-weighted in original) vs {persist_ts:.2f} (timestep-weighted)")
    print(f"    GRU-D:  {grud_ord_ts:.2f}  |  T-LSTM: {tlstm_ord_ts:.2f}  |  ODE: {ode_ord_ts:.2f}")

    # ─────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC 4: Timing experiment under both metrics
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("DIAGNOSTIC 4 — Timing experiment: Ordinal vs Real-dt")
    print("  Does ODE improve / recurrents degrade under both metrics?")
    print("=" * 75)

    print(f"\n  {'Model':<20}  {'Ord (ts-wt)':>12}  {'Rdt (ts-wt)':>12}  {'Delta(ts)':>10}  "
          f"{'Ord (eye)':>10}  {'Rdt (eye)':>10}  {'Delta(eye)':>11}")
    print("-" * 90)

    timing_rows = [
        ("GRU-D",      grud_ord_ts,  grud_rdt_ts,  grud_ord_eye,  grud_rdt_eye),
        ("T-LSTM",     tlstm_ord_ts, tlstm_rdt_ts, tlstm_ord_eye, tlstm_rdt_eye),
        ("Latent ODE", ode_ord_ts,   ode_rdt_ts,   ode_ord_eye,   ode_rdt_eye),
    ]
    for name, ots, rts, oew, rew in timing_rows:
        dts = rts - ots
        dew = rew - oew
        print(f"  {name:<20}  {ots:>12.2f}  {rts:>12.2f}  {dts:>+10.2f}  "
              f"{oew:>10.2f}  {rew:>10.2f}  {dew:>+11.2f}")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  Persistence (ts-weighted): {persist_ts:.2f} um")
    print(f"  Persistence (eye-weighted): {persist_eye:.2f} um  (published: 91.7)")
    print(f"  GRU-D ordinal  ts={grud_ord_ts:.2f}  eye={grud_ord_eye:.2f}")
    print(f"  T-LSTM ordinal ts={tlstm_ord_ts:.2f}  eye={tlstm_ord_eye:.2f}")
    print(f"  ODE ordinal    ts={ode_ord_ts:.2f}  eye={ode_ord_eye:.2f}")
    print()
    print(f"  Models beat persistence (ts-weighted)?  "
          f"GRU-D: {'YES' if grud_ord_ts < persist_ts else 'NO'}  "
          f"T-LSTM: {'YES' if tlstm_ord_ts < persist_ts else 'NO'}  "
          f"ODE: {'YES' if ode_ord_ts < persist_ts else 'NO'}")
    print(f"  Models beat persistence (eye-weighted)? "
          f"GRU-D: {'YES' if grud_ord_eye < persist_eye else 'NO'}  "
          f"T-LSTM: {'YES' if tlstm_ord_eye < persist_eye else 'NO'}  "
          f"ODE: {'YES' if ode_ord_eye < persist_eye else 'NO'}")
    print()
    print(f"  Timing experiment direction (ts-weighted):")
    print(f"    GRU-D delta: {grud_rdt_ts - grud_ord_ts:+.2f}  "
          f"T-LSTM delta: {tlstm_rdt_ts - tlstm_ord_ts:+.2f}  "
          f"ODE delta: {ode_rdt_ts - ode_ord_ts:+.2f}")
    print(f"  Timing experiment direction (eye-weighted):")
    print(f"    GRU-D delta: {grud_rdt_eye - grud_ord_eye:+.2f}  "
          f"T-LSTM delta: {tlstm_rdt_eye - tlstm_ord_eye:+.2f}  "
          f"ODE delta: {ode_rdt_eye - ode_ord_eye:+.2f}")


if __name__ == "__main__":
    main()
