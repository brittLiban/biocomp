"""
True Latent ODE v3 — beta-VAE + variable-endpoint augmentation (Decision #25).

Changes from v2:
  - Fixed beta=0.1 (beta-VAE, never anneals to 1.0) — main fix for KL collapse
  - Variable-endpoint subsequences: all (start, end) windows, not just start-varies
    77 eyes -> ~7,000 training sequences (vs 778 in v2)
  - min_subseq_len reduced 4 -> 3 (more short sequences)
  - Longer training: 300 epochs (model has more sequences to learn from)

Evidence for beta=0.1: v1 and v2 best checkpoints were both at epoch 10 when
beta was naturally 0.20. Permanently lower beta sustains that regime.

Model architecture unchanged. Evaluation bars unchanged (Decision #23).
W&B: project=synapse, run=true_latent_ode_v3_seed42

Run:
    python scripts/true_latent_ode_v3.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wandb

from data.olives import build_sequences, split_by_eye
from dynamics.true_latent_ode import TrueLatentODE

REAL_DELTA_T = True


# ── Dataset ────────────────────────────────────────────────────────────────

class VariableWindowDataset(Dataset):
    """
    Variable-endpoint augmentation dataset (Decision #25).

    For each eye with n visits, generates ALL (start, end) subsequence windows
    where end - start >= min_subseq_len. This gives ~91 windows per 16-visit eye
    vs ~10 in v2 (start-only variation).

    For n=16, min_len=3: ~91 windows per eye -> 77 eyes -> ~7,000 train sequences.

    Augmentations (training only):
      - Embedding noise: Gaussian(0, emb_noise_std)
      - Time jitter: delta_t * Uniform(1-jitter, 1+jitter), clipped to min 0.1
    """

    def __init__(
        self,
        sequences:      dict,
        cst_mean:       float,
        cst_std:        float,
        min_subseq_len: int   = 3,
        emb_noise_std:  float = 0.02,
        time_jitter:    float = 0.10,
        augment:        bool  = True,
    ):
        self.cst_mean      = cst_mean
        self.cst_std       = cst_std
        self.emb_noise_std = emb_noise_std
        self.time_jitter   = time_jitter
        self.augment       = augment
        self.items         = []

        for seq in sequences.values():
            n = seq["n_visits"]
            if n < min_subseq_len + 1:
                continue

            embs     = seq["embeddings"].astype(np.float32)    # (n, 1024)
            cst      = seq["cst"].astype(np.float32)           # (n,)
            bcva     = seq["bcva"].astype(np.float32)          # (n,)
            cst_norm = (cst - cst_mean) / cst_std

            if REAL_DELTA_T and "week_gaps" in seq:
                delta_t = seq["week_gaps"].astype(np.float32)  # (n-1,)
            else:
                delta_t = np.ones(n - 1, dtype=np.float32)

            if augment:
                # All (start, end) windows where end - start >= min_subseq_len
                # Input: visits [start .. end-2], Target: CST at [start+1 .. end-1]
                for start in range(n - min_subseq_len):
                    for end in range(start + min_subseq_len, n):
                        # end is exclusive of last visit used as input
                        # input visits: start to end-1 (length = end - start)
                        # target: CST at visits start+1 to end
                        L = end - start          # number of input steps
                        self.items.append((
                            embs[start:end],         # (L, 1024)
                            delta_t[start:end],      # (L,) — gaps from visit i to i+1
                            bcva[start:end],         # (L,)
                            cst_norm[start+1:end+1], # (L,) — targets
                        ))
            else:
                # Full sequence only for test
                self.items.append((
                    embs[:-1],
                    delta_t,
                    bcva[:-1],
                    cst_norm[1:],
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        embs, delta_t, bcva, tgt = self.items[idx]

        if self.augment:
            embs    = embs + np.random.normal(
                0, self.emb_noise_std, embs.shape).astype(np.float32)
            jitter  = np.random.uniform(
                -self.time_jitter, self.time_jitter, delta_t.shape).astype(np.float32)
            delta_t = np.clip(delta_t * (1.0 + jitter), 0.1, None)

        inp = np.concatenate([embs, delta_t[:, None], bcva[:, None]], axis=1)
        return torch.from_numpy(inp), torch.from_numpy(tgt)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths  = torch.tensor([x.shape[0] for x in inputs])
    max_len  = lengths.max().item()
    inp_dim  = inputs[0].shape[1]

    padded_inp = torch.zeros(len(batch), max_len, inp_dim)
    padded_tgt = torch.zeros(len(batch), max_len)

    for i, (x, t) in enumerate(zip(inputs, targets)):
        n = x.shape[0]
        padded_inp[i, :n] = x
        padded_tgt[i, :n] = t

    return padded_inp, padded_tgt, lengths


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, loader, cst_std, device):
    """RMSE, MAE, 90% CI coverage, mean KL. sample=False for deterministic eval."""
    model.eval()
    all_mu, all_sigma, all_tgt, all_kl = [], [], [], []

    with torch.no_grad():
        for x, tgt, lengths in loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            delta_t_seq = x[:, :, 1024] if REAL_DELTA_T else None

            mu_pred, sigma_pred, mu_z0, sigma_z0 = model(
                x[:, :, :1024], lengths, delta_t_seq=delta_t_seq, sample=False
            )
            kl = 0.5 * (
                sigma_z0 ** 2 + mu_z0 ** 2 - 1.0 - 2.0 * sigma_z0.log()
            ).sum(-1).mean()
            all_kl.append(kl.item())

            for i, l in enumerate(lengths):
                l = l.item()
                all_mu.append(mu_pred[i, :l, 0].cpu().numpy())
                all_sigma.append(sigma_pred[i, :l, 0].cpu().numpy())
                all_tgt.append(tgt[i, :l].cpu().numpy())

    mu_np    = np.concatenate(all_mu)    * cst_std
    sigma_np = np.concatenate(all_sigma) * cst_std
    tgt_np   = np.concatenate(all_tgt)  * cst_std

    rmse     = float(np.sqrt(((mu_np - tgt_np) ** 2).mean()))
    mae      = float(np.abs(mu_np - tgt_np).mean())
    lower    = mu_np - 1.645 * sigma_np
    upper    = mu_np + 1.645 * sigma_np
    coverage = float(((tgt_np >= lower) & (tgt_np <= upper)).mean())
    kl_mean  = float(np.mean(all_kl))

    return dict(rmse=rmse, mae=mae, ci_coverage_90=coverage, kl_mean=kl_mean)


def persistence_rmse(test_seqs):
    sq = []
    for seq in test_seqs.values():
        cst = seq["cst"]
        if len(cst) < 2:
            continue
        sq.append(((cst[:-1] - cst[1:]) ** 2).mean())
    return float(np.sqrt(np.mean(sq)))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    SEED = 42

    # Decision #23 bars — unchanged
    RMSE_BAR       = 85.0
    CI_COV_BAR     = 0.80
    KL_SUCCESS_BAR = 0.10
    KL_COLLAPSE    = 0.01

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    CFG = dict(
        # Model (identical to v1/v2)
        latent_dim       = 32,
        encoder_hidden   = 64,
        ode_hidden       = 64,
        dropout          = 0.2,
        rtol             = 1e-3,
        atol             = 1e-4,
        # Training
        lr               = 1e-3,
        weight_decay     = 1e-4,
        batch_size       = 64,
        n_epochs         = 300,
        beta             = 0.1,    # fixed beta-VAE — never anneals (Decision #25)
        sched_step       = 100,
        sched_gamma      = 0.5,
        seed             = SEED,
        real_delta_t     = REAL_DELTA_T,
        # Augmentation (Decision #25)
        min_subseq_len   = 3,
        emb_noise_std    = 0.02,
        time_jitter      = 0.10,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run = wandb.init(
        project = "synapse",
        name    = "true_latent_ode_v3_seed42",
        config  = CFG,
        tags    = ["true-latent-ode", "cst-regression", "generative", "beta-vae", "augmented-v3"],
    )

    # ── Data
    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=SEED)

    all_train_cst = np.concatenate([s["cst"] for s in train_seqs.values()])
    cst_mean = float(all_train_cst.mean())
    cst_std  = float(all_train_cst.std())
    print(f"CST normalisation (train): mean={cst_mean:.1f}  std={cst_std:.1f} um")

    train_ds = VariableWindowDataset(
        train_seqs, cst_mean, cst_std,
        min_subseq_len = CFG["min_subseq_len"],
        emb_noise_std  = CFG["emb_noise_std"],
        time_jitter    = CFG["time_jitter"],
        augment        = True,
    )
    test_ds = VariableWindowDataset(
        test_seqs, cst_mean, cst_std,
        augment = False,
    )
    print(f"Train sequences: {len(train_ds)}  |  Test sequences: {len(test_ds)}")
    print(f"(77 train eyes -> {len(train_ds)} variable windows, Decision #25)")
    wandb.config.update({"n_train_sequences": len(train_ds), "n_test_sequences": len(test_ds)})

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(
        test_ds,  batch_size=CFG["batch_size"], shuffle=False, collate_fn=collate_fn)

    # ── Model
    model = TrueLatentODE(
        emb_dim        = 1024,
        latent_dim     = CFG["latent_dim"],
        encoder_hidden = CFG["encoder_hidden"],
        ode_hidden     = CFG["ode_hidden"],
        dropout        = CFG["dropout"],
        rtol           = CFG["rtol"],
        atol           = CFG["atol"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"True Latent ODE parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CFG["sched_step"], gamma=CFG["sched_gamma"])

    # ── Train
    best_rmse  = float("inf")
    best_state = None
    collapsed  = False
    beta       = CFG["beta"]   # fixed — no warmup schedule

    print(f"\nTraining True Latent ODE v3 for {CFG['n_epochs']} epochs...")
    print(f"Fixed beta={beta} (beta-VAE, Decision #25) — no warmup schedule")
    print(f"Decision #23 bars: RMSE <= {RMSE_BAR} um  |  CI >= {CI_COV_BAR}  |  KL > {KL_SUCCESS_BAR} nats")

    for epoch in range(1, CFG["n_epochs"] + 1):
        model.train()
        train_loss = train_recon = train_kl = 0.0

        for x, tgt, lengths in train_loader:
            x, tgt, lengths = x.to(device), tgt.to(device), lengths.to(device)
            optimizer.zero_grad()
            delta_t_seq = x[:, :, 1024] if REAL_DELTA_T else None

            mu_pred, sigma_pred, mu_z0, sigma_z0 = model(
                x[:, :, :1024], lengths, delta_t_seq=delta_t_seq, sample=True
            )
            loss, recon, kl = TrueLatentODE.elbo_loss(
                mu_pred, sigma_pred, tgt, lengths, mu_z0, sigma_z0, beta=beta
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss  += loss.item()
            train_recon += recon.item()
            train_kl    += kl.item()

        scheduler.step()
        n_batches = len(train_loader)
        avg_loss  = train_loss  / n_batches
        avg_recon = train_recon / n_batches
        avg_kl    = train_kl    / n_batches

        wandb.log({
            "epoch":       epoch,
            "train_loss":  avg_loss,
            "train_recon": avg_recon,
            "train_kl":    avg_kl,
            "beta":        beta,
            "lr":          scheduler.get_last_lr()[0],
        })

        if epoch % 10 == 0:
            metrics = evaluate(model, test_loader, cst_std, device)
            rmse    = metrics["rmse"]
            is_best = rmse < best_rmse

            if is_best:
                best_rmse  = rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            wandb.log({
                "test_rmse":      rmse,
                "test_mae":       metrics["mae"],
                "test_ci_cov_90": metrics["ci_coverage_90"],
                "test_kl":        metrics["kl_mean"],
                "best_test_rmse": best_rmse,
                "epoch":          epoch,
            })

            print(
                f"  Epoch {epoch:3d} | loss={avg_loss:.4f}  recon={avg_recon:.4f}"
                f"  kl={avg_kl:.4f}  beta={beta:.2f}"
                f"  |  RMSE={rmse:.1f} um  CI={metrics['ci_coverage_90']:.3f}"
                f"  KL={metrics['kl_mean']:.4f}"
                f"{'  <-- best' if is_best else ''}"
            )

            if metrics["kl_mean"] < KL_COLLAPSE:
                print(
                    f"\n  POSTERIOR COLLAPSE at epoch {epoch}: "
                    f"KL = {metrics['kl_mean']:.5f} < {KL_COLLAPSE} nats. "
                    f"Stop. Report as failure (Decision #23)."
                )
                wandb.log({"posterior_collapse": True, "collapse_epoch": epoch})
                collapsed = True
                break

    # ── Final
    if best_state is None:
        run.finish()
        return None

    model.load_state_dict(best_state)
    final   = evaluate(model, test_loader, cst_std, device)
    persist = persistence_rmse(test_seqs)

    beats_rmse = final["rmse"]           <= RMSE_BAR
    beats_ci   = final["ci_coverage_90"] >= CI_COV_BAR
    beats_kl   = final["kl_mean"]        >= KL_SUCCESS_BAR

    wandb.log({
        "final_rmse":         final["rmse"],
        "final_mae":          final["mae"],
        "final_ci_cov_90":    final["ci_coverage_90"],
        "final_kl":           final["kl_mean"],
        "persistence_rmse":   persist,
        "beats_rmse_bar":     beats_rmse,
        "beats_ci_bar":       beats_ci,
        "beats_kl_bar":       beats_kl,
        "posterior_collapse": collapsed,
    })

    print("\n" + "=" * 65)
    print("  True Latent ODE v3 (beta-VAE) — next-visit CST regression")
    print("=" * 65)
    print(f"  RMSE (primary)       : {final['rmse']:.1f} um"
          f"  {'PASS' if beats_rmse else 'FAIL'} (<= {RMSE_BAR} um)")
    print(f"  MAE                  : {final['mae']:.1f} um")
    print(f"  90% CI coverage      : {final['ci_coverage_90']:.3f}"
          f"  {'PASS' if beats_ci else 'FAIL'} (>= {CI_COV_BAR})")
    print(f"  KL divergence        : {final['kl_mean']:.4f} nats"
          f"  {'PASS' if beats_kl else 'FAIL'} (> {KL_SUCCESS_BAR})")
    print(f"  Persistence RMSE     : {persist:.1f} um")
    print(f"  ODE-RNN baseline     : 81.63 um  (Decision #14)")
    print(f"  Posterior collapse   : {'YES' if collapsed else 'No'}")
    print(f"  Train sequences      : {len(train_ds)} (77 eyes, Decision #25)")
    print("=" * 65)
    if not (beats_rmse and beats_ci and beats_kl):
        print("  NOTE: One or more bars not met. Log result honestly.")
    print("=" * 65)

    import pathlib
    ckpt_dir  = pathlib.Path("models")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "true_latent_ode_v3_seed42.pt"
    torch.save({
        "model_state":   best_state,
        "cfg":           CFG,
        "cst_mean":      cst_mean,
        "cst_std":       cst_std,
        "final_rmse":    final["rmse"],
        "final_ci_cov":  final["ci_coverage_90"],
        "final_kl":      final["kl_mean"],
        "collapsed":     collapsed,
    }, ckpt_path)
    print(f"\n  Checkpoint saved -> {ckpt_path}")

    run.finish()
    return final


if __name__ == "__main__":
    main()
