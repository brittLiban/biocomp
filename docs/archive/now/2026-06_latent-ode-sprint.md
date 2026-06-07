> Parent: docs/Q1_PLAN.md (Q2, Track 1) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Latent ODE Sprint

## Goal
Build the first working latent ODE on OLIVES sequences and beat the baseline RMSE of 82.0 um
on next-visit CST prediction. This is Gate 2 (Month 6): does the architecture actually work
on real temporal data?

## Why This Now
Baselines are done. The bar is locked. The only thing left to do before claiming Rung 2
is showing the latent ODE beats strong temporal baselines on the same held-out split.

## Inputs
- `data/processed/olives_sequences.pkl` — 96 eyes, visit-averaged embeddings, BCVA, CST
- `src/data/olives.py` — `build_sequences()`, `split_by_eye()`, `OLIVESSequenceDataset`
- Baseline scripts in `scripts/` — for comparison and code patterns
- `torchdiffeq` — neural ODE integration library
- ODE target: **RMSE < 82.0 um** on next-visit CST, test split seed=42 (Decision #9)

## Critical Design Decisions (locked before coding)

**Architecture:**
- Encoder: linear projection 1024 → latent_dim (e.g. 32)
- ODE function: small MLP on latent state (2-layer, hidden 64)
- Decoder: linear latent_dim → 1 (CST prediction)
- Solver: `dopri5` (adaptive step, torchdiffeq default)
- Time input: ordinal visit index (same as baselines — upgrade to real gaps after first result)

**Same train/test split as baselines:**
- `split_by_eye(seqs, test_frac=0.2, seed=42)` — 77 train / 19 test eyes
- Any deviation invalidates the RMSE comparison

**Same target and metric:**
- Predict next-visit CST (continuous, μm)
- Primary metric: RMSE. Must be < 82.0 um.
- Secondary: MAE

## Tasks
1. **Install torchdiffeq** — `pip install torchdiffeq`. Verify ODE solver runs.
2. **Build latent ODE model** — encoder, ODE func, decoder. Start minimal (latent_dim=32).
3. **Training loop** — masked MSE on next-visit CST, same collate_fn as GRU-D.
4. **First result** — log to W&B `latent_ode_v1_seed42`. Compare RMSE to 82.0 um baseline.
5. **Iterate if needed** — tune latent_dim, ODE depth, regularization. Log each run.
6. **Log result to DECISIONS.md** — whether ODE beat the bar and by how much.
7. **Update CLAIMS.md** — only if RMSE < 82.0 um confirmed on held-out test set.

## Done When
- Latent ODE trains without error ✓
- Test RMSE logged to W&B ✓
- RMSE vs. 82.0 um baseline documented in DECISIONS.md ✓
- CLAIMS.md updated if result warrants it (human confirms) ✓
- Clean git commit ✓

## Hard Constraints
- Test split must be `split_by_eye(seqs, test_frac=0.2, seed=42)` — same as baselines
- No cherry-picking runs — first W&B run that completes is the result
- CLAIMS.md update is human-confirmed only — AI proposes, human approves
- delta_t stays ordinal until baselines are re-run with real gaps (don't mix methodologies)

## Next Up
If ODE beats bar: preprint draft (results section first). If ODE fails: diagnose why —
too little data? wrong architecture? pivot to simpler dynamics model (e.g. NODE with GRU encoder).
