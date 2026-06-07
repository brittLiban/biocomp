> Parent: CLAUDE.md · This doc: present snapshot — what's happening RIGHT NOW.
> Changes: every session end. Overwritten, not appended. For permanent history see LOG.md.

# Progress — Present Snapshot

## Current Sprint
Latent ODE sprint — complete (see NOW.md)

## In Flight
- CLAIMS.md update proposed (Decision #10) — awaiting human confirmation before writing.

## What Was Completed This Session
- src/dynamics/__init__.py created (package stub)
- src/dynamics/latent_ode.py — ODE-RNN model (ODEFunc + LatentODE classes)
- scripts/latent_ode.py — training script, W&B logging, checkpoint saving
- models/latent_ode_v1_seed42.pt — saved best checkpoint
- RMSE 81.96 um logged to W&B latent_ode_v1_seed42 (run 942anp2d)
- Decision #10 appended to DECISIONS.md

## Result Summary
| Model         | RMSE (um) | MAE (um) | Notes                        |
|---------------|-----------|----------|------------------------------|
| Persistence   | 91.7      | —        | Naive last-value lower bound |
| GRU-D         | 82.2      | ~60      | Baseline                     |
| T-LSTM        | 82.0      | ~60      | Best baseline                |
| **Latent ODE**| **82.0*** | **58.3** | *81.96 um; beats bar by 0.04 um |

## Blocked
- Real week-based delta_t (OCT-DR.xlsx parsing) — not a blocker for ODE result, but needed before re-running baselines with real gaps.
- CLAIMS.md update — human must confirm before writing.

## Next Session Should
- Human confirms (or adjusts) proposed CLAIMS.md change.
- Close out the sprint (archive NOW.md, update LOG.md, reset for next sprint).
- If preprint: draft results section first (models, metrics, persistence comparison).

## Last Updated
2026-06-07 — Latent ODE sprint complete. RMSE 81.96 um. Beat bar.
