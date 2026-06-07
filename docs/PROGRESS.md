> Parent: CLAUDE.md · This doc: present snapshot — what's happening RIGHT NOW.
> Changes: every session end. Overwritten, not appended. For permanent history see LOG.md.

# Progress — Present Snapshot

## Current Sprint
Real delta_t sprint (see NOW.md) — code and runs complete; awaiting human decision on CLAIMS.md.

## In Flight
- Nothing in flight.

## Completed This Sprint
- Task 1: OCT-DR.xlsx audited. Finding: no per-visit timing (demographics only). OCT-DME.xlsx Week columns ~98% NaN. Real timing only in file-path visit keys. Documented in DATA.md.
- Task 2: `build_sequences()` extended with `week_gaps` field (v2 cache). Prime: diff(visit_nums)/4; TREX: 1.0.
- Task 3: All three scripts updated — REAL_DELTA_T=True, W&B added to GRU-D/T-LSTM.
- Task 4: All three temporal models re-run with real delta_t, results logged to W&B.
- Task 5: 2×3 comparison table built (see Decision #11 in DECISIONS.md).
- Task 6: Decision #11 appended to DECISIONS.md.
- Task 7: CLAIMS.md update — proposed, awaiting human confirmation (see below).

## Full Results (Real Delta-T Sprint)

| Model | Ordinal RMSE | Real-ΔT RMSE | Change | Real-ΔT MAE |
|---|---|---|---|---|
| GRU-D | 82.2 um | 84.2 um | +2.0 um worse | 58.9 um |
| T-LSTM | 82.0 um | 85.0 um | +3.0 um worse | 59.0 um |
| Latent ODE | 81.96 um | 83.5 um | +1.5 um worse | 59.2 um |
| Persistence | 91.7 um | 91.7 um | no change | — |

Finding: Real timing degraded all three models. ODE degraded least (+1.5 um) vs recurrent models (+2-3 um). Dynamics thesis is NOT confirmed by real timing — the "ODE outperforms recurrents with real irregular time" hypothesis is not supported on this dataset. Ordinal results remain the canonical comparison.

## Blocked
- CLAIMS.md update requires human confirmation (sprint rule).
- Preprint draft (next sprint) can start; real delta-t is a complete negative result that belongs in the paper.

## Last Updated
2026-06-07 — Real delta-t sprint code and runs complete.
