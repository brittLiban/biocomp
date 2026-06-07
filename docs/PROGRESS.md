> Parent: CLAUDE.md · This doc: present snapshot — what's happening RIGHT NOW.
> Changes: every session end. Overwritten, not appended. For permanent history see LOG.md.

# Progress — Present Snapshot

## Current Sprint
Real delta_t sprint (see NOW.md) — all runs complete including ODE v2 correction.
Awaiting human decision on CLAIMS.md directional evidence section.

## In Flight
- Nothing in flight.

## Completed This Sprint
- Task 1: OCT-DR.xlsx audited. Finding: no per-visit timing (demographics only). OCT-DME.xlsx Week columns ~98% NaN. Real timing only in file-path visit keys. Documented in DATA.md.
- Task 2: `build_sequences()` extended with `week_gaps` field (v2 cache). Prime: diff(visit_nums)/4; TREX: 1.0.
- Task 3: All three scripts updated — REAL_DELTA_T=True, W&B added to GRU-D/T-LSTM.
- Task 4: All three temporal models re-run with real delta_t, results logged to W&B.
- Task 5: 2×3 comparison table built and corrected (see Decision #11 v2 in DECISIONS.md).
- Task 6: Decision #11 updated with corrected ODE result (grouped per-example odeint).
- Task 7: CLAIMS.md updated — directional evidence section added, awaiting human confirmation.
- Task 8: ODE v2 correction — replaced batch-mean odeint with grouped per-example odeint (reproducibility fix).

## Full Results (Real Delta-T Sprint, FINAL CORRECTED)

| Model | Ordinal RMSE | Real-ΔT RMSE | Change | Real-ΔT MAE |
|---|---|---|---|---|
| GRU-D | 82.2 um | 84.2 um | +2.0 um worse | 58.9 um |
| T-LSTM | 82.0 um | 85.0 um | +3.0 um worse | 59.0 um |
| Latent ODE | 81.96 um | **81.6 um** | **-0.4 um better** | 58.0 um |
| Persistence | 91.7 um | 91.7 um | no change | — |

W&B: grud_realdelta_seed42, tlstm_realdelta_seed42, ode_realdelta_v2_seed42

Finding: With correct per-example ODE integration, the ODE IMPROVES marginally on real delta-t while recurrents degrade. This is directional evidence for the ODE's structural advantage with irregular time. Still 19 test eyes — not definitive.

## Blocked
- CLAIMS.md "directional evidence" section requires human confirmation before promotion to CAN CLAIM.

## Last Updated
2026-06-07 — ODE v2 correction complete; all sprint results finalized.
