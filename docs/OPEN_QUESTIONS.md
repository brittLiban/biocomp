> Parent: CLAUDE.md · Related: docs/DECISIONS.md (resolved questions graduate into decisions)
> This doc: running list of unresolved unknowns. Append when one surfaces; check off when answered.
> Changes: continuously, as questions arise and resolve.

# Open Questions

Status: [ ] open · [x] answered (then move the answer to DECISIONS.md if it's a choice)

## Data
- [x] OLIVES temporal depth — 94/96 eyes have ≥4 visits; mean 16.6; latent ODE viable (2026-05-31)
- [ ] Do OCT scans share a timestamp/clock with the paired fundus image?
- [ ] Can injection/treatment dates be aligned to the imaging timeline? (`full_labels/OCT-DR.xlsx` has treatment arm + injection column; needs per-visit mapping — task for Weeks 4-5 baselines sprint)
- [ ] Does `full_labels/OCT-DME.xlsx` (195-col wide table, VisitNums 1–28) contain full longitudinal biomarkers for all visits? If so, pivot to long format for use.
- [ ] Does ACCORD EYE include raw image files or only derived datasets?
- [ ] UK Biobank: exact independent-researcher eligibility + current fee?

## Strategy
- [ ] Will a UW affiliation materialize, and is Dr. Dargahi open to a bridge thesis? (bonus, not critical-path)
- [ ] Which controlled-data route is most realistic first: UK Biobank, EyePACS partnership, or local clinic?

## Preprint / Evaluation
- [x] **METRIC INCONSISTENCY (2026-06-08) — RESOLVED: report both metrics (Decision #13).** Persistence RMSE was computed with eye-weighted metric (sqrt of mean-of-per-eye-MSEs = 91.7 μm). Model RMSEs were computed with timestep-weighted metric (sqrt of mean-over-all-timesteps = 82.x μm). These measure different things. Diagnostic (scripts/diagnose_metrics.py, 2026-06-08):
  - Eye-weighted: persistence 91.7 μm, GRU-D 75.2, T-LSTM 73.0, ODE 73.0 → models beat persistence ✓
  - Timestep-weighted: persistence 55.0 μm, GRU-D 82.2, T-LSTM 82.0, ODE 82.0 → models do NOT beat persistence ✗
  - Timing experiment direction (ODE −0.4/−0.6 μm, recurrents +1.2–3.0 μm) holds under BOTH metrics ✓
  - Root cause: a few eyes (eye 4: 1 visit, persistence 291 μm; eye 12: 2 visits, persistence 190 μm) dominate the eye-weighted persistence. Models do better on stable long-sequence eyes but fail on volatile short-sequence eyes. Models beat persistence on only 8/19 test eyes (42%).
  - Options: (A) eye-weighted for all — model numbers change, story holds; (B) timestep-weighted for all — reframe around timing experiment only; (C) report both explicitly. **Human must decide.**

## Technical
- [x] Is OLIVES temporal structure rich enough for a latent ODE? YES — 97.9% of eyes ≥4 visits (2026-05-31 → DECISIONS.md #3)
- [ ] RETFound frozen vs fine-tuned for v1 — defer until after encoding works.
- [ ] OLIVES OCT modality: does RETFound_cfp (colour fundus) work well on OCT slices, or do we need a separate OCT encoder? Check embedding quality after encoding. May need RETFound_oct variant.
- [x] RETFound embedding dim: **1024** (ViT-Large confirmed via probe_embedding_dim() during encoding, 2026-06-07)
- [x] EyePACS patient/eye IDs: public Kaggle release is **classification-only** — image-level labels, no patient-level temporal structure in available metadata. Used as classification training set only. (2026-06-07)
