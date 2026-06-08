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

## Technical
- [x] Is OLIVES temporal structure rich enough for a latent ODE? YES — 97.9% of eyes ≥4 visits (2026-05-31 → DECISIONS.md #3)
- [ ] RETFound frozen vs fine-tuned for v1 — defer until after encoding works.
- [ ] OLIVES OCT modality: does RETFound_cfp (colour fundus) work well on OCT slices, or do we need a separate OCT encoder? Check embedding quality after encoding. May need RETFound_oct variant.
- [x] RETFound embedding dim: **1024** (ViT-Large confirmed via probe_embedding_dim() during encoding, 2026-06-07)
- [x] EyePACS patient/eye IDs: public Kaggle release is **classification-only** — image-level labels, no patient-level temporal structure in available metadata. Used as classification training set only. (2026-06-07)
