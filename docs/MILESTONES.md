> Parent: docs/YEAR_1.md · Constitution: CLAUDE.md
> This doc: flat checklist of all key milestones + gates across 3 years.
> Changes: tick boxes + record gate outcomes as they happen.

# Synapse Milestones

Status key: [ ] not started · [~] in progress · [x] done · [!] blocked

## Year 1 — Prove the Architecture

### Q1 (Months 1-3)
- [x] OLIVES downloaded + feasibility audit complete (2026-06-01)
- [x] Reproducible infra (Git, W&B, env specs) (2026-06-01)
- [x] EyePACS + Messidor pipeline + RETFound integration (2026-06-07)
- [x] Baselines: logistic, GRU-D, T-LSTM, Cox survival (2026-06-07)
- [ ] UK Biobank application submitted (Track 2) — NOT STARTED
- [x] **GATE 1 (Month 3):** infra + baselines + audit done → outcome: CONTINUE (2026-06-07)

### Q2 (Months 4-6)
- [x] Per-eye sequence dataset from OLIVES (2026-06-07)
- [x] Latent ODE prototype on OLIVES (2026-06-07) — RMSE 81.96 um, matches baselines
- [x] Real delta_t re-run — complete (2026-06-07) — ODE 81.6 um with real timing, beats bar
- [ ] Treatment-conditioning experiments
- [x] Encoder external validation (Messidor-2) — AUC 0.77 OOD, frozen linear probe (2026-06-07, Decision #12)
- [ ] Uncertainty + calibration
- [ ] EyePACS partnership inquiry sent (Track 2)
- [ ] BioLINCC ACCORD request (Track 2)
- [x] **GATE 2 (Month 6):** architecture functions on real temporal data → outcome: CONTINUE — ODE-RNN matches baselines (82.0 um), Gate 2 passed in prototype form. Real delta_t and preprint to confirm. (2026-06-07)

### Q3 (Months 7-9)
- [ ] JEPA-style fine-tuning (compare vs frozen RETFound)
- [ ] Clinical encoder (OLIVES biomarkers)
- [ ] Preprint draft
- [ ] Retina-clinic pilot outreach (Track 2)

### Q4 (Months 10-12)
- [ ] Preprint submitted to arXiv
- [ ] Conference/journal submission (MICCAI, NeurIPS, JAMA Ophth)
- [ ] First investor/partner conversations
- [ ] Scale-data path secured
- [ ] **GATE 3 (Month 12):** earned the right to pursue seriously → outcome: ____

## Year 2 — Validate at Scale
- [ ] Controlled longitudinal data integrated
- [ ] Year 1 results replicated at scale
- [ ] Multimodal HAWVA fusion working
- [ ] Treatment-conditioned model validated vs literature
- [ ] First API endpoints
- [ ] Seed closed ($500K-$2M) or major grant
- [ ] AMD model begun (second disease)
- [ ] **GATE 4 (Month 18):** real data confirms thesis → outcome: ____
- [ ] **GATE 5 (Month 24):** real business or research project → outcome: ____

## Year 3 — Platform & Partnerships
- [ ] DR + AMD validated; Glaucoma pilot
- [ ] Full HAWVA Bayesian fusion
- [ ] Active inference decision support
- [ ] Synthetic cohort generation
- [ ] 2-3 pharma partnerships active
- [ ] $1-5M ARR or comparable funding
- [ ] Series A momentum or strategic partnership
- [ ] **GATE 6 (Month 36):** platform thesis working → outcome: ____
