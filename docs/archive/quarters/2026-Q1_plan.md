> Parent: docs/YEAR_1.md · Constitution: CLAUDE.md · Focus: docs/NOW.md
> This doc: the current quarter at high level. Each chunk is rolled down into NOW.md when reached.
> Changes: per quarter (archive to docs/archive/quarters/ when Q2 begins).

# Q1 Plan (Months 1-3) — Foundation + Feasibility

## Quarter Goal
Climb toward Rung 1-2: stand up reproducible infrastructure, complete the OLIVES
feasibility audit, build baselines, and start the Track 2 data-access process.
NO dynamics-model code until the audit is complete and the model class is decided.

## Work Chunks (roll each down into NOW.md when reached)

### Weeks 1-2 — OLIVES Feasibility Audit  [FIRST — do before anything else]
Download OLIVES. Answer the 4 questions: file structure, temporal depth per eye,
alignment of imaging+biomarkers+treatment, size of clean longitudinal subset.
Decision gate: ≥4 visits/eye for most eyes → latent ODE viable; 2-3 → simpler baselines.
Done when: DATA.md has the audit findings + the model-class decision is logged.

### Weeks 2-3 — Reproducible Infrastructure  [parallel with audit]
Git repo hygiene, W&B setup, environment specs, requirements pinned, seed utilities.
Done when: a trivial logged experiment runs reproducibly end-to-end.

### Weeks 3-4 — EyePACS + Messidor Pipeline
Data loaders for EyePACS + Messidor. RETFound integration. Encode images to embeddings.
Done when: embeddings cached + reloadable; shapes validated.

### Weeks 4-5 — Baselines
Logistic regression, GRU-D, T-LSTM, Cox survival on the appropriate tasks.
Done when: results table exists; all baselines reproducible. These are what we must beat.

### Week 1 (async, ongoing) — Track 2: UK Biobank Application
Start the UK Biobank application. Runs in background over months.
Done when: application submitted.

## End-of-Quarter Gate (Gate 1, Month 3)
Infra works + baselines run + OLIVES audited + model class decided + UKB app in.
Outcome recorded in MILESTONES.md: CONTINUE / ITERATE / PIVOT.

## Hard Rules This Quarter
- No dynamics-model code before the audit is done.
- Baselines before any fancy model.
- Every claim scoped to what the data supports (see CLAIMS.md).
