> Parent: CLAUDE.md · Related: docs/OPEN_QUESTIONS.md
> This doc: log of non-obvious choices and WHY. Stops re-litigating settled questions.
> Changes: append immediately when a real decision is made. Newest at top.

# Decisions

<!-- Format:
## #N — YYYY-MM-DD — <decision in one line>
Context: <what prompted it>
Choice: <what we decided>
Why: <reasoning>
Alternatives rejected: <what we didn't pick and why>
-->

## #5 — 2026-06-01 — OLIVES dataloader: adapt authors' code, not build from scratch
Context: explored the official OLIVES GitHub repo (https://github.com/olivesgatech/OLIVES_Dataset) in full.
Choice: adapt the authors' `Time-Series Treatment Analysis` dataloader code as the reference implementation for our OLIVES temporal dataloader.
Why: authors already solved the hard parts — visit extraction from file paths (`W1`, `W4`, etc.), grouping by Eye_ID, sorting chronologically, stacking OCT B-scans into 3D volumes, pre-computing `.npy` files for efficiency. No reason to reinvent this.
Sprint: Weeks 4-5 (Baselines sprint) — OLIVES dataloader is needed to run GRU-D/T-LSTM/Cox baselines. Not needed for the EyePACS/RETFound encoding sprint (Weeks 3-4).
Alternatives rejected: building from scratch — wastes time when a reference implementation exists.

## #4 — 2026-06-01 — HuggingFace OLIVES download: use for encoder pretraining only
Context: downloaded OLIVES from HuggingFace (28.86 GB, Parquet format) after Zenodo throttling. Discovered it strips file paths and visit ordering; only preserves Eye_ID, biomarkers, BCVA, CST, Disease Label.
Choice: use HuggingFace data for encoder pretraining (EyePACS/RETFound sprint, Weeks 3-4) where temporal order doesn't matter. Still need Zenodo zip for dynamics modeling and baselines.
Why: HuggingFace images are usable for representation learning. For temporal modeling we need the original folder structure that encodes visit numbers — only the Zenodo zip has this.
Action required: download Zenodo OLIVES.zip before Weeks 4-5 baselines sprint begins. Run overnight with `curl.exe` (Zenodo throttles to ~3 Mbps regardless of connection speed).

## #3 — 2026-05-31 — Model class decision: latent ODE is viable for OLIVES
Context: OLIVES feasibility audit completed. Needed to choose model class before building anything.
Choice: **Latent ODE** — proceed with tODE as the core dynamics model for Year 1.
Why: 94/96 eyes (97.9%) have ≥4 visits; mean 16.6, max 27. Temporal structure far exceeds the ≥4-visit threshold. 16 OCT biomarkers present with ~0% missing. Two sub-studies (TREX DME 56 eyes, Prime_FULL 40 eyes) can be unified on a relative time clock.
Alternatives rejected: GRU-D / T-LSTM / short-horizon baselines — appropriate for 2-3 visit datasets; OLIVES is clearly rich enough to justify the full latent ODE approach. Baselines will still be built for comparison, but are no longer the primary model.
Open item: full longitudinal biomarker alignment still needs `full_labels/OCT-DME.xlsx` parsing and treatment event alignment (follow-up task, not a blocker for this decision).

## #1 — Month 0 — Adopt a layered docs architecture for AI-driven development
Context: building primarily with AI (Claude, sometimes GPT) across many sessions.
Choice: guard-rail / how-we-build / plan / focus / memory doc layers, with archive + ritual.
Why: keeps an AI agent on-mission across sessions; the docs are the shared memory between models.
Alternatives rejected: single big doc (AI ignores it) and no docs (AI drifts).

## #2 — Month 0 — OLIVES-first, feasibility audit before any model code
Context: public DR data is mostly cross-sectional; OLIVES is the only open longitudinal set.
Choice: audit OLIVES temporal structure before building any dynamics model.
Why: a latent ODE on temporally-thin data is wasted effort; decide model class on evidence.
Alternatives rejected: jumping straight to model-building (risks months on an unprovable claim).
