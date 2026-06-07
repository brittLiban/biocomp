> Parent: CLAUDE.md · Related: docs/OPEN_QUESTIONS.md
> This doc: log of non-obvious choices and WHY. Stops re-litigating settled questions.
> Changes: append immediately when a real decision is made. Newest at top.

# Decisions

## #9 — 2026-06-07 — ODE minimum target: RMSE < 82.0 um on next-visit CST (pre-committed, no moving)
Context: baselines sprint complete. GRU-D 82.2 um, T-LSTM 82.0 um on test eyes. Need a locked bar before building the latent ODE.
Choice: ODE must achieve RMSE < 82.0 um on next-visit CST (same test split, seed=42, same normalisation). This number is fixed now and cannot be revised post-hoc.
Why: pre-committing the bar prevents cherry-picking. If the ODE cannot beat strong temporal baselines, the dynamics thesis is not demonstrated on this data.
Caveat: delta_t is currently ordinal (1.0 steps, not real week gaps). Once real gaps are parsed from OCT-DR.xlsx, baselines should be re-run. The ODE bar updates to match the re-run baselines — but only if baselines are re-run first, before ODE.
Alternatives rejected: setting bar post-hoc after seeing ODE results — invalidates the comparison.

## #8 — 2026-06-07 — Temporal baseline target: CST regression (continuous) not binary threshold
Context: OLIVES `Disease Label` is patient-level (static across all visits per eye — DME vs healthy control). A temporal model predicting "next-visit disease state" trivially scores ~1.0 by copying current state; no dynamics are learned.
Choice: temporal baselines (GRU-D, T-LSTM) predict **next-visit CST** (continuous, μm) via regression. Metrics: RMSE (primary), MAE (secondary).
Why: (1) Latent ODE will target continuous physiological state — RMSE baseline enables direct comparison. (2) Binary CST threshold (e.g. >300 μm) is arbitrary — a prediction of 298 when truth is 302 is meaningfully correct but scored wrong. (3) Binary threshold derivable post-hoc from continuous predictions at zero information cost.
Results table splits cleanly: disease classification (logistic reg, AUC) vs. treatment-response dynamics (GRU-D/T-LSTM/ODE, RMSE/MAE).
Alternatives rejected: binary CST threshold (Option B) — mixes arbitrary clinical cutoff into evaluation; not what the ODE will optimize.

## #7 — 2026-06-07 — OLIVES visit alignment: (Eye_ID, BCVA, CST) join key, not File_Path
Context: HuggingFace OLIVES `disease_classification` config has no `File_Path` field. Encoding (v3 notebook) saved only Eye_ID and Disease Label per embedding — no filename to join to Clinical_Data_Images.xlsx.
Choice: join embeddings to visits using (Eye_ID, BCVA, CST) as composite key. Drops ~5.9% of B-scans where key maps to >1 visit (BCVA/CST unchanged between consecutive visits). Averages all B-scan embeddings within a visit into one 1024-d vector.
Why: BCVA and CST are per-visit clinical measurements recorded identically in both the HuggingFace metadata and Clinical_Data_Images.xlsx. For 94.1% of B-scans the triple is unambiguous. Visit-averaged embeddings are the right input unit for temporal models anyway.
Alternatives rejected: re-encoding with filename tracking — would require another full Colab run; the join approach recovers 94.1% of data cleanly.

<!-- Format:
## #N — YYYY-MM-DD — <decision in one line>
Context: <what prompted it>
Choice: <what we decided>
Why: <reasoning>
Alternatives rejected: <what we didn't pick and why>
-->

## #6 — 2026-06-02 — Encoding strategy: EyePACS/Messidor for classification, OLIVES for dynamics
Context: peer review of encoding plan flagged that EyePACS has no per-eye temporal structure, and OLIVES has two modalities (fundus + OCT) that may need different encoders.
Choice: use RETFound_cfp (frozen) for EyePACS + Messidor + OLIVES fundus images. Treat EyePACS/Messidor embeddings as classification/representation baselines only. OLIVES fundus embeddings feed into temporal sequences. OLIVES OCT encoder TBD — probe quality first.
Why: clean separation of what each dataset is actually for. Don't overstate EyePACS as temporal data.
Action: probe embedding dim at runtime; never hardcode 1024.

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
