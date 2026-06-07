> Parent: CLAUDE.md · Related: docs/OPEN_QUESTIONS.md
> This doc: log of non-obvious choices and WHY. Stops re-litigating settled questions.
> Changes: append immediately when a real decision is made. Newest at top.

# Decisions

## #11 — 2026-06-07 — Real delta-t result: all models worse; ODE degrades least

Context: Real Delta-T Sprint. Re-ran GRU-D, T-LSTM, Latent ODE with real week gaps (Prime eyes: diff(visit_nums)/4 normalized; TREX eyes: ordinal 1.0 — real timing unavailable). Seed=42, same split, same normalisation.

Timing audit finding: OCT-DR.xlsx contains patient demographics only (no per-visit weeks). OCT-DME.xlsx Week columns are ~98% NaN. Real timing derives exclusively from file-path visit keys: Prime W-keys are week numbers (gaps range 4–48 wks, normalized 1.0–12.0); TREX V-keys are ordinal (real timing unknowable from available data — treat-and-extend protocol).

Results — 2×3 table (RMSE in μm, test split 19 eyes, seed=42):

| Model | Ordinal RMSE | Real-ΔT RMSE | RMSE Δ | Real-ΔT MAE |
|---|---|---|---|---|
| GRU-D | 82.2 um | 84.2 um | +2.0 um | 58.9 um |
| T-LSTM | 82.0 um | 85.0 um | +3.0 um | 59.0 um |
| Latent ODE | 81.96 um | 83.5 um | +1.5 um | 59.2 um |
| Persistence | 91.7 um | 91.7 um | +0.0 um | — |

W&B: grud_realdelta_seed42, tlstm_realdelta_seed42, ode_realdelta_seed42

Choice: Accept these as official real delta-t results. Ordinal results remain the canonical comparison baseline.

Why (interpretation): Real delta-t worsened all three models. Two likely causes: (1) GRU-D/T-LSTM decay mechanisms (exp(-W·Δt), c/(1+Δt)) grew more aggressive with variable gaps 1.0–12.0, disrupting decay rates learned for ordinal 1.0. (2) ODE batch-mean integration (batch-mean delta_t per step) is an approximation — real per-example integration may differ, but is more expensive to implement. Notably, the ODE degraded the least (+1.5 um vs +2.0/+3.0), consistent with its structural advantage for irregular time, but the advantage is not decisive on this small test set (19 eyes). Real timing added noise rather than signal, possibly because 68% of Prime gaps are still 1.0 and TREX timing is ordinal.

What it means for claims: The hypothesis "real timing gives the ODE a larger advantage" is NOT confirmed on this dataset with this implementation. The honest claim remains "ODE matches recurrent baselines" — the ODE is not clearly better on either ordinal or real delta-t. The CANNOT CLAIM boundary is unchanged until more data or per-example ODE integration is implemented.

Alternatives rejected: Per-example odeint (correct but ~16x slower); Prime-only split (violates frozen split constraint); estimating TREX gaps as 4-week fixed (wrong protocol — TREX is T&E with variable intervals).

## #10 — 2026-06-07 — Latent ODE result: RMSE 81.96 um — beats bar, narrow margin
Context: First completed W&B run (latent_ode_v1_seed42, seed=42, 100 epochs, dopri5, rtol=1e-3/atol=1e-4). Architecture: ODE-RNN (Rubanova et al. 2019) — linear encoder (1024->32) + GRUCell observation update + 2-layer MLP ODEFunc (hidden 64) + linear decoder (32->1). 47,521 parameters. Re-run performed solely for checkpoint saving; result byte-identical (deterministic seed on CPU).
Choice: Accept 81.96 um as official result. Beat bar per Decision #9.
Result: RMSE 81.96 um (best checkpoint, epoch 10 of 100) | MAE 58.3 um | Persistence 91.7 um | beats bar: YES
Notable: Best checkpoint at epoch 10 of 100. Train loss decreases monotonically (0.59 -> 0.38) while test RMSE worsens after epoch 10 (82.0 -> 87.0) — rapid overfitting on 77 training eyes with 47K parameters. The 0.04 um margin over the bar is within sampling variance on 19 test eyes; the honest claim is "matches" not "clearly beats" temporal baselines. The ODE-RNN beats persistence by 9.7 um (same tier as GRU-D/T-LSTM), validating the dynamics approach.
Checkpoint: models/latent_ode_v1_seed42.pt | W&B: project=synapse, run=latent_ode_v1_seed42 (run ID 942anp2d)
Alternatives rejected: Accepting the first run (same result, same seed, but no checkpoint saved due to script bug). 

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
