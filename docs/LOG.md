> Parent: CLAUDE.md · This doc: permanent append-only history of completed sprints.
> Changes: one entry per completed sprint, NEWEST AT TOP. For present state see PROGRESS.md.

# Log — Completed Sprints

<!-- Newest entries at top. Format:
## YYYY-MM-DD — <sprint name>
<one or two lines: what finished>
Outcome: <result>. → docs/archive/now/YYYY-MM_description.md
Decision(s) logged: <DECISIONS.md ref if any>
-->

## 2026-06-07 — Baselines Sprint (Weeks 4-5)
Built OLIVES visit alignment (join via BCVA/CST, no File_Path in HuggingFace), sequence dataloader, and all four baselines. Discovered Disease Label is patient-level (static); pivoted temporal targets to CST regression. Logged all runs to W&B.
Outcome: **Results table complete. ODE target locked: RMSE < 82.0 um. CLAIMS.md updated.** → docs/archive/now/2026-06_baselines-sprint.md
Decisions logged: DECISIONS.md #7 (OLIVES alignment join key), #8 (CST regression over binary threshold), #9 (ODE target RMSE < 82.0 um — pre-committed, no post-hoc revision)

Results:
| Model | Task | Metric | Result |
|---|---|---|---|
| Logistic Reg | Patient classification | AUC-ROC | 0.9906 |
| GRU-D | Next-visit CST regression | RMSE | 82.2 um |
| T-LSTM | Next-visit CST regression | RMSE | 82.0 um |
| Cox PH | Time to CST normalization | C-index | 0.7955 |
| Persistence | Next-visit CST regression | RMSE | 91.7 um |

## 2026-06-07 — EyePACS + Messidor + RETFound Encoding
Built data pipelines for EyePACS and Messidor. Encoded all three datasets with frozen RETFound (bitfount/RETFound_MAE, ViT-Large, 1024-dim) on Google Colab T4 GPU. CPU encoding crashed after 26 hrs (OOM) — moved to Colab. Multiple disk/space issues worked through. All embeddings validated and saved locally.
Outcome: **3 datasets encoded, 110,109 images total. Embeddings cached to data/processed/embeddings/.** → docs/archive/now/2026-06_eyepacs-messidor-retfound-encoding.md
Decisions logged: DECISIONS.md #4 (HuggingFace use), #5 (authors dataloader), #6 (encoding strategy)

## 2026-06-01 — Reproducible Infrastructure
Created requirements.txt (pinned deps), seed utility, W&B project synapse-v1, and smoke test. Verified two runs with seed=42 produce identical losses. W&B dashboard live.
Outcome: **Infra working. Every future experiment is logged and reproducible.** → docs/archive/now/2026-06_reproducible-infra.md
Decisions logged: none (no new non-obvious choices)

## 2026-06-01 — OLIVES Feasibility Audit
Downloaded OLIVES (31.6 GB Zenodo zip + 6.3 MB labels), extracted both sub-studies (TREX DME: 96,051 files; Prime_FULL: 66,813 files). Audited temporal depth: 94/96 eyes ≥4 visits, mean 16.6, max 27. Confirmed 16 OCT biomarkers with ~0% missing. Explored authors' GitHub code — pre-computed `.npy` sequences already exist per eye. Also downloaded HuggingFace mirror (28.86 GB, usable for encoder pretraining).
Outcome: **Latent ODE viable. Model class decided.** → docs/archive/now/2026-06_olives-feasibility-audit.md
Decisions logged: DECISIONS.md #3 (model class), #4 (HuggingFace use), #5 (dataloader from authors' code)
