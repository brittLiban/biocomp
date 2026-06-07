> Parent: CLAUDE.md · This doc: permanent append-only history of completed sprints.
> Changes: one entry per completed sprint, NEWEST AT TOP. For present state see PROGRESS.md.

# Log — Completed Sprints

<!-- Newest entries at top. Format:
## YYYY-MM-DD — <sprint name>
<one or two lines: what finished>
Outcome: <result>. → docs/archive/now/YYYY-MM_description.md
Decision(s) logged: <DECISIONS.md ref if any>
-->

## 2026-06-07 — Real Delta-T Sprint
Re-ran GRU-D, T-LSTM, and Latent ODE with real week gaps from OLIVES visit keys (Prime: diff(visit_nums)/4; TREX: ordinal — real timing unavailable). Mid-sprint correction: ODE batch-mean delta_t (batch-composition-dependent, reproducibility defect) replaced with grouped per-example odeint. GRU-D and T-LSTM degrade with real timing (+2.0/+3.0 um); ODE improves (-0.4 um, 81.6 um). Directional evidence for ODE structural advantage with irregular time.
Outcome: **ODE 81.6 um with real delta_t — beats bar. Recurrent models degrade with real timing. CLAIMS.md updated with directional evidence section (human confirmed: not promoted to CAN CLAIM, 19 eyes too small).** → docs/archive/now/2026-06_realdelta.md
Decisions logged: DECISIONS.md #11 (corrected result, v1 batch-mean defect documented, v2 grouped odeint)

Results (corrected, real delta_t, seed=42):
| Model      | Ordinal RMSE | Real-ΔT RMSE | Δ          |
|------------|-------------|--------------|------------|
| GRU-D      | 82.2 um     | 84.2 um      | +2.0 worse |
| T-LSTM     | 82.0 um     | 85.0 um      | +3.0 worse |
| Latent ODE | 81.96 um    | 81.6 um      | -0.4 better|
| Persistence| 91.7 um     | 91.7 um      | 0.0        |

## 2026-06-07 — Latent ODE Sprint (Gate 2)
Built ODE-RNN (Rubanova et al. 2019) on OLIVES sequences: linear encoder (1024->32) + GRUCell observation update + 2-layer MLP ODEFunc + dopri5 solver + linear decoder. 47,521 parameters. Smoke-tested import and gradient flow. Trained 100 epochs (seed=42, batch=16, lr=1e-3). Best checkpoint at epoch 10 — rapid overfitting on 77 training eyes. Checkpoint saved to models/latent_ode_v1_seed42.pt (gitignored). Re-run confirmed deterministic.
Outcome: **RMSE 81.96 um — beats 82.0 um bar (Decision #9) by 0.04 um. Honest claim: ODE-RNN matches, not clearly beats, recurrent baselines. Gate 2 passed in prototype form. CLAIMS.md updated.** -> docs/archive/now/2026-06_latent-ode-sprint.md
Decisions logged: DECISIONS.md #10 (result + overfitting pattern + margin caveat)

Results (full table, ordinal delta_t=1.0):
| Model        | RMSE (um) | MAE (um) | Notes                              |
|--------------|-----------|----------|------------------------------------|
| Persistence  | 91.7      | --       | Naive last-value lower bound       |
| GRU-D        | 82.2      | ~60      | Baseline                           |
| T-LSTM       | 82.0      | ~60      | Best baseline (tied with ODE)      |
| Latent ODE   | 82.0*     | 58.3     | *81.96 um; 0.04 um over bar        |

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
