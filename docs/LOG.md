> Parent: CLAUDE.md · This doc: permanent append-only history of completed sprints.
> Changes: one entry per completed sprint, NEWEST AT TOP. For present state see PROGRESS.md.

# Log — Completed Sprints

<!-- Newest entries at top. Format:
## YYYY-MM-DD — <sprint name>
<one or two lines: what finished>
Outcome: <result>. → docs/archive/now/YYYY-MM_description.md
Decision(s) logged: <DECISIONS.md ref if any>
-->

## 2026-06-01 — OLIVES Feasibility Audit
Downloaded OLIVES (31.6 GB Zenodo zip + 6.3 MB labels), extracted both sub-studies (TREX DME: 96,051 files; Prime_FULL: 66,813 files). Audited temporal depth: 94/96 eyes ≥4 visits, mean 16.6, max 27. Confirmed 16 OCT biomarkers with ~0% missing. Explored authors' GitHub code — pre-computed `.npy` sequences already exist per eye. Also downloaded HuggingFace mirror (28.86 GB, usable for encoder pretraining).
Outcome: **Latent ODE viable. Model class decided.** → docs/archive/now/2026-06_olives-feasibility-audit.md
Decisions logged: DECISIONS.md #3 (model class), #4 (HuggingFace use), #5 (dataloader from authors' code)
