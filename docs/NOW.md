> Parent: docs/Q1_PLAN.md (Weeks 4-5 chunk) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Baselines Sprint

## Goal
Run the first real models on OLIVES temporal sequences. Establish the results table
that the latent ODE must beat. Every future claim depends on having honest baselines.

## Why This Now
We have embeddings. We have the OLIVES data. The decision gate said latent ODE is viable.
Before building it, we need baselines — logistic regression, GRU-D, T-LSTM, Cox survival.
These are what we compare against. Without them, no result means anything.

## Inputs
- `data/processed/embeddings/olives_retfound.npy` — 78,822 × 1024 OLIVES embeddings (we encoded these)
- `data/processed/embeddings/olives_eye_ids.npy` — Eye_ID per embedding row
- `data/raw/olives/labels/OLIVES_Dataset_Labels/ml_centric_labels/Clinical_Data_Images.xlsx` — 78,185 rows with File_Path encoding visit order
- `data/raw/olives/labels/OLIVES_Dataset_Labels/full_labels/OCT-DR.xlsx` — treatment/injection timing
- Authors' reference code: github.com/olivesgatech/OLIVES_Dataset (Time-Series Treatment Analysis)

## Critical Design Decisions (locked before coding)

**What we predict:**
- Logistic regression: current disease state (DR vs DME) — floor baseline only
- GRU-D / T-LSTM: **next-visit disease state** given all prior visits — tests dynamics thesis
- Cox survival: time to first injection — event = first injection from OCT-DR.xlsx

**Train/test split: BY Eye_ID ONLY**
- 96 eyes total → ~77 train / 19 test (80/20 by eye, not by image)
- If any image from eye #47 appears in both train and test, results are leaked and meaningless

**Metrics (defined before running anything):**
| Model | Primary | Secondary |
|---|---|---|
| Logistic regression | AUC-ROC | Accuracy, per-class F1 |
| GRU-D | AUC-ROC (next-visit) | Accuracy |
| T-LSTM | AUC-ROC (next-visit) | Accuracy |
| Cox | C-index | — |

**ODE target:** Best baseline AUC → logged in DECISIONS.md as the minimum the latent ODE must beat.

## Tasks
1. **Align embeddings to visit order** — join our 78K embeddings (Eye_ID only) to Clinical_Data_Images.xlsx (has File_Path with visit number). Extract visit number from path. Sort each eye's embeddings chronologically. This is the make-or-break task.
2. **Build temporal dataloader** — group by Eye_ID, sort by visit, output sequences. Split train/test by Eye_ID. Adapt authors' Time-Series Treatment Analysis code.
3. **Logistic regression baseline** — single-visit embeddings → DR vs DME. Scikit-learn.
4. **GRU-D baseline** — sequence of visit embeddings → next-visit disease state.
5. **T-LSTM baseline** — same but handles irregular time gaps explicitly.
6. **Cox survival baseline** — time to first injection. Define event from OCT-DR.xlsx.
7. **Results table** — all metrics logged to W&B synapse-v1. Best result logged to DECISIONS.md as ODE target.

## Done When
- Train/test split is by Eye_ID ✓
- All 4 models trained and evaluated ✓
- Results table: Model | AUC | Accuracy | C-index | Notes ✓
- Best baseline AUC logged in DECISIONS.md as ODE minimum target ✓
- All W&B runs have configs, seeds, reproducible ✓
- Clean git commit ✓

## Hard Constraints
- No latent ODE code this sprint — baselines first
- All runs seeded and logged to W&B
- Use OLIVES embeddings as input, not raw images
- Split MUST be by Eye_ID — never random across images

## Next Up
Q2: Latent ODE prototype on OLIVES. Must beat the best baseline AUC logged this sprint.
