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
- `data/processed/embeddings/olives_retfound.npy` — 78,822 × 1024 OLIVES embeddings ✅
- `data/processed/embeddings/olives_eye_ids.npy` — eye IDs per image ✅
- `data/raw/olives/labels/` — visit timing, biomarkers, treatment labels ✅
- Authors' reference code: github.com/olivesgatech/OLIVES_Dataset (Time-Series Treatment Analysis)

## Tasks
1. Build OLIVES temporal dataloader — adapt authors' code, group embeddings by Eye_ID and visit order
2. Logistic regression baseline — predict DR/DME from single-visit embeddings
3. GRU-D baseline — temporal model on visit sequences
4. T-LSTM baseline — temporal model with irregular time gaps
5. Cox survival baseline — time-to-event modeling
6. Results table — log all baseline metrics to W&B

## Done When
- Results table exists with all 4 baselines reproducible
- All runs logged to W&B synapse-v1
- DECISIONS.md updated with which baselines were run and scores
- Clean git commit

## Hard Constraints
- No latent ODE code this sprint — baselines first
- All runs seeded and logged to W&B
- Use OLIVES embeddings as input, not raw images

## Next Up
Q2: Latent ODE prototype on OLIVES — the core dynamics model.
