> Parent: docs/Q1_PLAN.md (Weeks 2-3 chunk) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Reproducible Infrastructure

## Goal
Stand up the scaffolding that every future experiment depends on: environment spec,
experiment tracking, and seed utilities. A trivial logged experiment must run reproducibly
end-to-end before this sprint is done.

## Why This Now
Without reproducible infra, no experiment result can be trusted or re-run. This is the
foundation everything else builds on. Takes 1-2 sessions; unblocks all modeling work.

## Inputs
- Python 3.11 (already installed)
- W&B account (free tier — create at wandb.ai)
- OLIVES data already on disk (not needed this sprint)

## Tasks
1. Create `environment.yml` — pin all dependencies (Python version, torch, torchvision,
   numpy, pandas, matplotlib, wandb, torchdiffeq, huggingface-hub, openpyxl, pyarrow).
2. Create `src/utils/seeds.py` — seed utility (torch, numpy, random, cudnn deterministic).
3. Set up W&B: create account + project "synapse-v1", log in via CLI (`wandb login`).
4. Write `experiments/smoke_test.py` — trivial experiment (random data, one forward pass,
   logs loss + config to W&B). Must run end-to-end and appear in W&B dashboard.
5. Verify reproducibility: run smoke_test.py twice with same seed → identical loss values.
6. Commit everything (except data + wandb/ dir) with a clean git commit.

## Done When
- `environment.yml` exists and `conda env create -f environment.yml` works cleanly.
- `experiments/smoke_test.py` runs and logs to W&B without errors.
- Running twice with same seed produces identical output.
- Clean git commit on main.

## Hard Constraints
- No model code beyond the smoke test dummy forward pass.
- No data loading beyond what's needed for the smoke test (use random tensors).
- Keep dependencies minimal — only what Q1 actually needs.

## Next Up
Weeks 3-4: EyePACS + Messidor pipeline + RETFound encoding.
(HuggingFace OLIVES data is usable for this — no additional downloads needed.)
