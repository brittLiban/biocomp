> Parent: docs/Q1_PLAN.md (Weeks 3-4 chunk) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — EyePACS + Messidor Pipeline + RETFound Encoding

## Goal
Build data loaders for EyePACS and Messidor, integrate RETFound, and encode images
into embeddings that are cached and reloadable. This is Rung 1 — strong retinal representation.

## Why This Now
The encoder is the foundation of everything above it. Baselines, dynamics models, and the
preprint all depend on having good retinal embeddings. RETFound is the state-of-the-art
pretrained retinal foundation model — we use it rather than train from scratch.

## Inputs
- EyePACS public dataset (Kaggle — needs download)
- Messidor / Messidor-2 (ADCIS — needs download)
- RETFound weights (HuggingFace: openmedlab/RETFound_cfp)
- HuggingFace OLIVES data already on disk (usable here — images without temporal structure are fine for encoder work)
- `requirements.txt` and `src/utils/seeds.py` from infra sprint

## Tasks
1. Download EyePACS public (Kaggle CLI) → `data/raw/eyepacs/`
2. Download Messidor-2 → `data/raw/messidor/`
3. Build `src/data/eyepacs.py` — PyTorch Dataset class for EyePACS
4. Build `src/data/messidor.py` — PyTorch Dataset class for Messidor
5. Download RETFound weights from HuggingFace
6. Build `src/encoders/retfound.py` — wrapper that loads RETFound and runs forward pass
7. Run encoding: iterate dataset, encode images, cache embeddings as `.npy` files
8. Validate: load cached embeddings, confirm shapes and that they're non-trivial (not all zeros)

## Done When
- Embeddings cached under `data/processed/embeddings/` for EyePACS + Messidor
- Shapes validated and logged to W&B
- `src/data/` and `src/encoders/` modules importable and tested
- Clean git commit

## Hard Constraints
- No fine-tuning RETFound this sprint — frozen weights only
- No dynamics model code
- Cache embeddings to disk — do not recompute every run

## Next Up
Weeks 4-5: Baselines — logistic regression, GRU-D, T-LSTM, Cox survival on OLIVES.
Needs OLIVES temporal dataloader (adapt authors' code from github.com/olivesgatech/OLIVES_Dataset).
