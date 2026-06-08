> Parent: docs/Q1_PLAN.md (Q2, Track 1) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Messidor External Validation Sprint

## Goal
Validate that RETFound embeddings generalize out-of-distribution: train a binary DR
classifier on EyePACS embeddings, evaluate on Messidor. Report AUC. This closes the
"no external validation" gap in CLAIMS.md before writing the preprint.

## Why This Now
Every result so far is within-distribution (OLIVES or EyePACS only). A reviewer will ask
about generalization. Messidor is the standard external benchmark — doing it now means the
preprint can make a stronger Rung 1 claim: not just "strong representation on OLIVES" but
"RETFound embeddings generalize to held-out DR dataset."

## Current Blockers (must resolve before coding)
- **Label file missing**: `messidor-2.csv` is a left/right pairing file — no DR grades.
  Need the ADCIS Messidor annotation Excel files (`Annotation_Base11.xls` through
  `Annotation_Base34.xls` for original Messidor; separate file for Messidor-2 PNGs).
  Download from the ADCIS Messidor page and put in `data/raw/messidor/annotations/`.
- **Partial encoding**: only 745 of 1748 images encoded, with no filename index saved.
  Need to re-encode all images with filename tracking so embeddings can be joined to labels.

## Inputs
- `data/raw/messidor/images/IMAGES/` — 1748 images (690 JPG Messidor-1, 1058 PNG Messidor-2)
- `data/raw/messidor/annotations/` — **NEEDS TO BE DOWNLOADED** from ADCIS
- `data/processed/embeddings/eyepacs_retfound.npy` + `eyepacs_labels.npy` — 31,542 × 1024, grades 0-4
- `src/data/messidor.py` — dataset class already built (expects labels_csv + image_dir)
- `src/encoders/retfound.py` — encoder already built

## Tasks
1. **Download labels** — get ADCIS Messidor annotation XLS files, put in
   `data/raw/messidor/annotations/`. Parse all batches into a single
   `data/processed/messidor_labels.csv` with columns: filename, dr_grade (0-3).
2. **Re-encode Messidor** — run full encoding with filename index saved alongside
   embeddings. Save: `messidor_retfound.npy` (N, 1024) + `messidor_filenames.npy` (N,).
3. **Join labels to embeddings** — match filenames in `messidor_filenames.npy` to
   `messidor_labels.csv`. Keep only matched rows.
4. **Train/eval** — logistic regression on EyePACS (binary: referable = grade ≥ 2).
   Evaluate on Messidor with same binary threshold. Report AUC-ROC, accuracy, sensitivity,
   specificity. Script: `scripts/validate_messidor.py`.
5. **Log result** — W&B run `messidor_val_v1`. Append to DECISIONS.md (#12).
6. **Update CLAIMS.md** — if AUC is strong (>0.85), the representation claim upgrades
   from "strong on OLIVES" to "generalizes OOD." Human confirms.

## Done When
- `messidor_labels.csv` parsed and saved
- All 1748 Messidor images re-encoded with filename tracking
- Logistic regression trained on EyePACS, evaluated on Messidor
- AUC + confusion matrix reported and logged to W&B
- Decision #12 appended to DECISIONS.md
- CLAIMS.md updated (human confirms if warranted)
- Clean git commit

## Hard Constraints
- No fine-tuning — frozen RETFound embeddings only
- Train on EyePACS, test on Messidor — no Messidor data in training
- CLAIMS.md change is human-confirmed only

## Next Up
Preprint Draft Sprint — once Messidor validation is done, the full results table is
complete (representation + temporal baselines + ODE + real delta_t + external validation)
and the paper has a complete arc.
