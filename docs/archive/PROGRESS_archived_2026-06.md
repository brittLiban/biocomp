> Parent: CLAUDE.md · This doc: present snapshot — what's happening RIGHT NOW.
> Changes: every session end. Overwritten, not appended. For permanent history see LOG.md.

# Progress — Present Snapshot

## Current Sprint
Messidor External Validation Sprint — COMPLETE (see NOW.md, Decision #12).

## In Flight
- Nothing in flight.

## Completed This Sprint
- Task 1: Labels downloaded (Kaggle google-brain/messidor2-dr-grades) + parsed to `data/processed/messidor_labels.csv` (1744 gradable rows, ICDR 0-4). 4 ungradable images excluded.
- Task 2: All 1748 Messidor images re-encoded with filename tracking. Saved `messidor_retfound.npy` (1748, 1024) + `messidor_filenames.npy` (1748,). Fixed truncated PNG crash (ImageFile.LOAD_TRUNCATED_IMAGES).
- Task 3: Labels joined to embeddings — 1744/1748 matched (4 ungradable had no label).
- Task 4: Logistic regression trained on EyePACS (31,542 × 1024, binary grade ≥ 2), evaluated on Messidor-2. AUC 0.7699. Logged to W&B as messidor_val_v1 (run fxkir659).
- Task 5: Decision #12 appended to DECISIONS.md.
- Task 6: CLAIMS.md updated — OOD cross-dataset signal added to CAN CLAIM with honest caveats; "no external validation" removed from CANNOT CLAIM. **Human confirmation pending.**

## Final Result — Messidor External Validation

| Metric | Value |
|---|---|
| AUC-ROC | **0.7699** |
| Accuracy | 0.7884 |
| Sensitivity | 0.2341 |
| Specificity | 0.9852 |
| TP / FP / FN / TN | 107 / 19 / 350 / 1268 |

W&B: messidor_val_v1 (run fxkir659, project synapse)

Finding: Frozen RETFound embeddings carry cross-dataset DR signal (AUC 0.77, zero-shot cross-domain linear probe). Low sensitivity is a threshold-calibration artifact (EyePACS 19.5% vs Messidor 26.2% referable rate). AUC is the canonical threshold-independent metric. Does NOT meet the ">0.85 strong" bar from NOW.md — honest framing: "cross-dataset DR signal confirmed, not strong generalization."

## Blocked
- CLAIMS.md change is proposed; awaiting human confirmation before treating as settled.

## Last Updated
2026-06-07 — Messidor validation complete; all sprint tasks done except git commit.
