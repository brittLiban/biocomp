> Parent: CLAUDE.md · This doc: present snapshot — what's happening RIGHT NOW.
> Changes: every session end. Overwritten, not appended. For permanent history see LOG.md.

# Progress — Present Snapshot

## Current Sprint
EyePACS + Messidor Pipeline + RETFound Encoding (see NOW.md)

## In Flight
- [x] EyePACS downloaded + extracted (35,126 images)
- [x] Messidor-2 downloaded + extracted (1,748 images)
- [x] src/data/eyepacs.py — EyePACS Dataset class
- [x] src/data/messidor.py — Messidor Dataset class
- [x] src/encoders/retfound.py — RETFound wrapper (bitfound/RETFound_MAE, public, no HF login)
- [x] experiments/encode_datasets.py — encoding script
- [x] notebooks/retfound_encoding_colab.ipynb — Colab notebook for GPU encoding
- [~] RETFound encoding — running on Google Colab GPU (T4), ~15 min ETA
- [ ] Download embeddings from Google Drive → data/processed/embeddings/
- [ ] Validate embedding shapes + W&B log
- [ ] Commit + close sprint

## Blocked
- CPU encoding crashed after 26 hours (OOM, exit code 5). Switched to Google Colab GPU.
- Colab tab must stay open — disconnecting stops the job.

## Next Session Should
- Confirm embeddings landed in Google Drive
- Download to data/processed/embeddings/
- Validate + commit + close sprint

## Last Updated
2026-06-02 — CPU encoding crashed. Moved to Colab GPU. Encoding in progress.
