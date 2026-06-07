> Parent: CLAUDE.md · This doc: present snapshot — what's happening RIGHT NOW.
> Changes: every session end. Overwritten, not appended. For permanent history see LOG.md.

# Progress — Present Snapshot

## Current Sprint
Real delta_t sprint (see NOW.md) — not started yet.

## In Flight
- Nothing started yet.

## Blocked
- Nothing. OCT-DR.xlsx is on disk; sprint is ready to start.

## Next Session Should
1. Read NOW.md for the real delta_t sprint spec.
2. Audit OCT-DR.xlsx — find the week/visit number columns, confirm Eye_ID join key.
3. Extend build_sequences() to return week_gaps per eye.
4. Re-run GRU-D, T-LSTM, Latent ODE with real delta_t on the same seed=42 split.
5. Compare ordinal vs real-delta RMSE table.

## Last Updated
2026-06-07 — Latent ODE sprint closed. RMSE 81.96 um (matches bar, Decision #10).
CLAIMS.md updated. Real delta_t sprint opened.
