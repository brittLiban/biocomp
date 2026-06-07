> Parent: docs/Q1_PLAN.md (Q2, Track 1) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Real Delta-T Sprint

## Goal
Replace ordinal visit steps (delta_t = 1.0 everywhere) with real week gaps parsed from
OCT-DR.xlsx. Re-run GRU-D, T-LSTM, and the Latent ODE on the same split with real gaps.
Determine whether real timing changes the relative ranking — especially whether the ODE
gains further advantage over recurrent models (which it should, by design).

## Why This Now
Every result so far used delta_t = 1.0 (ordinal). The comparison is internally consistent
but scientifically impoverished: the models' time-awareness is not actually tested.
The latent ODE's core advantage — continuous time integration — is inert when all gaps
are identical. Real week gaps may widen or close the margin. We need to know before
writing the preprint or pitching the dynamics thesis to anyone.

This is also a scientific integrity issue: if the preprint shows "ODE matches baselines"
on ordinal time, a reviewer will immediately ask why we didn't use real timing. We need
that result in the table.

## Inputs
- `data/raw/olives/labels/OLIVES_Dataset_Labels/full_labels/OCT-DR.xlsx` — injection
  timing and visit week numbers. Parse VisitNum or Week column per Eye_ID.
- `src/data/olives.py` — extend `build_sequences()` to include `week_gaps` per eye.
- `scripts/baseline_grud.py`, `scripts/baseline_tlstm.py`, `scripts/latent_ode.py`
  — all need a real-delta_t mode (swap ordinal 1.0 for parsed week gaps).
- Ordinal baseline RMSEs: GRU-D 82.2, T-LSTM 82.0, Latent ODE 81.96 um (all seed=42).

## Tasks
1. **Audit OCT-DR.xlsx** — inspect columns. Find per-eye, per-visit week numbers.
   Confirm the join key to Eye_ID matches olives.py. Document findings in DATA.md.
2. **Extend build_sequences()** — add `week_gaps` field per eye: gaps between consecutive
   visits in weeks (length n_visits - 1). Cache invalidation: increment pkl version or
   set `force=True`. Do NOT break the existing ordinal path.
3. **Update baseline scripts** — add a `--real-delta-t` flag (or a constant at top)
   that swaps `delta_t = np.ones(n)` for `seq['week_gaps']`. Both modes must work;
   don't delete the ordinal path (it is the canonical comparison baseline).
4. **Re-run all three temporal models** with real delta_t, seed=42, same split.
   Log each to W&B: `grud_realdelta_seed42`, `tlstm_realdelta_seed42`, `ode_realdelta_seed42`.
5. **Compare results** — build a 2x3 table: ordinal vs real-delta for each model.
   Does the ODE gain more than the recurrent models? If yes, that is the dynamics thesis.
6. **Log to DECISIONS.md** — Decision #11: real delta_t outcome and what it means for claims.
7. **Update CLAIMS.md** — if real delta_t gives the ODE a larger, statistically distinguishable
   lead, upgrade the claim from "matches" to "outperforms." Human confirms.

## Done When
- OCT-DR.xlsx audited and week gap field documented in DATA.md
- `build_sequences()` returns `week_gaps` per eye
- All three temporal models re-run with real delta_t, results logged to W&B
- 2x3 comparison table written (ordinal vs real delta_t, RMSE + MAE)
- Decision #11 appended to DECISIONS.md
- CLAIMS.md updated if real delta_t result warrants it (human confirms)
- Clean git commit

## Hard Constraints
- Same train/test split: `split_by_eye(seqs, test_frac=0.2, seed=42)` — never change this.
- Same normalization: CST stats computed on training eyes only.
- Ordinal results remain the canonical comparison baseline — document both in the table,
  do not replace ordinal results, only add real-delta results alongside.
- No cherry-picking: first completed W&B run per model is the result.
- CLAIMS.md change is human-confirmed only.

## Next Up
Preprint draft — results section first. The table of models x metrics (ordinal + real delta_t)
is the core of the paper. Write that section once real delta_t results are in.
