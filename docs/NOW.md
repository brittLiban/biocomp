> Parent: docs/Q3_PLAN.md (Chunk 1) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Preprint Draft Sprint

## Orientation
> Rewritten each sprint closeout. The single source of current state. Read this first.
> **Last rewritten: 2026-06-11**

**Phase:** Rung 1 complete · Rung 2 complete (prototype) · Preprint written (`paper/preprint_v1.md`) · arXiv submission pending. Rung 3 requires controlled data (Year 2).

**Active bet:** Bet 1 only — continuous-time ODE-RNN vs recurrent baselines on irregular clinical observation timing.

**Live results (all from val-split corrected runs, Decision #14):**
- AUC 0.9906 — OLIVES within-dist DME classification (static label; representation quality, not dynamics)
- AUC 0.77 OOD — Messidor-2 frozen linear probe (cross-dataset DR signal confirmed; not strong; Decision #12)
- ODE-RNN: 81.63 μm ts-wt / 70.89 μm eye-wt (ordinal, 60/17/19 split, 19 test eyes)
- T-LSTM: 83.35 μm ts-wt / 73.07 μm eye-wt · GRU-D: 88.21 μm ts-wt / 76.70 μm eye-wt
- Timing experiment: inconclusive at n=19 — GRU-D moved opposite direction (see DIRECTIONAL EVIDENCE in CLAIMS.md)

**Cannot claim:** strong OOD generalization (0.77 < 0.85 bar) · clear ODE advantage (margins within noise) · treatment effects · anything Rung 3+

**Decisions binding this sprint:** #14 (val-split; all prior numbers superseded) · #13 (report both metrics) · #8 (CST regression target)

**Docs needed this sprint:** CLAIMS.md (required before writing any sentence that could be a claim)

## Goal
Write the first full preprint draft. The results table is complete — representation,
temporal baselines, ODE, real delta-t, and external validation all exist. Turn that
into a coherent paper arguing Bet 1: continuous-time models better match irregular
clinical observation structure than discrete recurrent models.

## Why This Now
Every result needed for the paper is in hand. The arc is complete:
- Rung 1 (representation): AUC 0.9906 within-distribution + AUC 0.77 OOD (Messidor)
- Rung 2 (temporal): GRU-D/T-LSTM baselines + ODE-RNN (81.96 um, matches baselines)
- Real-dt: ODE improves (-0.4 um), recurrents degrade (+2-3 um) — directional support for Bet 1

A preprint now lets us get priority, gather feedback, and support grant applications (SBIR).

## Inputs
- `docs/CLAIMS.md` — what we can and cannot say (read first, every writing session)
- `docs/DECISIONS.md` — rationale for every result (especially #8-12)
- `docs/DATA.md` — dataset details for Methods section
- W&B runs: latent_ode_v1_seed42, grud_realdelta_seed42, tlstm_realdelta_seed42,
  ode_realdelta_v2_seed42, messidor_val_v1

## Tasks
1. **Outline** — title, abstract (~250 words), section skeleton. Agree with human before drafting.
2. **Introduction** — motivate irregular-time disease modeling; position vs. recurrent baselines.
   Cite: Rubanova 2019 (ODE-RNN), Che 2018 (GRU-D), Jarrett 2020 (T-LSTM), Krause 2018 (Messidor).
   Use "to our knowledge" framing; no "first ever" claims.
3. **Methods** — OLIVES dataset, RETFound encoder (frozen), temporal model architectures,
   evaluation protocol (CST RMSE, AUC), train/test split (77/19 eyes, seed=42).
4. **Results** — four subsections: (a) representation quality, (b) temporal baselines,
   (c) ODE vs. baselines, (d) real-delta-t analysis + Messidor OOD validation.
5. **Discussion** — mechanism first (why ODE benefits from real timing), then numbers.
   Honest caveats: 96 eyes, 19 test, small margin, threshold calibration on Messidor.
6. **Human review pass** — check every claim against CLAIMS.md before finalising.
7. **arXiv submission** — upload PDF + source.

## Done When
- Full draft exists as `paper/preprint_v1.pdf` (or equivalent)
- Every claim in the draft maps to an entry in CLAIMS.md CAN CLAIM section
- Human has read and approved the abstract and Discussion caveats
- arXiv submission ID recorded in DECISIONS.md

## Hard Constraints
- Every claim must have a matching result — no aspirational language
- Use "to our knowledge" not "first ever" or "nobody has done this"
- Lead with mechanism; numbers support mechanism
- CLAIMS.md is the veto — if it's in CANNOT CLAIM, it cannot go in the paper

## Next Up
arXiv submission → share with 2-3 domain experts for informal feedback before
submitting to a venue (MICCAI, MIDL, or Nature Comms Medicine).

---

## Status
> Updated each session. Replaces PROGRESS.md.

**Last session:** 2026-06-14 — CLAIMS.md directional evidence downgrade human-confirmed (Decision #15). ODE-RNN vs Latent ODE architectural distinction logged. Updates applied:
1. **CLAIMS.md** — DIRECTIONAL EVIDENCE section confirmed as inconclusive; robustness framing added; date updated to 2026-06-14.
2. **Paper Section 4.1 title** — updated from stale "Why Real Timing Helps the ODE and Hurts Recurrents" → "Continuous-Time Integration vs. Learned Decay — Structural Differences Under Irregular Timing".
3. **DECISIONS.md** — Decision #15 added (confirmation + ODE-RNN distinction).
4. **GLOSSARY.md** — ODE-RNN and Latent ODE defined separately; Falsifiable Hypothesis updated to name ODE-RNN explicitly.

**In flight:** Nothing. All corrections applied. Paper is ready for submission.

**Next:** git commit → confirm arXiv repository URL → arXiv submission.
