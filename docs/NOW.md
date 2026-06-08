> Parent: docs/Q1_PLAN.md (Q2, Track 1) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Preprint Draft Sprint

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

**Last session:** 2026-06-07 — Messidor validation complete; sprint closed; this sprint opened.
**In flight:** Nothing — sprint just opened.
**Blocked:** Nothing.
