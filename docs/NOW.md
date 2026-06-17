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

**Last session:** 2026-06-15 — Preprint submitted to medRxiv (MEDRXIV/2026/355647). Paper finalized: draft markers removed, [CITE] resolved (Yau et al. 2012), Figure 1 reference removed, code availability upon request added. PDF generated via Pandoc, committed and pushed (ae12e6d). Docs updated: Q3_PLAN Chunk 2 marked done, DECISIONS #16 logged (medRxiv over arXiv), NOW.md next steps written. Strategic direction confirmed: UK Biobank application → incorporate + SAM.gov → SBIR → Latent ODE.

**Current session (2026-06-16):** medRxiv issued final rejection — affiliation policy, no further appeal. Pivoted to Research Square (Springer Nature); submitted 2026-06-16, prescreening in progress. Decisions #17 and #18 logged. Discussed LLC formation (Washington State, $200 via sos.wa.gov). Discussed build timeline (14-30 months to Tier 6 product) and strategic direction confirmed: disease engine company, SBIR as near-term financial bridge. Docs committed and pushed (3a17edf).

**In flight:** Research Square prescreening — DOI expected within 24-48 hrs.

**Next (in order):**
1. Research Square DOI — passive wait, 24-48 hrs
2. Incorporate Synapse as Washington State LLC ($200, sos.wa.gov) — urgent, blocks SBIR + SAM.gov
3. UK Biobank application — start now (free to apply); £9K fee requires SBIR funding
5. SAM.gov UEI registration (2-4 weeks, free, requires LLC first)
6. NIH SBIR Phase I application (NEI) — December 5, 2026 target
7. Share preprint with 2-3 domain experts for informal feedback
8. Return to building: true Latent ODE — scale data source TBD (AI-READI or UK Biobank)
