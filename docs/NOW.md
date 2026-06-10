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

**Last session:** 2026-06-10 — Four pre-submission peer-review corrections applied:
1. **Naming**: Renamed "Latent ODE" → "ODE-RNN" throughout (we implement the deterministic ODE-RNN variant, not variational Latent ODE with ELBO).
2. **Test-set leakage fix**: Added 60/17/19 train/val/test split with val-checkpoint selection. Re-ran all 6 models × 2 timing conditions. New ts-wt numbers: ODE 81.63/81.69 μm, T-LSTM 83.35/84.68 μm, GRU-D 88.21/85.15 μm. Eye-wt: ODE 70.89/71.90 μm, T-LSTM 73.07/73.63 μm, GRU-D 76.70/74.23 μm. Decision #14 logged. All 6 val-split runs in W&B.
3. **Input fairness**: Section 3.2 now explicitly frames the asymmetric input regime (temporal models: embeddings+BCVA; persistence: current CST).
4. **Modality clarification**: Section 2.2 documents the RETFound_cfp-on-OCT mismatch; Section 4.2 adds encoder scope limitation.
Paper updated: Abstract, Contributions, Sections 2.2, 2.3, 2.4, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 5. CLAIMS.md downgraded timing directional evidence (old story doesn't hold; results inconclusive). Table 1 now has full eye-wt column.

**In flight:** CLAIMS.md change flagged for human review — the DIRECTIONAL EVIDENCE section was downgraded (timing experiment inconclusive; GRU-D went opposite direction).

**Next:** Human confirm CLAIMS.md directional evidence downgrade → git commit → confirm repository URL → arXiv submission.
