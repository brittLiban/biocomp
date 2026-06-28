> Parent: docs/Q3_PLAN.md (Chunk 1) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Latent ODE + GRAPE Sprint

## Orientation
> Rewritten each sprint closeout. The single source of current state. Read this first.
> **Last rewritten: 2026-06-16**

**Phase:** Rung 1 complete · Rung 2 complete (prototype) · Preprint live on Research Square (Springer Nature) 2026-06-16, DOI: 10.21203/rs.3.rs-10060892/v1. ODE-RNN is the published baseline model. Next: true Latent ODE (generative/VAE) as the primary architecture. Rung 3 requires controlled data (DRCR, Year 2).

**Model architecture (critical distinction):**
- **ODE-RNN** (Rubanova 2019, discriminative) — BUILT, published in preprint. Point prediction only. This is now a baseline, not the primary model.
- **True Latent ODE** (Chen 2018, generative/VAE) — NEXT BUILD. Learns a distribution over hidden disease state. Outputs calibrated uncertainty. This is the core architecture going forward and what the business is built on.

**Bet status (precise — do not drift from this):**
- **Bet 1** (continuous-time > discrete-time for irregular observations): pilot done, inconclusive. ODE-RNN best RMSE at n=19 but margins within noise. NOT confirmed. True Latent ODE deepens Bet 1 (calibrated uncertainty) but does NOT confirm it. Only scale data (DRCR, UK Biobank) closes Bet 1.
- **Bet 2** (treatment-aware dynamics): NOT STARTED. No experiment in the current plan touches treatment as a model input. DRCR Protocol T/I data required before Bet 2 begins.
- **Bet 3** (cross-disease generalization): NOT TESTED. GRAPE gives an early, cheap, informal pilot signal only — years ahead of schedule. A positive GRAPE result = "directional pilot toward Bet 3." Cannot claim generalization at this scale.

**What this sprint advances:** Bet 1 depth (calibrated uncertainty via Latent ODE) + early Bet 3 pilot (GRAPE transfer). Does NOT touch Bet 2.

**Pre-committed bars (Decision #23, human confirmed):**
- Latent ODE success: 90% CI coverage ≥ 80%, KL > 0.1 nats, RMSE ≤ 85 μm
- Posterior collapse: KL < 0.01 nats → stop, report as failure
- GRAPE bars: set AFTER feasibility audit (different units, can't pre-commit yet)

**Live results (val-split corrected, Decision #14):**
- AUC 0.9906 — OLIVES within-dist DME classification (representation quality, not dynamics)
- AUC 0.77 OOD — Messidor-2 frozen linear probe (cross-dataset DR signal; not strong)
- ODE-RNN: 81.63 μm ts-wt / 70.89 μm eye-wt (ordinal) · 81.69 μm (real-dt)
- T-LSTM: 83.35 μm ts-wt · GRU-D: 88.21 μm ts-wt
- Timing experiment: inconclusive at n=19

**Cannot claim:** strong OOD generalization · clear ODE advantage · treatment effects · Bet 2 or Bet 3 confirmed · anything Rung 3+

**Data pipeline status:**
- OLIVES: fully processed (60/17/19 split, seed=42, sequences cached)
- GRAPE: not yet downloaded — download from Figshare, audit before any modeling
- DRCR Protocol T: form request not yet submitted — submit at public.jaeb.org
- AI-READI: confirmed cross-sectional, useful for representation only (Decision #20)
- UK Biobank: apply (free), hold £9K fee — lower priority than DRCR for treatment dynamics

**Admin status:**
- Synapse Computational LLC: NOT YET FILED — blocked by funds ($200 filing fee). File at sos.wa.gov when funds available. Gates SBIR + SAM.gov.
- SAM.gov UEI: not started, requires LLC first
- SBIR Phase I: target December 5, 2026. Cannot start until LLC + SAM.gov complete.

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
Build the true Latent ODE on OLIVES, transfer to GRAPE. Deepen Bet 1
(calibrated uncertainty) and produce an early Bet 3 pilot signal. These two
results together form the core of Paper 2 and the strongest possible SBIR
preliminary data at this stage.

## Why This Now
- Preprint (Paper 1) is submitted — ODE-RNN baseline is published
- ODE-RNN is now a comparison point, not the primary model
- True Latent ODE is the actual business architecture (generative, uncertainty-aware)
- GRAPE is free, longitudinal, downloadable today — second disease at zero cost
- DRCR Protocol T request can run in parallel — Bet 2 data incoming

## Build Sequence
1. **Download GRAPE** (Figshare, free) → run feasibility audit
   - Inspect: visit counts per eye, prediction target (VF MD or RNFL), interval distribution
   - Budget this like the OLIVES audit — it is a separate feasibility check, not a quick reuse
   - Outcome: set GRAPE prediction target + pre-commit GRAPE evaluation bars (separate DECISIONS entry)

2. **Run ODE-RNN + baselines on GRAPE** (~1-2 weeks)
   - Reuse existing model code, new dataloader + prediction target
   - Establishes baseline rows for Paper 2 Table 1
   - Bet 3 language: "directional pilot" only if ODE-RNN wins

3. **Build true Latent ODE on OLIVES** (~2-4 weeks)
   - Architecture: RNN encoder (backward pass) → q(z₀) distribution → ODE forward → decode
   - Train via ELBO (reconstruction + KL divergence with warmup schedule)
   - Evaluate against Decision #23 bars BEFORE interpreting results
   - Watch for posterior collapse (KL < 0.01 = failure, report honestly)

4. **Transfer Latent ODE to GRAPE** (~1-2 weeks after step 3)
   - Use GRAPE bars (set after audit) — do not interpret without pre-committed thresholds
   - If calibration holds: "directional pilot toward Bet 3" — not "generalization confirmed"

5. **Submit DRCR Protocol T request** (parallel, this week)
   - public.jaeb.org, free, form-based — Bet 2 data pipeline starts here

## Hard Constraints
- CLAIMS.md is the veto on every sentence
- Bet 3 language: never write "generalizes" — write "directional pilot toward Bet 3"
- Bet 2 is not touched by any experiment in this sprint — do not imply otherwise
- Posterior collapse must be reported if it occurs — do not suppress or reframe
- GRAPE bars must be pre-committed BEFORE training on GRAPE

## Done When
- True Latent ODE trained on OLIVES, evaluated against Decision #23 bars, result logged
- ODE-RNN baselines on GRAPE established
- True Latent ODE transferred to GRAPE, bars pre-committed and evaluated
- All results logged in RUNS.md
- Paper 2 outline drafted

---

## Status
> Updated each session. Replaces PROGRESS.md.

**Last session (2026-06-15):** Preprint submitted to medRxiv (MEDRXIV/2026/355647). Paper finalized and pushed (ae12e6d).

**Session (2026-06-16) — major session, all docs updated:**
- medRxiv final rejection (no org affiliation, appeal denied) → pivoted to Research Square; submitted, prescreening in progress (Decisions #17, #18)
- AI-READI investigated as free UK Biobank alternative → confirmed cross-sectional (1 visit/participant), ruled out for Bet 1 (Decisions #19, #20)
- Comprehensive longitudinal dataset catalog reviewed → GRAPE identified (free, Figshare, longitudinal glaucoma, download today); DRCR Protocol T/I prioritized for Bet 2 treatment data; UK Biobank downgraded (visits years apart, weaker for treatment dynamics) (Decision #21)
- Three bets precisely restated and locked: Bet 1 pilot done/inconclusive, Bet 2 not started, Bet 3 early pilot only via GRAPE (Decision #22)
- Pre-committed Latent ODE evaluation bars confirmed by human (Decision #23)
- ODE-RNN now baseline; True Latent ODE confirmed as primary architecture going forward
- LLC filing blocked — no $200 currently; will file at sos.wa.gov when funds available
- Sprint rewritten: NOW is Latent ODE + GRAPE sprint
- Full doc audit complete: ARCHITECTURE, Q3_PLAN, MILESTONES, RISKS all updated to reflect session decisions

**Research Square:** Submitted 2026-06-16. DOI confirmed: 10.21203/rs.3.rs-10060892/v1. Preprint live.

**Session (2026-06-27):**
- DOI confirmed and logged across NOW.md, Q3_PLAN.md, ARCHITECTURE.md
- ARCHITECTURE.md Tier 4 expanded: full implementation specs for ODE-RNN (baseline) and True Latent ODE (target) side by side
- True Latent ODE built and tested: src/dynamics/true_latent_ode.py, scripts/true_latent_ode.py, tests/test_true_latent_ode.py (7/7 passing)
- Architecture: backward GRU encoder → q(z0) → reparameterize → ODE forward → mu_pred + sigma_pred; ELBO loss with KL warmup; sample=False for deterministic eval RMSE
- All Decision #23 bars wired into training script (RMSE ≤ 85 um, CI coverage ≥ 80%, KL > 0.1 nats, collapse < 0.01 nats = stop)
- Ready to run: python scripts/true_latent_ode.py (OLIVES sequences already cached)

**Next (in order):**
1. **Run True Latent ODE on OLIVES** — `python scripts/true_latent_ode.py` — evaluate against Decision #23 bars, log result to RUNS.md
2. **Download GRAPE** (Figshare, free) — run feasibility audit before any modeling
3. **Submit DRCR Protocol T request** (public.jaeb.org, free) — starts Bet 2 data pipeline
4. Incorporate Synapse as Washington State LLC ($200, sos.wa.gov) — blocks SBIR + SAM.gov
5. Transfer True Latent ODE to GRAPE — after GRAPE audit and bars pre-committed
6. SAM.gov UEI registration (requires LLC first)
7. NIH SBIR Phase I (NEI) — December 5, 2026 target
