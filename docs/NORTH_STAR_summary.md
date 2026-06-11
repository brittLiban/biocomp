> Source: Synapse_North_Star_3Year_Strategic_Roadmap.pdf (v3.0) + docs/YEAR_1.md
> This doc: scannable markdown summary for in-session use. The PDF is authoritative for full detail.
> Changes: when strategic direction materially changes (major pivot or quarterly review).
> **Last updated: 2026-06-11**

# North Star — 3-Year Strategic Summary

## The Company

**Synapse** is a computational disease dynamics company. The product is a "physics engine" for how disease evolves under treatment — starting with diabetic retinopathy, expanding to other chronic diseases.

**The core bet:** disease progression is a learnable dynamical system. Observations (images, biomarkers, labs) are noisy projections of a hidden state. If we learn the transition function `f(state, treatment, patient_factors)`, we can simulate trajectories, optimise treatment timing, and generate synthetic cohorts for drug trials.

## The Three Bets (in order)

| Bet | Claim | Status |
|---|---|---|
| **Bet 1** | Continuous-time ODE models better match irregular clinical observation structure than discrete recurrent models | **ACTIVE — testing now. Prototype result in hand.** |
| **Bet 2** | Treatment-aware disease dynamics can be learned from observational longitudinal data | Year 2. Requires controlled data. Not claimed yet. |
| **Bet 3** | A reusable disease engine generalises across diseases and institutions | Year 3+. Not claimed. Not tested. |

Do not claim Bet 2 or Bet 3 until evidence supports them.

## The Capability Ladder

| Rung | What it proves | Status |
|---|---|---|
| Rung 1 | Strong retinal representation on public data | **COMPLETE** — AUC 0.9906 within-dist, AUC 0.77 OOD |
| Rung 2 | Temporal dynamics prototype on real sequences | **COMPLETE** — ODE-RNN 81.63 μm on OLIVES, 19 test eyes |
| Rung 3 | Scale validation on controlled longitudinal data | Year 2 — UK Biobank / EyePACS partnership needed |
| Rung 4 | Treatment conditioning | Year 2-3 — requires Rung 3 data |
| Rung 5+ | Generalisation, simulation, products | Year 3+ |

## The 3-Year Arc

**Year 1 (now) — Prove the architecture, solo:**
OLIVES feasibility audit → RETFound encoding → temporal baselines → ODE prototype → preprint → SBIR grant. Near-zero burn (<$500). All data free and public. Gate 3 at Month 12: have we earned the right to pursue seriously?

**Year 2 — Scale and validate, with data:**
UK Biobank or EyePACS private (need grant funding ~£9K+). Rung 3: does the architecture generalise beyond 96 eyes? Add treatment conditioning (Bet 2). Begin building relationships with retina clinicians and pharma.

**Year 3+ — Products and defensibility:**
Multi-disease platform. Synthetic cohort generation. REST API. Pharma partnerships for virtual trial arms. The physics engine as infrastructure. Exit paths: acqui-hire by EHR vendor, pharma partnership, or independent Series A.

## Revenue Model

Not a diagnostic device (avoids FDA SaMD pathway in early years). Revenue via:
- Pharma: synthetic patient cohorts for trial design
- Health systems: treatment-trajectory APIs
- Research: data access + compute partnerships

## Funding Path

- **Now–Month 6:** near-zero burn, building proof
- **Month 6–12:** NIH SBIR Phase I (~$300K), NSF SBIC, or Wellcome equivalent — fund UK Biobank access
- **Month 12–18:** seed round if Gate 3 passes and Rung 3 data is in hand
- **Year 2+:** Series A aligned with first pharma partnership or Rung 4 result

## Strategic Posture

**Build independently.** No single institution dependency. Multiple parallel data access routes (UK Biobank, EyePACS, BioLINCC ACCORD, local clinic). UW affiliation is a bonus accelerant, not a gate.

**Honest claims compound.** One credible prototype result published with caveats is worth more than three impressive-sounding overclaims. Every claim must map to evidence.

**The pivot trigger:** if Gate 2 or Gate 3 fails and the dynamics thesis cannot be demonstrated at prototype scale, reassess the core bet before spending further. The willingness to pivot or fold at a failed gate separates discipline from delusion.
