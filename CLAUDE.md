# Synapse — CLAUDE.md

> **CURRENT TASK:** OLIVES feasibility audit — Q1, Month 1 (see `docs/q1_sprint.md` when created)
> **Honest odds:** 80% working prototype by Month 6 · 5-10% chance of $100M+ company · 2-5% unicorn

## What This Repo Is

This is the codebase for **Synapse**, a computational disease dynamics company founded by Liban Britt. The goal is a "physics engine" for how disease evolves under treatment, starting with diabetic retinopathy (DR).

The source-of-truth strategy document is `Synapse_North_Star_3Year_Strategic_Roadmap.pdf` (v3.0). Read it before making architectural decisions.

## The Core Idea

Disease is a hidden dynamical system. Observations (retinal images, biomarkers, labs) are noisy projections of an underlying biological state. We learn the transition function:

```
state(t+dt) = f(state(t), treatment(t), patient_factors, noise)
```

Everything — encoders, fusion, simulation, synthetic generation — is infrastructure around learning `f`.

## Current Phase

**Year 1, Rung 1-2 of the capability ladder.**

- Rung 1: Strong retinal representation (public data)
- Rung 2: Temporal modeling on real sequences (OLIVES)

We do not claim Rung 3+ until data and evidence support it.

## Immediate Priority (Month 1)

1. **OLIVES feasibility audit** — literal first task. Answer 4 questions before any modeling:
   - File structure on disk
   - Temporal depth per eye (histogram of visit counts)
   - Alignment of imaging + 16 biomarkers + treatment timeline
   - Size of clean longitudinal subset
   - Decision gate: if most eyes have 4+ visits → latent ODE viable; if 2-3 → simpler temporal baselines

   **Claim boundary:** The honest Year 1 claim is *"open-data temporal prototype showing multimodal retinal disease states can be modeled over time, including treatment information."* Not "we solved DR progression." This applies to code comments, commit messages, and the preprint equally.

2. **UK Biobank application** — start Month 1, runs in parallel (async, takes months)

## Key Datasets

| Dataset | Access | Role |
|---|---|---|
| OLIVES | Free, CC BY 4.0 | Dynamics PoC — 96 eyes, longitudinal, treatment timing |
| EyePACS public | Free | Encoder pretraining |
| Messidor / -2 | Free | External validation |
| FGADR | Free | Lesion-aware auxiliary tasks |
| UK Biobank | ~£9K, grant-funded | Scale validation (Y2) |
| EyePACS private | Partnership | Scale validation (Y2) |

## Architecture (7 Tiers)

- **Tier 1** — Data ingestion (OLIVES + EyePACS loaders) — MVP
- **Tier 2** — Encoder: RETFound backbone; JEPA fine-tuning (Y1+) — MVP is RETFound
- **Tier 3** — HAWVA belief state: probabilistic hidden state — minimal in MVP
- **Tier 4** — Disease dynamics engine: latent ODE — MVP core
- **Tier 5** — Active inference (Y2-3)
- **Tier 6** — Simulation & generation (Y2-3)
- **Tier 7** — Products & APIs (Y2-3)

## What We Are NOT Building

Not a DR screening tool, not a synthetic-data vendor, not a clinical chatbot, not a diagnostic device. We model where disease is going, not what it is right now.

## Scientific Principles

- Prioritize falsifiability. The dynamics thesis might be wrong — that's acceptable.
- Decide model class BEFORE building it (the feasibility audit exists for this reason).
- External validation on held-out datasets before claiming any result.
- Every claim must be evidence-backed. Scope claims honestly.

## Code Standards

- Reproducible experiments: all runs logged to W&B, seeds set, env specs committed.
- Clean separation: data loaders / encoders / dynamics model / evaluation are separate modules.
- Baselines first: logistic regression, GRU-D, T-LSTM, Cox survival before latent ODE.
- No model code before the feasibility audit is complete.

## Funding Context

Building alongside other income — no dedicated runway. Near-zero burn through Month 6 (<$500). Do not introduce costs that require personal runway; time first cash needs to coincide with SBIR/grant funding.
