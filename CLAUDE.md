# Synapse — CLAUDE.md

> **CURRENT TASK:** Reproducible Infrastructure — Q1, Weeks 2-3 (see `docs/NOW.md`)
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

---

## How to Work in This Repo (Read First, Every Session)

At the start of every session, orient yourself:
1. Read this file (CLAUDE.md) for mission, phase, and rules.
2. Read `docs/NOW.md` for the current task — this is what we work on.
3. Consult as needed: `docs/TECH_STACK.md` (how), `docs/ARCHITECTURE.md` (where code goes), `docs/DATA.md` (data shape), `docs/CLAIMS.md` (what we can honestly claim).

If GPT or another model is being used, the human will paste the relevant docs manually. Same rules apply regardless of model.

## The Document System

Guard rails (stable — the why and the limits):
- `docs/NORTH_STAR.pdf` — 3-year strategy, the constitution
- `docs/YEAR_1.md` — what we achieve this year; rungs + gates
- `docs/MILESTONES.md` — flat checklist of all milestones
- `docs/CLAIMS.md` — what we CAN and CANNOT currently claim (overclaim guard)

How we build (stable-ish):
- `docs/TECH_STACK.md`, `docs/CODE_STYLE.md`, `docs/ARCHITECTURE.md`, `docs/GLOSSARY.md`

The plan / the focus:
- `docs/Q1_PLAN.md` — current quarter at high level
- `docs/NOW.md` — THE current sprint, fully specified (this churns most)

Memory (continuous):
- `docs/PROGRESS.md` — present snapshot (in-flight, blocked)
- `docs/LOG.md` — permanent append-only history of completed sprints
- `docs/DECISIONS.md` — log of choices: X over Y because Z
- `docs/OPEN_QUESTIONS.md` — unresolved unknowns
- `docs/DATA.md` — dataset schemas + audit findings

Archive: retired docs live in `docs/archive/now/` and `docs/archive/quarters/`,
named `YYYY-MM_short-description.md`.

## Update Protocol — When Docs Get Touched

During a session:
- A real decision was made → append to `docs/DECISIONS.md` immediately.
- A new unknown surfaced, or one got answered → update `docs/OPEN_QUESTIONS.md`.
- Session ending → update `docs/PROGRESS.md` to current state.

The three phrases the human will use:
- "log that decision" → append a DECISIONS.md entry now.
- "update progress, we're done" → refresh PROGRESS.md.
- "close out this sprint" → run the Sprint Closeout Ritual below.

## Sprint Closeout Ritual

When the human says "close out this sprint," do these in order:
1. Copy `docs/NOW.md` → `docs/archive/now/YYYY-MM_description.md`
2. Append one entry to `docs/LOG.md` (newest at top): date, what finished, outcome, link to the archived NOW.
3. Confirm any decisions from this sprint are in `docs/DECISIONS.md`.
4. Reset `docs/PROGRESS.md` to the new present.
5. Copy `docs/templates/NOW_template.md` → fresh `docs/NOW.md`, filled with the next chunk from `docs/Q1_PLAN.md`. End it with a "Next up" line.
6. If the sprint changed what we can honestly claim → update `docs/CLAIMS.md`. (The human owns this judgment — flag it, let them confirm.)
7. If the sprint produced data findings → update `docs/DATA.md`.
8. Update the "Current Task" pointer at the top of this file (CLAUDE.md).

At a decision gate or quarter end (Month 3, 6, 12...):
- Tick the milestone + record gate outcome in `docs/MILESTONES.md`.
- Review `docs/RISKS.md` — retire, escalate, or add risks.
- Archive the quarter plan → `docs/archive/quarters/`, activate the next one.

## The One Human-Owned Judgment

You (the AI) may move files and draft all entries. But CLAIMS.md honesty is the
human's call. When closing a sprint, do NOT silently upgrade what we claim —
propose the change and let the human confirm. We round DOWN, not up. The honest
claim is always preferred over the impressive one.

## Current Task

> NOW: Reproducible Infrastructure — Q1, Weeks 2-3 (see `docs/NOW.md`)
