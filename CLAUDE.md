# Synapse — CLAUDE.md

## What This Repo Is

This is the codebase for **Synapse**, a computational disease dynamics company founded by Liban Britt. The goal is a "physics engine" for how disease evolves under treatment, starting with diabetic retinopathy (DR).

## The Core Idea

Disease is a hidden dynamical system. Observations (retinal images, biomarkers, labs) are noisy projections of an underlying biological state. We learn the transition function:

```
state(t+dt) = f(state(t), treatment(t), patient_factors, noise)
```

Everything — encoders, fusion, simulation, synthetic generation — is infrastructure around learning `f`.

## Current Phase

See `docs/NOW.md → ## Orientation` for current phase, rung status, and live results.
The full Year 1 roadmap is in `docs/YEAR_1.md`. Strategic 3-year context is in `docs/NORTH_STAR_summary.md`.

## Architecture (7 Tiers)

- **Tier 1** — Data ingestion (`src/data/`) — MVP built
- **Tier 2** — Encoder: RETFound backbone (`src/encoders/`) — MVP built (frozen)
- **Tier 3** — HAWVA belief state (`src/belief/`) — minimal in MVP
- **Tier 4** — Disease dynamics engine: ODE-RNN (`src/dynamics/`) — MVP built
- **Tier 5** — Active inference — Year 2-3, not built
- **Tier 6** — Simulation & generation — Year 2-3, not built
- **Tier 7** — Products & APIs — Year 2-3, not built

Only Tiers 1-4 are in scope for Year 1. See `docs/ARCHITECTURE.md` for module boundaries and dependency rules.

## What We Are NOT Building

Not a DR screening tool, not a synthetic-data vendor, not a clinical chatbot, not a diagnostic device. We model where disease is going, not what it is right now.

## Scientific Principles

- Prioritize falsifiability. The dynamics thesis might be wrong — that is acceptable.
- External validation on held-out datasets before claiming any result.
- Every claim must be evidence-backed. Scope claims honestly.
- Lead with mechanism. Numbers support mechanism. Mechanism is what researchers remember.

## Code Standards

- Reproducible experiments: all runs logged to W&B, seeds set, env specs committed.
- Clean separation: data loaders / encoders / dynamics / evaluation are separate modules.
- Baselines first: logistic regression, GRU-D, T-LSTM, Cox survival before ODE.
- See `docs/CODE_STYLE.md` for naming conventions, docstrings, and definition of done.

## Funding Context

Building alongside other income — no dedicated runway. Near-zero burn through Month 6 (<$500). Do not introduce costs that require personal runway; time first cash needs to coincide with SBIR/grant funding.

---

## How to Work in This Repo

**Every session, in order:**

1. **Read `docs/NOW.md`** — the Orientation block gives you current phase, live claims, bound decisions, and which docs this sprint needs. This is the one required read.
2. **Consult the Document Tier Table below** for anything beyond the current sprint's scope.

**Session-start stale check:** If NOW.md Status has no entry dated within the last few days, or if recent work clearly happened that isn't logged — ask before starting: *"Last session logged was [date] — anything to capture before we start?"*

*If another AI model is being used: the human will paste the relevant docs manually. Same rules apply regardless of model.*

---

## Document Tier Table

### Tier 1 — Always (every session, no exceptions)
| Doc | What it gives you |
|---|---|
| `docs/NOW.md` | Current phase · live claims · bound decisions · docs needed this sprint |

### Tier 2 — Task-Triggered (read when the trigger applies, before starting the work)
| Doc | Read when... |
|---|---|
| `docs/ARCHITECTURE.md` | Writing or restructuring Python code · adding new `src/` modules |
| `docs/TECH_STACK.md` | Adding a dependency · choosing a tool or library |
| `docs/DATA.md` | Data pipeline work · writing Methods section · working with any dataset |
| `docs/CODE_STYLE.md` | Writing new Python code · reviewing existing code |
| `docs/GLOSSARY.md` | Using domain terms: CST, BCVA, TREX, Prime_FULL, HAWVA, ODE-RNN |

### Tier 3 — Reference (look up specific things; do not read start-to-finish)
| Doc | Use when... |
|---|---|
| `docs/CLAIMS.md` | Writing anything that could be a claim — commit messages, paper sentences, READMEs, grant text |
| `docs/DECISIONS.md` | Wondering if a question is already settled — scan the index table first |
| `docs/RUNS.md` | Referencing any experiment result |
| `docs/NORTH_STAR_summary.md` | Architectural decisions · scope questions · grant positioning |

### Tier 4 — Periodic (sprint closeout or quarterly review only)
| Doc | When |
|---|---|
| `docs/YEAR_1.md` | Quarterly review — gates and rungs |
| `docs/MILESTONES.md` | Quarterly review — tick outcomes |
| `docs/RISKS.md` | Quarterly review — retire or escalate |
| `docs/Q3_PLAN.md` | Sprint closeout — tick completed chunk checkbox |
| `docs/FUNDING.md` | Quarterly review — deadlines, actions |
| `docs/CONTACTS.md` | Quarterly review — new contacts, follow-ups |

### Tier 5 — Constitutional (major pivots and architectural decisions only)
| Doc | When |
|---|---|
| `docs/NORTH_STAR.pdf` | Fundamental scope changes · major strategic pivots |
| `docs/YEAR_1.md` | Go / no-go gate decisions |

---

## In-Session Update Rules

These happen **immediately** during the session — do not batch to closeout:

- **Decision made** → append to `docs/DECISIONS.md` + update the Quick Index at the top. Do this as soon as the decision is settled, not at the end.
- **W&B run completes** → append row to `docs/RUNS.md` immediately.
- **Open question resolves** → **delete** it from `docs/OPEN_QUESTIONS.md` immediately. The answer lives in `DECISIONS.md`.
- **New open question surfaces** → add it to `docs/OPEN_QUESTIONS.md` immediately.
- **Code changes module structure** → update `docs/ARCHITECTURE.md` before committing.
- **New dependency added** → update `docs/TECH_STACK.md` before committing.
- **New dataset added or schema changes** → update `docs/DATA.md` immediately.
- **New domain term used** → add it to `docs/GLOSSARY.md` immediately.
- **All sprint tasks checked off** → surface: *"All tasks are done — say 'close out this sprint' when ready."*

**Session-end rule (non-negotiable):** Before ending any session, prompt: *"Before we stop — should I update the Status block in NOW.md?"* Do not let a session end without the Status block being current.

---

## Sprint Closeout Ritual

When the human says **"close out this sprint"**, run the tiers in order. Stopping after Tier 1 is acceptable if time is short — the next session will still start correctly oriented.

### Tier 1 — Must Do (5 minutes)
1. Copy `docs/NOW.md` → `docs/archive/now/YYYY-MM_description.md`
2. Append one entry to `docs/LOG.md` (newest at top): date, what finished, outcome, link to archived NOW.
3. Copy `docs/templates/NOW_template.md` → fresh `docs/NOW.md`, filled with the next Q-plan chunk.
4. **Write the Orientation block first and accurately — this is the load-bearing step.** It is the only source of current state loaded at every session.
5. Set Status: *"Sprint just opened."*

*Stopping here is safe. The next session starts correctly oriented.*

### Tier 2 — Should Do (add 10 minutes)
6. Verify all sprint decisions are in `docs/DECISIONS.md` and the Quick Index is current.
7. Verify all W&B runs from this sprint are in `docs/RUNS.md`.
8. Clean `docs/OPEN_QUESTIONS.md` — delete answered items, add new ones surfaced this sprint.
9. Tick the completed chunk checkbox in `docs/Q3_PLAN.md`.
10. If the sprint changed what we can honestly claim → **propose** the `docs/CLAIMS.md` change and wait for human confirmation. Do NOT update CLAIMS.md without explicit human sign-off.

### Tier 3 — Full Closeout (add 10 minutes)
11. If sprint produced data findings → update `docs/DATA.md`.
12. If funding actions were taken → update `docs/FUNDING.md`.
13. If new contacts were made or are newly relevant → update `docs/CONTACTS.md`.
14. **Quarter check** — state out loud: which quarter we are in, which Q-plan chunks are done vs. remaining, and whether it is time to run the quarterly review. Trigger the review if all chunks are done or the quarter's month window has elapsed.

### Quarterly Review (trigger when quarter is complete)
- Tick milestones + record outcome in `docs/MILESTONES.md` (CONTINUE / ITERATE / PIVOT).
- Review `docs/RISKS.md` — retire resolved risks, escalate worsened ones, add new ones.
- Spot-check Tier 2 reference docs (`ARCHITECTURE`, `TECH_STACK`, `DATA`, `GLOSSARY`, `CODE_STYLE`) for staleness.
- Review `docs/FUNDING.md` — deadlines coming up? Anything to act on?
- Archive the quarter plan → `docs/archive/quarters/`, activate the next one; update the Q-plan reference in this section.

---

## The One Human-Owned Judgment

You (the AI) may move files and draft all entries. But CLAIMS.md honesty is the human's call. When closing a sprint, do NOT silently upgrade what we claim — propose the change and let the human confirm. We round DOWN, not up. The honest claim is always preferred over the impressive one.

---

## Scientific Discipline

We are testing Bet 1 only:
*"Continuous-time models better match irregular clinical observation structure than discrete recurrent models."*

We do not claim Bet 2 (treatment conditioning works) or Bet 3 (generalizes across diseases) until evidence supports them.

Every claim must have a matching result. If there is no result, there is no claim.

Never write "nobody has done this" or "we are the first." Always write "to our knowledge" and cite the closest prior work.

---

## Working With AI

- **Liban works in session bursts**, often with days or weeks between sessions. Do not ask orientation questions — read NOW.md Orientation and start from current state.
- **Scientific claim discipline is non-negotiable.** When uncertain about a claim, flag it and wait for human confirmation. Never write a claim that is not in CLAIMS.md CAN CLAIM.
- **Respond to session-end prompts.** When the AI asks "should I update the Status block?" answer yes or no. When it says "all tasks are done — ready to close out?" answer yes or no.
- **The three phrases:**
  - *"log that decision"* → append to DECISIONS.md now, update index
  - *"update progress, we're done"* → update Status section in NOW.md now
  - *"close out this sprint"* → run the Sprint Closeout Ritual above
- **CLAIMS.md is the veto.** Before any claim appears in writing — paper, README, commit message, grant — check CLAIMS.md. If it is not in CAN CLAIM, it cannot be written.
