# Synapse

A computational disease dynamics company. We model disease as a hidden dynamical system
observed through noisy signals and modified by intervention — starting with diabetic
retinopathy progression.

## This Repo Uses a Context Architecture for AI-Driven Development

The `docs/` folder is a structured system that steers AI coding agents (Claude, GPT)
across many sessions. It is the project's shared memory and guard rails.

### How to read it
- Start with `CLAUDE.md` (root) — mission, rules, and the work protocol.
- `docs/NORTH_STAR.pdf` — the full 3-year strategy (the constitution).
- `docs/YEAR_1.md` / `docs/MILESTONES.md` — the year and the milestone checklist.
- `docs/Q1_PLAN.md` — the current quarter.
- `docs/NOW.md` — the single thing being worked on right now.

### How to work in it
1. Open a session with Claude Code (auto-reads CLAUDE.md) — or paste CLAUDE.md + NOW.md into GPT.
2. Work the task in `docs/NOW.md`.
3. Use the three phrases: "log that decision" · "update progress, we're done" · "close out this sprint."
4. The AI maintains the docs as you go.

### Doc map
| Layer | Docs |
|---|---|
| Guard rails | NORTH_STAR.pdf, YEAR_1.md, MILESTONES.md, CLAIMS.md |
| How we build | TECH_STACK.md, CODE_STYLE.md, ARCHITECTURE.md, GLOSSARY.md |
| Plan / focus | Q1_PLAN.md, NOW.md |
| Memory | PROGRESS.md, LOG.md, DECISIONS.md, OPEN_QUESTIONS.md, DATA.md |
| Templates | docs/templates/ |
| Archive | docs/archive/now/, docs/archive/quarters/ |

### Code
`src/` mirrors the architecture tiers (data → encoders → belief → dynamics → eval).
No dynamics-model code until the OLIVES feasibility audit is complete (see NOW.md).
