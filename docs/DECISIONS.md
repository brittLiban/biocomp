> Parent: CLAUDE.md · Related: docs/OPEN_QUESTIONS.md
> This doc: log of non-obvious choices and WHY. Stops re-litigating settled questions.
> Changes: append immediately when a real decision is made. Newest at top.

# Decisions

<!-- Format:
## #N — YYYY-MM-DD — <decision in one line>
Context: <what prompted it>
Choice: <what we decided>
Why: <reasoning>
Alternatives rejected: <what we didn't pick and why>
-->

## #1 — Month 0 — Adopt a layered docs architecture for AI-driven development
Context: building primarily with AI (Claude, sometimes GPT) across many sessions.
Choice: guard-rail / how-we-build / plan / focus / memory doc layers, with archive + ritual.
Why: keeps an AI agent on-mission across sessions; the docs are the shared memory between models.
Alternatives rejected: single big doc (AI ignores it) and no docs (AI drifts).

## #2 — Month 0 — OLIVES-first, feasibility audit before any model code
Context: public DR data is mostly cross-sectional; OLIVES is the only open longitudinal set.
Choice: audit OLIVES temporal structure before building any dynamics model.
Why: a latent ODE on temporally-thin data is wasted effort; decide model class on evidence.
Alternatives rejected: jumping straight to model-building (risks months on an unprovable claim).
