> Parent: CLAUDE.md · Related: docs/TECH_STACK.md, docs/ARCHITECTURE.md
> This doc: how code is written so the repo stays consistent across many AI sessions.
> Changes: occasionally, as conventions settle.

# Code Style

## Module Boundaries (mirror the architecture)
Code lives in `src/` mirroring the 7 tiers:
- `src/data/` — Tier 1: ingestion, loaders, temporal alignment
- `src/encoders/` — Tier 2: RETFound, JEPA, clinical, temporal encoders
- `src/belief/` — Tier 3: HAWVA / uncertainty
- `src/dynamics/` — Tier 4: latent ODE, transition models
- `src/eval/` — baselines, metrics, calibration
- `src/utils/` — shared helpers

Keep tiers separate. A dynamics model should not contain data-loading code.

## Naming
- Files & functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE`
- Descriptive over short: `encode_oct_sequence` not `enc`

## Docstrings
Every public function/class gets a docstring: one-line summary, args, returns, and
shape annotations for tensors (e.g., "x: [batch, time, 768]").

## Reproducibility (non-negotiable)
- Set seeds at the top of every experiment.
- Log every run to W&B with config.
- No hard-coded paths — use a config file or constants in `src/utils/`.
- Save configs alongside results so any run can be rerun.

## "Done When" — definition of done for a piece of code
1. It runs end-to-end without error.
2. It has a docstring + tensor shapes documented.
3. It has at least a sanity-check test in `tests/`.
4. Output is logged/saved reproducibly.
5. Any non-obvious choice is logged in DECISIONS.md.

## Baselines First
Always implement and beat simple baselines (logistic, GRU-D, T-LSTM, Cox) BEFORE
the fancy model. A latent ODE that can't beat GRU-D is not a result.

## No Model Code Before the Audit
Hard rule for Q1: no dynamics-model code until the OLIVES feasibility audit is complete
and the model class is decided. The audit determines whether a latent ODE is even viable.
