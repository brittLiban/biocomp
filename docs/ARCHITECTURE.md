> Parent: docs/NORTH_STAR.pdf · Related: docs/TECH_STACK.md, docs/CODE_STYLE.md
> This doc: the 7-tier system and where code goes.
> Changes: when a tier is added or substantially restructured.

# Architecture — 7 Tiers

The whole company reduces to learning:
**state(t+dt) = f( state(t), treatment(t), patient_factors, noise )**
Everything below is infrastructure around learning f.

## Tier 1 — Data Ingestion  →  src/data/
Ingest asynchronous, sparse, irregular observations; normalize; align timelines.
MVP: OLIVES + EyePACS loaders, temporal alignment, per-eye sequence construction.

## Tier 2 — Encoder Layer  →  src/encoders/
Convert raw observations to latent embeddings.
- RETFound backbone (pretrained, frozen initially) — MVP
- JEPA-style predictive fine-tuning on top — Year 1+, OPTIONAL enhancement, not load-bearing
- Clinical encoder (OLIVES biomarkers), temporal encoder (visit spacing)

## Tier 3 — HAWVA Belief State  →  src/belief/
Persistent probabilistic belief over hidden disease state.
- MVP: uncertainty + calibration only (one modality)
- Y2: simple multimodal fusion
- Y3: full Bayesian fusion brain

## Tier 4 — Disease Dynamics Engine  →  src/dynamics/
The scientific core. Learns the transition function f.
- MVP: latent ODE on OLIVES sequences
- Y2: treatment-conditioned transitions, physician-policy separation

## Tier 5 — Active Inference  (Y2-3, not built yet)
Recommend optimal next observation to reduce uncertainty. Decision support WITHOUT
treatment recommendation (stays out of SaMD regulatory territory early).

## Tier 6 — Simulation & Generation  (Y2-3, not built yet)
Trajectory forecasting, counterfactual simulation, synthetic cohorts, virtual trials.

## Tier 7 — Products & APIs  (Y2-3, not built yet)
REST/GraphQL API, SDK, dashboards, pharma integration, audit logging.

## Dependency Rule
Lower tiers must not import higher tiers. data → encoders → belief → dynamics is the
allowed flow. eval/ may read from any tier. utils/ is imported by all.

## Current Build Scope
Only Tiers 1-4 (minimal) are in scope for Year 1. Tiers 5-7 are documented for
reference but NOT built until Year 2+.
