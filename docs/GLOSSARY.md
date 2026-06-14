> Parent: CLAUDE.md
> This doc: definitions of loaded project terms so AI and humans read shorthand correctly.
> Changes: append when a new term enters the project vocabulary.

# Glossary

- **Synapse** — the company; a computational disease dynamics platform.
- **Disease dynamics** — modeling how disease state evolves over time and under treatment.
- **Latent / hidden state** — the true underlying disease state; not directly observed.
- **Observation** — a noisy projection of hidden state (retinal image, biomarker, lab).
- **The core equation** — state(t+dt) = f(state(t), treatment(t), patient_factors, noise).
- **HAWVA** — our fusion layer: maintains probabilistic belief over hidden disease state.
  MVP = uncertainty only; Y3 = full Bayesian fusion brain.
- **JEPA** — Joint-Embedding Predictive Architecture; self-supervised method that predicts
  latent representations of future inputs. An optional Tier-2 enhancement, NOT the brand.
- **RETFound** — pretrained retinal foundation model (1.6M images). Our encoder backbone.
- **ODE-RNN** — what was actually built (Rubanova et al. 2019, discriminative variant). Encodes each visit, updates a GRU hidden state, then integrates that state forward in continuous time using a neural ODE before predicting next-visit CST. Deterministic — no posterior, no ELBO, no trajectory sampling. Tests Bet 1 (continuous-time vs discrete recurrent). This is the Rung 2 prototype.
- **Latent ODE** — the generative architecture (Chen et al. 2018). Runs an RNN encoder backwards over the full sequence to infer a posterior z_0, then integrates forward in latent space and reconstructs observations via a decoder. Enables trajectory sampling, counterfactual simulation, and synthetic cohort generation. This is the Year 2 / Rung 3 target. **Not what was built for the preprint.**
- **The capability ladder** — 8 rungs from "strong representation" to "disease infrastructure."
  We are on Rungs 1-2.
- **The feasibility audit** — the OLIVES data-structure check done BEFORE any modeling.
- **OLIVES** — open dataset (96 eyes, fundus+OCT, biomarkers, treatment timing). First
  Synapse-shaped data.
- **DR** — diabetic retinopathy. **DME** — diabetic macular edema. **AMD** — age-related
  macular degeneration. **ETDRS** — standard 0-4 DR severity scale.
- **The two tracks** — Track 1: build on open data now. Track 2: secure controlled data.
- **The claim boundary** — the line between what we can and cannot honestly claim. Lives in CLAIMS.md.
- **Rung** — a single concrete capability on the ladder. We climb one at a time.
- **The Three Bets** — the three sequential scientific claims Synapse is built on.
  Only Bet 1 is currently being tested. See CLAIMS.md.
- **Mechanism vs Benchmark** — a benchmark result reports a number (RMSE 81.6 um).
  A mechanistic result explains WHY (ODE improves with real timing because it integrates
  over biological intervals rather than using learned discrete-step decay that degrades
  under distributional shift). We always lead with mechanism. Numbers support the
  mechanism — they are not the point.
- **Falsifiable Hypothesis (current)** — "The ODE-RNN is structurally better suited for
  irregular clinical time series than discrete recurrent models, because ODE integration
  over real biological intervals does not require a learned decay parameter that can
  miscalibrate under distributional shift."
  This is what the timing experiment tests. The timing experiment is inconclusive at n=19
  (Decision #15). The ODE-RNN's near-zero timing sensitivity is consistent with the
  hypothesis but does not confirm it. Nothing more is claimed.
