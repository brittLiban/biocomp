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

### Model 1 — ODE-RNN (BUILT · baseline)
**File:** `src/dynamics/latent_ode.py` · **Ref:** Rubanova et al. 2019 (NeurIPS)
**Status:** Trained on OLIVES, published. DOI: 10.21203/rs.3.rs-10060892/v1

Architecture (discriminative — point prediction, no uncertainty):
```
Input:  (batch, seq_len, emb_dim=1024)  +  lengths  +  delta_t_seq
        ↓  for each visit t:
encoder:    Linear(1024 → latent_dim=32)
gru_update: GRUCell(32 → 32)           ← assimilates new observation into state
ode_func:   MLP  32 → 64 → 64 → 32  (Tanh activations, autonomous)
            integrated via dopri5  [0, δt],  rtol=1e-3, atol=1e-4
dropout:    p=0.2  (before decoding)
decoder:    Linear(32 → 1)
Output: (batch, seq_len, 1) — predicted normalised CST at next visit
```
Training objective: MSE on next-visit CST (masked by sequence lengths)
Best result: 81.63 μm ts-weighted RMSE on OLIVES test split (n=19 eyes)

---

### Model 2 — True Latent ODE (NOT BUILT · primary target)
**File:** `src/dynamics/true_latent_ode.py` (to create) · **Ref:** Chen et al. 2018 (NeurIPS)
**Status:** Not started. This is the core business architecture.

Architecture (generative/VAE — outputs calibrated uncertainty):
```
Input:  (batch, seq_len, emb_dim=1024)  +  observation times
        ↓
Step 1 — Backward RNN encoder:
    GRU runs BACKWARD over the observed sequence
    Final hidden state → context vector c  (shape: latent_dim)

Step 2 — Variational bottleneck:
    μ(z₀)  = Linear(c → latent_dim)
    σ(z₀)  = Softplus(Linear(c → latent_dim))   ← always positive
    z₀    ~ N(μ, σ²)   (reparameterization trick)

Step 3 — ODE forward pass:
    z(t)   = odeint(ode_func, z₀, t_grid)
    ode_func: MLP  latent_dim → ode_hidden → ode_hidden → latent_dim  (Tanh)
    solver: dopri5

Step 4 — Decode:
    μ_pred(t) = Linear(z(t) → 1)
    σ_pred(t) = Softplus(Linear(z(t) → 1))   ← predictive uncertainty

Output: μ_pred, σ_pred per time point — a distribution, not a point
```
Training objective: ELBO = reconstruction NLL + β·KL(q(z₀) ‖ N(0,I))
- KL warmup: β anneals 0 → 1 over first N epochs to avoid posterior collapse
- Monitor: if KL < 0.01 nats → posterior collapse → stop, report as failure

Pre-committed evaluation bars (Decision #23, human confirmed):
- 90% CI coverage ≥ 80%  (calibration)
- KL > 0.1 nats  (encoder is using the latent space)
- RMSE ≤ 85 μm  (matches or beats ODE-RNN baseline)

---

### Y2+ — Treatment-Conditioned Dynamics (not built)
Treatment vector injected into ode_func: dz/dt = f(z, treatment, patient_factors)
Requires DRCR Protocol T data. Do not build until Bet 2 data is in hand.

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
