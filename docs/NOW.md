> Parent: docs/Q1_PLAN.md (Q2/Q3, Track 1) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — Preprint Draft Sprint

## Goal
Write the Synapse Year 1 preprint. Results section first — the 2×3 table (ordinal + real
delta_t, three models) is the paper's core claim. Then methods, introduction, and
discussion. Target: arXiv-ready draft by end of sprint.

## Why This Now
The results table is complete and the story has a clean arc:
1. Temporal structure in OLIVES contains learnable signal (GRU-D/T-LSTM beat persistence ~10%)
2. Continuous ODE dynamics match the best recurrent baselines on ordinal time (81.96 um)
3. Real irregular timing helps the ODE (-0.4 um) and hurts recurrents (+2-3 um) — directional
   evidence for the continuous-time structural advantage
4. Conclusion: architecture is validated at prototype scale; controlled data needed for
   statistical confirmation

This is a complete, honest, publishable arc. The preprint (a) makes the work citable, (b)
establishes priority, (c) provides concrete material for the EyePACS partnership ask, and
(d) gates an honest upgrade of CLAIMS.md to "Rung 2 demonstrated" post-submission.

## Inputs
All results are final. Key numbers:
- Ordinal: Persistence 91.7, GRU-D 82.2, T-LSTM 82.0, Latent ODE 81.96 um (seed=42)
- Real-ΔT: GRU-D 84.2, T-LSTM 85.0, Latent ODE 81.6 um (seed=42, grouped odeint)
- Representation: Logistic reg AUC 0.9906 (patient-level DME/healthy classification)
- Cox PH C-index 0.7955 (time to CST normalization)
- Dataset: OLIVES, 96 eyes, 77 train / 19 test (seed=42), two sub-studies (TREX DME, Prime_FULL)
- Model: ODE-RNN (Rubanova et al. 2019), 47,521 params, dopri5 solver, latent_dim=32
- W&B runs: latent_ode_v1_seed42, grud_realdelta_v2_seed42, ode_realdelta_v2_seed42 (project: synapse)
- DECISIONS.md #7–#11, CLAIMS.md, DATA.md — read before writing claims

## Tasks
1. **Results section** — write the full results section: representation quality, baseline
   comparison table (ordinal), real delta_t comparison table (with correct caveat on n=19),
   and the ODE vs recurrent timing differential. Include all RMSE/MAE numbers.
2. **Methods section** — dataset (OLIVES, two sub-studies, visit alignment), models
   (GRU-D, T-LSTM, ODE-RNN), training setup (seed=42, Adam, cosine/step LR, dopri5),
   timing encoding (ordinal vs real, Prime/TREX handling), evaluation (next-visit CST RMSE).
3. **Introduction** — motivate the dynamics framing (disease as hidden state, observations
   as noisy projections). Position vs screening tools. State the OLIVES experiment clearly.
4. **Discussion** — interpret the real-timing experiment; acknowledge the 19-eye limitation;
   frame "directional evidence, not confirmation"; say what controlled data would confirm.
5. **Abstract + title** — write last; should match the honest arc above.
6. **Parallel: EyePACS inquiry email** — draft a short, specific ask:
   "We have a working latent ODE pipeline with real-timing differential results. We need
   longitudinal data at scale to confirm this statistically." Attach or link the preprint
   draft. This is a Track 2 task that unblocks scale validation.

## Done When
- Full draft exists as `docs/preprint_draft.md` (or .tex if LaTeX preferred)
- Every claim in the draft is traceable to CLAIMS.md CAN CLAIM section
- Results section contains the 2×3 table with correct caveat language
- Discussion explicitly names the limitations (19 test eyes, no external validation,
  no treatment conditioning yet)
- EyePACS email draft exists
- PROGRESS.md updated; MILESTONES.md ticked

## Hard Constraints
- EVERY claim in the preprint must be in CLAIMS.md CAN CLAIM or DIRECTIONAL EVIDENCE.
  If writing forces an upgrade, flag it for human confirmation — never silently upgrade.
- Do NOT claim treatment effects are modeled — no experiments yet.
- Do NOT claim generalization — no external validation.
- "Directional evidence" language is mandatory in any sentence about the real-timing result.
- The 19-test-eye limitation must appear in the limitations section.

## Next Up
Encoder external validation on Messidor-2 — this would let us upgrade the representation
claim from "strong on OLIVES" to "generalizes out-of-distribution." That plus the preprint
submission completes the honest Gate 2 story for investors and collaborators.
