> Parent: CLAUDE.md · THE OVERCLAIM GUARD.
> This doc: what we CAN and CANNOT currently claim. Read before any commit message,
> README line, preprint sentence, or pitch.
> Changes: ONLY when evidence changes the boundary. The HUMAN owns this judgment. Round DOWN, not up.

# Claims — What We Can Honestly Say (as of: 2026-06-07, baselines complete)

## WE CAN CURRENTLY CLAIM
- We are building a computational disease dynamics prototype.
- We are testing whether DR progression can be modeled as a latent state-space system.
- OLIVES has the temporal structure to support dynamics modeling — 94/96 eyes with >=4 visits, mean 16.6 visits per eye (confirmed by audit, 2026-06-01).
- Temporal structure in OLIVES visit sequences contains learnable signal for CST prediction: GRU-D and T-LSTM beat naive persistence by ~10% RMSE (82 vs 92 um) on 19 held-out eyes.
- RETFound embeddings strongly encode disease type: logistic regression AUC 0.9906 on patient-level DME/healthy classification (static label — this is representation quality, not a dynamics result).

## WE CANNOT CLAIM (YET)
- That the latent ODE improves on temporal baselines — not built yet. ODE must beat RMSE 82.0 um (pre-committed bar, Decision #9).
- That time-decay mechanisms are fully utilized — delta_t is ordinal (1.0 steps), not real week gaps. Real gaps pending OCT-DR.xlsx parsing.
- That results generalize beyond 96 OLIVES eyes — no external validation.
- That treatment effects are modelable — no experiments yet.
- "Validated DR progression at scale" — requires controlled data (Year 2).
- Synthetic cohort validity — requires validated dynamics (Year 3).

## CLAIM BOUNDARY BY RUNG
- Rung 1 (representation): claimable once encoder beats baselines on public data + external val.
- Rung 2 (temporal): claimable as PROTOTYPE once OLIVES dynamics PoC works (with small-n caveat).
- Rung 3+ (scale, causal, synthetic): NOT claimable until controlled data + validation.

## THE RULE
Round DOWN, not up. The honest claim is always preferred over the impressive one.
When closing a sprint, the AI proposes any claim-boundary change; the HUMAN confirms.
