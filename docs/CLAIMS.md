> Parent: CLAUDE.md · THE OVERCLAIM GUARD.
> This doc: what we CAN and CANNOT currently claim. Read before any commit message,
> README line, preprint sentence, or pitch.
> Changes: ONLY when evidence changes the boundary. The HUMAN owns this judgment. Round DOWN, not up.

# Claims — What We Can Honestly Say (as of: Month 0, pre-build)

## WE CAN CURRENTLY CLAIM
- We are building a computational disease dynamics prototype.
- We are testing whether DR progression can be modeled as a latent state-space system.
- OLIVES has the temporal structure to support dynamics modeling — 94/96 eyes with ≥4 visits, mean 16.6 visits per eye (confirmed by audit, 2026-06-01).
- (No modeling results yet — no experiments run.)

## WE CANNOT CLAIM (YET)
- That latent dynamics models work for DR progression — UNPROVEN until OLIVES PoC.
- That we beat baselines — no baselines run yet.
- That treatment effects are modelable — no experiments yet.
- That anything generalizes — no external validation yet.
- "Validated DR progression at scale" — requires controlled data (Year 2).
- Synthetic cohort validity — requires validated dynamics (Year 3).

## CLAIM BOUNDARY BY RUNG
- Rung 1 (representation): claimable once encoder beats baselines on public data + external val.
- Rung 2 (temporal): claimable as PROTOTYPE once OLIVES dynamics PoC works (with small-n caveat).
- Rung 3+ (scale, causal, synthetic): NOT claimable until controlled data + validation.

## THE RULE
Round DOWN, not up. The honest claim is always preferred over the impressive one.
When closing a sprint, the AI proposes any claim-boundary change; the HUMAN confirms.
