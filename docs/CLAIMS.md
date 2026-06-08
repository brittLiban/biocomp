> Parent: CLAUDE.md · THE OVERCLAIM GUARD.
> This doc: what we CAN and CANNOT currently claim. Read before any commit message,
> README line, preprint sentence, or pitch.
> Changes: ONLY when evidence changes the boundary. The HUMAN owns this judgment. Round DOWN, not up.

# Claims — What We Can Honestly Say (as of: 2026-06-07, Messidor external validation complete)

## WE CAN CURRENTLY CLAIM (as of: 2026-06-07, Messidor external validation complete)
- We are building a computational disease dynamics prototype.
- We are testing whether DR progression can be modeled as a latent state-space system.
- OLIVES has the temporal structure to support dynamics modeling — 94/96 eyes with >=4 visits, mean 16.6 visits per eye (confirmed by audit, 2026-06-01).
- Temporal structure in OLIVES visit sequences contains learnable signal for CST prediction: GRU-D and T-LSTM beat naive persistence by ~10% RMSE (82 vs 92 um) on 19 held-out eyes.
- RETFound embeddings strongly encode disease type: logistic regression AUC 0.9906 on patient-level DME/healthy classification (static label — this is representation quality, not a dynamics result).
- RETFound embeddings carry cross-dataset DR signal OOD: frozen linear probe trained on EyePACS (N=31,542) achieves AUC 0.77 on Messidor-2 (N=1,744, held-out, different camera/country/format). This confirms generalization of the representation beyond OLIVES. CAVEAT: 0.77 is not "strong" — the ">0.85 strong OOD" bar from NOW.md was NOT met. Honest framing: "cross-dataset DR signal confirmed, not strong generalization." Low sensitivity (0.23) reflects threshold calibration to EyePACS class distribution (19.5% referable vs. Messidor's 26.2%); AUC is threshold-independent. (Decision #12, 2026-06-07)
- A continuous-dynamics ODE-RNN achieves next-visit CST RMSE of ~82 um on 19 held-out OLIVES eyes — **matching** the best recurrent baselines (T-LSTM 82.0, GRU-D 82.2 um). This validates that continuous ODE dynamics modeling is feasible on this dataset at prototype scale. CAVEAT: the winning margin is 0.04 um on 19 eyes, which is within sampling variance. The honest framing is "ODE is comparable to strong temporal baselines," not "ODE clearly beats recurrent models." (Decision #10, 2026-06-07)

## WE CANNOT CLAIM (YET)
- That the latent ODE clearly outperforms temporal baselines — the margin over T-LSTM ordinal is 0.4 um on 19 test eyes, within sampling noise. Definitive separation requires more data.
- That RETFound embeddings **strongly** generalize OOD — AUC 0.77 is cross-dataset signal, not strong performance. ">0.85 strong" bar NOT met.
- That treatment effects are modelable — no experiments yet.
- "Validated DR progression at scale" — requires controlled data (Year 2).
- Synthetic cohort validity — requires validated dynamics (Year 3).

## DIRECTIONAL EVIDENCE (not a confirmed claim)
- Real irregular timing benefits the ODE and hurts recurrent baselines (Real Delta-T Sprint v2, 2026-06-07, Decision #11). With per-example grouped odeint: ODE real-dt 81.6 um (−0.4 um from ordinal), GRU-D real-dt 84.2 um (+2.0 um), T-LSTM real-dt 85.0 um (+3.0 um). This is consistent with the dynamics thesis — the ODE integrates over exact biological intervals while recurrent decay parameters are miscalibrated outside their training range. The direction is right; the sample is too small (19 eyes) for a definitive claim. **Human must confirm before promoting to CAN CLAIM.**

## CLAIM BOUNDARY BY RUNG
- Rung 1 (representation): claimable once encoder beats baselines on public data + external val.
- Rung 2 (temporal): claimable as PROTOTYPE — ODE-RNN matches baselines, continuous dynamics validated with small-n caveat (96 eyes, 19 test, ordinal time). Gate 2 passed in prototype form.
- Rung 3+ (scale, causal, synthetic): NOT claimable until controlled data + validation.

## THE RULE
Round DOWN, not up. The honest claim is always preferred over the impressive one.
When closing a sprint, the AI proposes any claim-boundary change; the HUMAN confirms.

## THE THREE BETS (what we are actually testing)

Bet 1: Continuous-time models better match irregular
clinical observation structure than discrete recurrent
models.
→ CURRENT PREPRINT. This is what we are testing now.

Bet 2: Treatment-aware disease dynamics can be learned
from observational longitudinal data.
→ Year 2. Requires controlled data. Not claimed yet.

Bet 3: A reusable disease engine generalizes across
diseases and institutions.
→ Year 3+. Not claimed. Not tested.

We are on Bet 1. We do not claim Bet 2 or 3
until evidence supports them.

## PHRASES TO NEVER USE
- "Nobody has published exactly that"
- "We are the first to..."
- "No one has done this before"

## SAFE ALTERNATIVES
- "To our knowledge, no widely adopted reproducible
  demonstration exists of..."
- "We are not aware of prior work that..."
- "The closest prior work is [X], which differs
  because..."
