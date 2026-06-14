> Parent: CLAUDE.md · THE OVERCLAIM GUARD.
> This doc: what we CAN and CANNOT currently claim. Read before any commit message,
> README line, preprint sentence, or pitch.
> Changes: ONLY when evidence changes the boundary. The HUMAN owns this judgment. Round DOWN, not up.

# Claims — What We Can Honestly Say (as of: 2026-06-14, directional evidence downgrade human-confirmed)

## WE CAN CURRENTLY CLAIM (as of: 2026-06-07, Messidor external validation complete)
- We are building a computational disease dynamics prototype.
- We are testing whether DR progression can be modeled as a latent state-space system.
- OLIVES has the temporal structure to support dynamics modeling — 94/96 eyes with >=4 visits, mean 16.6 visits per eye (confirmed by audit, 2026-06-01).
- Temporal structure in OLIVES visit sequences contains learnable signal for CST prediction. IMPORTANT CAVEAT ON INPUT REGIME: temporal models (GRU-D, T-LSTM, ODE-RNN) receive RETFound OCT B-scan embeddings and BCVA as inputs; they do NOT receive current CST. Persistence predicts next CST using current CST directly. These are different input regimes — persistence has privileged access to the single strongest scalar predictor of next-visit CST. The honest framing: "image-based temporal modeling recovers complementary signal not available to persistence." Under eye-weighted RMSE, temporal models (73–75 μm) substantially outperform persistence (91.7 μm), confirming OCT embeddings carry learnable signal that persistence cannot use. Under timestep-weighted RMSE, temporal models (82–88 μm) score higher than persistence (55 μm) because a few stable long-follow-up eyes dominate. Both results are reported; the input-regime asymmetry must be disclosed in any comparison.
- RETFound embeddings strongly encode disease type: logistic regression AUC 0.9906 on patient-level DME/healthy classification (static label — this is representation quality, not a dynamics result).
- RETFound embeddings carry cross-dataset DR signal OOD: frozen linear probe trained on EyePACS (N=31,542) achieves AUC 0.77 on Messidor-2 (N=1,744, held-out, different camera/country/format). This confirms generalization of the representation beyond OLIVES. CAVEAT: 0.77 is not "strong" — the ">0.85 strong OOD" bar from NOW.md was NOT met. Honest framing: "cross-dataset DR signal confirmed, not strong generalization." Low sensitivity (0.23) reflects threshold calibration to EyePACS class distribution (19.5% referable vs. Messidor's 26.2%); AUC is threshold-independent. (Decision #12, 2026-06-07)
- A continuous-dynamics ODE-RNN achieves next-visit CST RMSE of 81.63 μm on 19 held-out OLIVES eyes under ordinal timing — outperforming T-LSTM (83.35 μm) and GRU-D (88.21 μm). This validates that continuous ODE dynamics modeling is feasible on this dataset at prototype scale. CAVEAT: all margins are on 19 eyes and are within sampling variance. The honest framing is "ODE outperforms recurrent baselines in this experiment," not "ODE definitively beats recurrent models." Numbers from val-split corrected runs (Decision #14, 2026-06-10); old (leaky) numbers were T-LSTM 82.0, GRU-D 82.2 μm — see Decision #14.

## WE CANNOT CLAIM (YET)
- That the ODE-RNN clearly outperforms temporal baselines — margins on 19 test eyes are within sampling noise. ODE-ordinal leads T-LSTM-ordinal by 1.72 μm and GRU-D-ordinal by 6.58 μm, but definitive separation requires more data. (Note: GRU-D's large gap likely reflects its 444K parameters overfitting the 60-eye training set, not a robust model quality difference.)
- That RETFound embeddings **strongly** generalize OOD — AUC 0.77 is cross-dataset signal, not strong performance. ">0.85 strong" bar NOT met.
- That treatment effects are modelable — no experiments yet.
- "Validated DR progression at scale" — requires controlled data (Year 2).
- Synthetic cohort validity — requires validated dynamics (Year 3).

## DIRECTIONAL EVIDENCE (not a confirmed claim)
> Human-confirmed downgrade: 2026-06-14. Decision #15.

- ~~Real irregular timing benefits the ODE and hurts recurrent baselines~~ — **SUPERSEDED and downgraded to inconclusive (Decisions #14, #15).** Val-split corrected results: ODE +0.06 μm (flat), T-LSTM +1.33 μm (marginal degradation), GRU-D −3.06 μm (improved — opposite direction to the old story). The timing experiment is **inconclusive at n=19**. Not promotable to CAN CLAIM.
- **What IS consistent with mechanism:** The ODE-RNN's near-zero timing response is structurally expected — unlike GRU-D/T-LSTM, it has no explicit decay parameter that can miscalibrate on real vs ordinal gaps. This robustness is worth noting in the paper but is not a confirmed claim of benefit. Distinguishing robustness from advantage requires larger n.

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
