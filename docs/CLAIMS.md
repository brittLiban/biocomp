> Parent: CLAUDE.md · THE OVERCLAIM GUARD.
> This doc: what we CAN and CANNOT currently claim. Read before any commit message,
> README line, preprint sentence, or pitch.
> Changes: ONLY when evidence changes the boundary. The HUMAN owns this judgment. Round DOWN, not up.

# Claims — What We Can Honestly Say (as of: 2026-06-07, real delta-t sprint complete)

## WE CAN CURRENTLY CLAIM
- We are building a computational disease dynamics prototype.
- We are testing whether DR progression can be modeled as a latent state-space system.
- OLIVES has the temporal structure to support dynamics modeling — 94/96 eyes with >=4 visits, mean 16.6 visits per eye (confirmed by audit, 2026-06-01).
- Temporal structure in OLIVES visit sequences contains learnable signal for CST prediction: GRU-D and T-LSTM beat naive persistence by ~10% RMSE (82 vs 92 um) on 19 held-out eyes.
- RETFound embeddings strongly encode disease type: logistic regression AUC 0.9906 on patient-level DME/healthy classification (static label — this is representation quality, not a dynamics result).
- A continuous-dynamics ODE-RNN achieves next-visit CST RMSE of ~82 um on 19 held-out OLIVES eyes — **matching** the best recurrent baselines (T-LSTM 82.0, GRU-D 82.2 um). This validates that continuous ODE dynamics modeling is feasible on this dataset at prototype scale. CAVEAT: the winning margin is 0.04 um on 19 eyes, which is within sampling variance. The honest framing is "ODE is comparable to strong temporal baselines," not "ODE clearly beats recurrent models." (Decision #10, 2026-06-07)

## WE CANNOT CLAIM (YET)
- That the latent ODE clearly outperforms temporal baselines — the 0.04 um margin on 19 test eyes is within sampling noise. Definitive separation requires more data.
- That real irregular timing gives the ODE a structural advantage on OLIVES — real delta-t was tested (Real Delta-T Sprint, 2026-06-07, Decision #11). All three models degraded: ODE +1.5 um, GRU-D +2.0 um, T-LSTM +3.0 um. ODE degraded least, consistent with its design, but the effect is not decisive on 19 test eyes. Hypothesis "ODE outperforms recurrents with real time" NOT confirmed on this dataset and implementation. Per-example ODE integration and more data are the next tests.
- That results generalize beyond 96 OLIVES eyes — no external validation.
- That treatment effects are modelable — no experiments yet.
- "Validated DR progression at scale" — requires controlled data (Year 2).
- Synthetic cohort validity — requires validated dynamics (Year 3).

## CLAIM BOUNDARY BY RUNG
- Rung 1 (representation): claimable once encoder beats baselines on public data + external val.
- Rung 2 (temporal): claimable as PROTOTYPE — ODE-RNN matches baselines, continuous dynamics validated with small-n caveat (96 eyes, 19 test, ordinal time). Gate 2 passed in prototype form.
- Rung 3+ (scale, causal, synthetic): NOT claimable until controlled data + validation.

## THE RULE
Round DOWN, not up. The honest claim is always preferred over the impressive one.
When closing a sprint, the AI proposes any claim-boundary change; the HUMAN confirms.
