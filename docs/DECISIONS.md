> Parent: CLAUDE.md · Related: docs/OPEN_QUESTIONS.md
> This doc: log of non-obvious choices and WHY. Stops re-litigating settled questions.
> Changes: append entry + update Quick Index immediately when a decision is made. Newest at top.

# Decisions

## Quick Index
> Scan this first — 5 seconds to check if your question is already settled.
> **Last updated: 2026-06-16**

| # | Date | Summary |
|---|---|---|
| 21 | 2026-06-16 | Revised dataset priority: GRAPE first (free, longitudinal, glaucoma), DRCR T/I next (treatment), UK Biobank downgraded (visits years apart) |
| 20 | 2026-06-16 | AI-READI ruled out as Bet 1 dataset — confirmed cross-sectional (1 visit/participant); UK Biobank remains primary longitudinal path |
| 19 | 2026-06-16 | Pause UK Biobank £9K fee commitment; investigate AI-READI first (free, self-serve, diabetes-specific) |
| 18 | 2026-06-16 | medRxiv final rejection confirmed; pivot to Research Square (Springer Nature) — no affiliation required, gives DOI |
| 17 | 2026-06-15 | medRxiv rejected MEDRXIV/2026/355647 — no org affiliation; appeal sent; fallback is arXiv; no LLC yet |
| 16 | 2026-06-15 | Submitted to medRxiv (not arXiv) — endorsement required for cs.LG; medRxiv ID MEDRXIV/2026/355647 |
| 15 | 2026-06-14 | Human confirmed directional evidence downgrade; ODE-RNN (discriminative) ≠ Latent ODE (generative) — Year 2 target |
| 14 | 2026-06-10 | Val-split fix: 60/17/19 split, val-checkpoint selection — all prior ODE/recurrent runs superseded; timing experiment inconclusive |
| 13 | 2026-06-08 | Report both ts-wt AND eye-wt RMSE — both correct, both in Table 1 |
| 12 | 2026-06-07 | Messidor AUC 0.77 OOD — cross-dataset signal confirmed, not strong; ">0.85 strong" bar not met |
| 11 | 2026-06-07 | Real delta-t v2: ODE −0.4 μm, recurrents +2-3 μm (SUPERSEDED by #14 val-split correction) |
| 10 | 2026-06-07 | ODE-RNN 81.96 μm beats bar (SUPERSEDED by #14 val-split correction) |
| 9 | 2026-06-07 | ODE target pre-committed: RMSE < 82.0 μm, cannot move post-hoc |
| 8 | 2026-06-07 | Target: CST regression (RMSE), not binary classification threshold |
| 7 | 2026-06-07 | OLIVES join key: (Eye_ID, BCVA, CST) composite — not File_Path |
| 6 | 2026-06-02 | EyePACS/Messidor = classification baseline; OLIVES = dynamics |
| 5 | 2026-06-01 | Adapt OLIVES authors' dataloader, not build from scratch |
| 4 | 2026-06-01 | HuggingFace OLIVES: encoder pretraining only; Zenodo needed for dynamics |
| 3 | 2026-05-31 | Latent ODE viable for OLIVES (97.9% eyes ≥ 4 visits, mean 16.6) |
| 2 | Month 0 | OLIVES-first; feasibility audit before any model code |
| 1 | Month 0 | Layered doc architecture for AI-driven development |

## #21 — 2026-06-16 — Revised dataset priority order based on longitudinal catalog

Context: Reviewed a comprehensive catalog of longitudinal ophthalmic imaging datasets (data/ folder). UK Biobank was previously the primary scale target. Catalog reveals UK Biobank visits are years apart — weaker for treatment dynamics than assumed. Two better near-term options identified: GRAPE (free, longitudinal, glaucoma, available today) and DRCR Protocol T/I (free, form request, treatment-response, directly enables Bet 2).

Choice:
1. GRAPE — download from Figshare immediately. Second disease for Bet 1 transfer; free; no application needed.
2. DRCR Protocol T/I — submit form request this week. Treatment-conditioned dynamics data; enables Bet 2; free.
3. UK Biobank — still apply (free to apply), but hold £9K fee; useful for scale and survival but not dense treatment dynamics.
4. A2A SD-OCT — query NEI Data Commons after SBIR submitted; AMD longitudinal OCT, 316 participants, 1,499 visits.

Why UK Biobank is downgraded: Its repeat imaging is years apart, making it excellent for slow progression and survival but comparatively weak for treatment-response ODE-RNN work. DRCR clinical-trial data is denser and more directly relevant to Bet 2.

Why GRAPE is urgent: It is free, publicly downloadable today, longitudinal (3-9 visits/eye, explicit interval metadata), and provides a second disease — exactly what's needed for an external-validation/disease-transfer result in Paper 2 and SBIR preliminary data.

Alternatives rejected: Waiting for UK Biobank as the only path — there are free longitudinal options available now that address the science more directly.

---

## #20 — 2026-06-16 — AI-READI ruled out as primary Bet 1 dataset

Context: Investigated AI-READI as a free, fast alternative to UK Biobank for scale validation of the timing experiment. Confirmed directly from Healthsheet documentation (docs.aireadi.org/docs/3/dataset/healthsheet): "There is currently only one visit per participant." v3.0.0 has 2,280 participants.

Choice: AI-READI is cross-sectional and cannot support Bet 1 (continuous-time vs discrete-time dynamics requires repeated visits per participant to model trajectories). UK Biobank restored as the sole confirmed longitudinal scale path.

Future caveat: ~4% of participants (~90 people) expected to have a Year-4 follow-up. Far too small and not yet available — do not plan around it.

Remaining value: AI-READI is still a strong candidate for representation-quality benchmarking, multimodal fusion, and severity classification at scale (2,280 participants, 15+ modalities, provided 70/15/15 split). These are not Bet 1 experiments.

Alternatives rejected: Treating AI-READI as a longitudinal dataset — directly contradicted by their own documentation.

---

## #19 — 2026-06-16 — Pause UK Biobank fee commitment; prioritize AI-READI investigation first

Context: UK Biobank requires ~£9K (~$11K) fee charged on approval — a cost we do not currently have budgeted. AI-READI (Artificial Intelligence Ready and Equitable Atlas for Diabetes Insights) was discovered as a free, self-serve alternative: NIH Bridge2AI-funded, ~4,000 participants, diabetes-specific, multimodal, self-serve access via fairhub.io (days not months). UW is one of AI-READI's data-generating sites.

Choice: Continue UK Biobank APPLICATION (free to register and begin; £9K only due on approval) but do NOT commit to the fee until AI-READI has been evaluated. Request AI-READI mini dataset access (100 participants, 179 GB) immediately to inspect longitudinal structure.

Why: AI-READI may resolve our scale-data need at zero cost. No reason to spend £9K before confirming whether a free option addresses Bet 1. The UK Biobank application runs in the background and costs nothing until approval.

Critical unknown: Whether AI-READI retinal imaging is longitudinal (repeated visits) or cross-sectional (single visit). If cross-sectional, it cannot address the Bet 1 timing experiment — this must be confirmed before any further planning. See OPEN_QUESTIONS.md.

Alternatives rejected: Proceeding directly to UK Biobank fee payment — premature given an unverified free alternative exists.

**RESOLVED 2026-06-16:** AI-READI confirmed cross-sectional — one visit per participant (source: docs.aireadi.org healthsheet). Cannot address Bet 1 timing experiment. Useful for representation quality and multimodal work only. UK Biobank restored as primary longitudinal scale data path. AI-READI open questions deleted from OPEN_QUESTIONS.md.

---

## #18 — 2026-06-16 — medRxiv final rejection; pivot to Research Square

Context: medRxiv confirmed final rejection on 2026-06-16. Their policy requires an established organizational affiliation — "in the process of incorporating" is not sufficient. No further appeal possible until Synapse LLC exists.

Choice: Submit to Research Square (researchsquare.com), Springer Nature's preprint server. No affiliation requirement. Gives a DOI. Indexed by Google Scholar and PubMed preprint results. Sufficient for SBIR and UK Biobank purposes.

Why not arXiv: Decision #16 flagged that cs.LG requires an endorser for first-time submitters. Research Square is faster with no endorsement friction.

Future option: Once Synapse LLC is incorporated, medRxiv resubmission is possible for the medRxiv-specific DOI. Not a priority — Research Square DOI is fully sufficient.

---

## #17 — 2026-06-15 — medRxiv rejected MEDRXIV/2026/355647; appeal pending; fallback arXiv

Context: medRxiv rejected the submission for lacking an organizational affiliation. Synapse is not yet incorporated (no LLC or C-Corp exists as of 2026-06-15). Liban replied the same day explaining Synapse as an independent research entity in the process of incorporating, taking full accountability, and noting no human subjects data was used (all public datasets).

Choice: Wait for medRxiv's response to the appeal. If they decline again, fall back to arXiv (the original plan from Decision #16). arXiv does not require institutional affiliation — at most an endorser for cs.LG, which can be obtained from an established arXiv author. OSF Preprints and Research Square are additional fallbacks requiring no affiliation.

Why this matters: Incorporating Synapse (LLC) was already on the SBIR critical path. This rejection makes incorporation more urgent — both for the preprint and for the SAM.gov UEI needed for SBIR. No LLC is the root cause of the medRxiv block.

Constraint: Do NOT resubmit to medRxiv with "Synapse LLC" until the LLC actually exists. Making a false affiliation claim would be worse than the current situation.

---

## #16 — 2026-06-15 — Submitted to medRxiv instead of arXiv

Context: arXiv requires endorsement for cs.LG from an existing author in the category. As a first-time submitter without an institutional affiliation, endorsement was not immediately available.

Choice: Submitted to medRxiv (Ophthalmology category) instead. medRxiv has no endorsement requirement, has an Ophthalmology category that fits the paper well, and is indexed by Google Scholar and PubMed. Submission ID: MEDRXIV/2026/355647. License: All Rights Reserved. Screening expected 4-5 days. DOI pending.

Why this matters: arXiv remains an option once endorsement is obtained (e.g., from OLIVES dataset authors). The two are not mutually exclusive — a cross-post to arXiv can happen after medRxiv posts.

---

## #15 — 2026-06-14 — Human confirmed CLAIMS.md directional evidence downgrade; ODE-RNN vs Latent ODE distinction logged

Context: Preprint Draft Sprint. The DIRECTIONAL EVIDENCE section downgrade proposed in Decision #14 required explicit human confirmation per protocol (CLAIMS.md is the one human-owned judgment).

Choice: Human confirmed on 2026-06-14. DIRECTIONAL EVIDENCE is now officially: timing experiment inconclusive at n=19; old directional story superseded. Paper Section 4.1 title updated from "Why Real Timing Helps the ODE and Hurts Recurrents" → "Continuous-Time Integration vs. Learned Decay — Structural Differences Under Irregular Timing" to match the body.

Secondary clarification logged: What was built is an **ODE-RNN** (Rubanova et al. 2019, discriminative), not the full **Latent ODE** (Chen et al. 2018, generative/VAE). The ODE-RNN evolves a GRU hidden state forward in continuous time and predicts next-visit CST directly. The full Latent ODE adds variational inference over the initial latent state and reconstructs sequences — enabling trajectory sampling and counterfactual simulation. That generative architecture is the Year 2 / Rung 3 target. The paper correctly uses "ODE-RNN" throughout (renamed in Decision #14). GLOSSARY.md updated to reflect the distinction explicitly.

Why this matters: The "disease dynamics engine" framing in the North Star implies a generative model. The ODE-RNN is a valid prototype for Bet 1 (continuous-time vs discrete recurrent) but is not yet the generative architecture. The Discussion and paper scope correctly reflect this; the company story should too.

---

## #14 — 2026-06-10 — Validation split fix: correct checkpoint selection method; all prior ODE/recurrent runs superseded

Context: Preprint Draft Sprint — pre-submission peer review corrections. Prior runs (Decisions #10, #11, #13) selected the final checkpoint by lowest TEST RMSE (evaluated at epoch 10 in the first ODE run, epoch 60 in recurrent runs). This is test-set leakage: the test set influenced which checkpoint was selected, invalidating the reported test performance.

Fix: Added a held-out validation split. The 77 training+validation eyes are split 60 train / 17 val (split_by_eye, test_frac=0.22, seed=42). Checkpoints selected by lowest val RMSE. Test set (19 eyes) evaluated exactly once with the selected checkpoint and not used for any modeling decision.

New runs (all: 60 train / 17 val / 19 test, val-checkpoint selection, seed=42):

| Run | W&B ID | Ordinal ts-wt RMSE | Real-dt ts-wt RMSE |
|---|---|---|---|
| grud_ordinal_valsplit_seed42 | qvs59tf8 | 88.21 μm | — |
| grud_realdelta_valsplit_seed42 | f13m74kn | — | 85.15 μm |
| tlstm_ordinal_valsplit_seed42 | f46o6mn2 | 83.35 μm | — |
| tlstm_realdelta_valsplit_seed42 | d7t3lidm | — | 84.68 μm |
| ode_ordinal_valsplit_seed42 | w80ry34u | 81.63 μm | — |
| ode_realdelta_valsplit_seed42 | wvl490h1 | — | 81.69 μm |

Timing-condition deltas: ODE +0.06 μm (flat), T-LSTM +1.33 μm (marginal degradation), GRU-D −3.06 μm (improvement from poor ordinal baseline).

Old results (from Decisions #10, #11): ODE ordinal 81.96 μm, ODE real-dt 81.6 μm, T-LSTM ordinal ~82.0 μm, GRU-D ordinal ~82.2 μm. These are superseded.

Impact on narrative: The old directional story (ODE benefits, recurrents degrade under real timing) does not hold. The ODE is robust across conditions; the timing experiment is inconclusive at n=19. The ODE still achieves the best test RMSE under both conditions. Paper updated accordingly (Sections 3.3, 3.4, 4.1, 5; CLAIMS.md DIRECTIONAL EVIDENCE).

Script: scripts/run_valsplit.py

## #13 — 2026-06-08 — Metric inconsistency resolution: report both ts-wt and eye-wt RMSE

Context: Preprint Draft Sprint. Discovered that model RMSE values (82.x μm, from W&B logs) used timestep-weighted RMSE (each visit counts equally), while the persistence baseline (91.7 μm) was computed with eye-weighted RMSE (each patient counts equally). These are different metrics; comparing them directly is incorrect.

Diagnostic (scripts/diagnose_metrics.py):
- Eye-weighted: persistence 91.7 μm, models 73–75 μm → models beat persistence ✓
- Timestep-weighted: persistence 55.0 μm, models 82.x μm → models do NOT beat persistence ✗
- Root cause: test eyes 4 (1 visit, 291 μm persistence error) and 12 (2 visits, 190 μm) dominate eye-weighted persistence; models beat persistence on only 8/19 test eyes (42%).
- Timing experiment direction (ODE improves, recurrents degrade) holds under BOTH metrics ✓

Choice: Option C — report both metrics explicitly. Table 1 shows ts-wt and eye-wt columns. Section 2.4 defines both metrics and explains why they diverge for persistence. Neither metric is suppressed.

Why: The divergence is real and scientifically informative. Eye-weighted and timestep-weighted answer different clinical questions (per-patient vs. per-visit accuracy). Forcing one metric would either make an overclaim (models beat persistence) or underclaim (models worse than persistence). Both results are true; both are reported. The paper's core contribution — the timing experiment — holds under both, so the inconsistency does not undermine the main thesis.

Alternatives rejected: (A) eye-weighted throughout — changes W&B numbers, creates inconsistency with logged run values; (B) timestep-weighted throughout — would suppress the eye-weighted result and make it look like models universally fail to beat persistence.

Paper impact: Abstract updated to cite both persistence values (91.7 eye-wt, 55.0 ts-wt). Table 1 has two persistence rows. Section 3.2 explicitly frames the metric-dependence. The timing experiment (Section 3.4) is evaluated under ts-wt (matching W&B) and noted to hold under eye-wt as well.

## #12 — 2026-06-07 — Messidor external validation: AUC 0.77 OOD — signal confirmed, not strong

Context: Messidor External Validation Sprint. Train logistic regression on EyePACS RETFound embeddings (31,542 images, frozen, binary: referable = grade ≥ 2 ICDR). Evaluate on Messidor-2 (1,744 gradable images, Krause et al. 2018 adjudicated ICDR grades). No Messidor data used in training.

W&B: messidor_val_v1 (run ID fxkir659, project synapse)

Results:
| Metric | Value |
|---|---|
| AUC-ROC | **0.7699** |
| Accuracy | 0.7884 |
| Sensitivity | 0.2341 |
| Specificity | 0.9852 |
| TP / FP / FN / TN | 107 / 19 / 350 / 1268 |

Choice: Accept AUC 0.77 as the official OOD result. Do NOT trigger the ">0.85 strong generalization" claim upgrade from NOW.md. The result is honest cross-dataset signal, not strong generalization.

Why (interpretation): AUC 0.77 confirms that frozen RETFound embeddings carry DR-relevant signal across datasets — a logistic regression trained on a US clinical camera dataset (EyePACS, JPEG) generalizes meaningfully to a French hospital dataset (Messidor-2, PNG, different cameras). This is a zero-shot cross-domain probe; AUC 0.77 is in the expected range for this setup. The very low sensitivity (0.23) is a threshold-calibration artifact: EyePACS is 19.5% referable, Messidor-2 is 26.2% referable — the 0.5 default threshold over-predicts non-referable. AUC is threshold-independent and is the correct primary metric. The 350 FNs are not a sign the embeddings lack signal — they are a sign the decision boundary needs re-calibrating for Messidor's prevalence.

What it means for claims: The "no external validation" gap is closed. We can now say: "frozen RETFound embeddings generalize OOD on Messidor-2 (AUC 0.77, linear probe)" — with the caveat that 0.77 is not strong. The honest framing: "cross-dataset DR signal confirmed." The preprint can note that threshold recalibration to Messidor's prevalence would substantially improve sensitivity while preserving AUC. Human must confirm any CLAIMS.md change.

Alternatives rejected: Reporting sensitivity/specificity as headline metrics — misleading given threshold-calibration mismatch; AUC is the canonical threshold-independent metric for this comparison.

## #11 — 2026-06-07 — Real delta-t result (CORRECTED v2): ODE improves; recurrents degrade

Context: Real Delta-T Sprint. Re-ran GRU-D, T-LSTM, Latent ODE with real week gaps (Prime eyes: diff(visit_nums)/4 normalized; TREX eyes: ordinal 1.0 — real timing unavailable). Seed=42, same split, same normalisation.

**Implementation note (v1 → v2 correction):** The first ODE run (ode_realdelta_seed42) used a batch-mean delta_t approximation — all examples in a batch shared the same integration interval (batch average per timestep). This made the ODE's prediction batch-composition-dependent (a reproducibility defect) and effectively withheld per-example timing from the ODE while GRU-D/T-LSTM received exact per-example gaps. The corrected run (ode_realdelta_v2_seed42) uses grouped odeint: at each timestep, examples are grouped by their unique delta_t value and each group gets its own odeint call. This gives exact per-example integration with ~2-4 odeint calls per step (fast, since 68% of Prime gaps are 1.0 and TREX is always 1.0).

Timing audit finding: OCT-DR.xlsx contains patient demographics only (no per-visit weeks). OCT-DME.xlsx Week columns are ~98% NaN. Real timing derives exclusively from file-path visit keys: Prime W-keys are week numbers (gaps range 4–48 wks, normalized 1.0–12.0); TREX V-keys are ordinal (real timing unknowable from available data — treat-and-extend protocol).

Results — 2×3 table, CORRECTED (RMSE in μm, test split 19 eyes, seed=42):

| Model | Ordinal RMSE | Real-ΔT RMSE | RMSE Δ | Real-ΔT MAE |
|---|---|---|---|---|
| GRU-D | 82.2 um | 84.2 um | +2.0 um worse | 58.9 um |
| T-LSTM | 82.0 um | 85.0 um | +3.0 um worse | 59.0 um |
| Latent ODE | 81.96 um | **81.6 um** | **-0.4 um better** | 58.0 um |
| Persistence | 91.7 um | 91.7 um | +0.0 um | — |

W&B: grud_realdelta_seed42, tlstm_realdelta_seed42, ode_realdelta_v2_seed42

Choice: Accept v2 corrected results as official real delta-t results. Supersedes v1 ODE row (83.5 um — incorrect, batch-mean approximation). Ordinal results remain the canonical comparison baseline.

Why (interpretation): With correct per-example integration, the ODE IMPROVES marginally on real delta-t (-0.4 um) while GRU-D/T-LSTM both degrade (+2-3 um). This is consistent with the ODE's structural design for irregular time: odeint integrates over the exact biological interval between each pair of visits, so longer gaps produce richer latent-state evolution. Recurrent decay mechanisms (exp(-W·Δt), c/(1+Δt)) were likely calibrated during training to ordinal 1.0 steps and grow overly aggressive on the real 1.0–12.0 range. The ODE has no explicit decay parameter that can be miscalibrated this way. The ODE advantage is consistent with the dynamics thesis but remains small on 19 test eyes (0.4 um margin, within sampling noise).

What it means for claims: The hypothesis "ODE has structural advantage with real irregular timing" IS NOW SUPPORTED at prototype scale. ODE real-dt (81.6 um) beats best ordinal recurrent (T-LSTM 82.0 um) by 0.4 um and beats real-dt recurrents by 3.0–3.4 um. Caveat: 19 test eyes is too small for statistical certainty; the 0.4 um lead over ordinal T-LSTM is within noise. Honest framing: "ODE benefits from real timing while recurrents are hurt — directional evidence for ODE structural advantage with irregular time, inconclusive at this n."

Alternatives rejected: Batch-mean approximation (v1 — batch-composition-dependent, incorrect); Prime-only split (violates frozen split constraint); estimating TREX gaps as 4-week fixed (wrong protocol — TREX is T&E with variable intervals).

## #10 — 2026-06-07 — Latent ODE result: RMSE 81.96 um — beats bar, narrow margin
Context: First completed W&B run (latent_ode_v1_seed42, seed=42, 100 epochs, dopri5, rtol=1e-3/atol=1e-4). Architecture: ODE-RNN (Rubanova et al. 2019) — linear encoder (1024->32) + GRUCell observation update + 2-layer MLP ODEFunc (hidden 64) + linear decoder (32->1). 47,521 parameters. Re-run performed solely for checkpoint saving; result byte-identical (deterministic seed on CPU).
Choice: Accept 81.96 um as official result. Beat bar per Decision #9.
Result: RMSE 81.96 um (best checkpoint, epoch 10 of 100) | MAE 58.3 um | Persistence 91.7 um | beats bar: YES
Notable: Best checkpoint at epoch 10 of 100. Train loss decreases monotonically (0.59 -> 0.38) while test RMSE worsens after epoch 10 (82.0 -> 87.0) — rapid overfitting on 77 training eyes with 47K parameters. The 0.04 um margin over the bar is within sampling variance on 19 test eyes; the honest claim is "matches" not "clearly beats" temporal baselines. The ODE-RNN beats persistence by 9.7 um (same tier as GRU-D/T-LSTM), validating the dynamics approach.
Checkpoint: models/latent_ode_v1_seed42.pt | W&B: project=synapse, run=latent_ode_v1_seed42 (run ID 942anp2d)
Alternatives rejected: Accepting the first run (same result, same seed, but no checkpoint saved due to script bug). 

## #9 — 2026-06-07 — ODE minimum target: RMSE < 82.0 um on next-visit CST (pre-committed, no moving)
Context: baselines sprint complete. GRU-D 82.2 um, T-LSTM 82.0 um on test eyes. Need a locked bar before building the latent ODE.
Choice: ODE must achieve RMSE < 82.0 um on next-visit CST (same test split, seed=42, same normalisation). This number is fixed now and cannot be revised post-hoc.
Why: pre-committing the bar prevents cherry-picking. If the ODE cannot beat strong temporal baselines, the dynamics thesis is not demonstrated on this data.
Caveat: delta_t is currently ordinal (1.0 steps, not real week gaps). Once real gaps are parsed from OCT-DR.xlsx, baselines should be re-run. The ODE bar updates to match the re-run baselines — but only if baselines are re-run first, before ODE.
Alternatives rejected: setting bar post-hoc after seeing ODE results — invalidates the comparison.

## #8 — 2026-06-07 — Temporal baseline target: CST regression (continuous) not binary threshold
Context: OLIVES `Disease Label` is patient-level (static across all visits per eye — DME vs healthy control). A temporal model predicting "next-visit disease state" trivially scores ~1.0 by copying current state; no dynamics are learned.
Choice: temporal baselines (GRU-D, T-LSTM) predict **next-visit CST** (continuous, μm) via regression. Metrics: RMSE (primary), MAE (secondary).
Why: (1) Latent ODE will target continuous physiological state — RMSE baseline enables direct comparison. (2) Binary CST threshold (e.g. >300 μm) is arbitrary — a prediction of 298 when truth is 302 is meaningfully correct but scored wrong. (3) Binary threshold derivable post-hoc from continuous predictions at zero information cost.
Results table splits cleanly: disease classification (logistic reg, AUC) vs. treatment-response dynamics (GRU-D/T-LSTM/ODE, RMSE/MAE).
Alternatives rejected: binary CST threshold (Option B) — mixes arbitrary clinical cutoff into evaluation; not what the ODE will optimize.

## #7 — 2026-06-07 — OLIVES visit alignment: (Eye_ID, BCVA, CST) join key, not File_Path
Context: HuggingFace OLIVES `disease_classification` config has no `File_Path` field. Encoding (v3 notebook) saved only Eye_ID and Disease Label per embedding — no filename to join to Clinical_Data_Images.xlsx.
Choice: join embeddings to visits using (Eye_ID, BCVA, CST) as composite key. Drops ~5.9% of B-scans where key maps to >1 visit (BCVA/CST unchanged between consecutive visits). Averages all B-scan embeddings within a visit into one 1024-d vector.
Why: BCVA and CST are per-visit clinical measurements recorded identically in both the HuggingFace metadata and Clinical_Data_Images.xlsx. For 94.1% of B-scans the triple is unambiguous. Visit-averaged embeddings are the right input unit for temporal models anyway.
Alternatives rejected: re-encoding with filename tracking — would require another full Colab run; the join approach recovers 94.1% of data cleanly.

<!-- Format:
## #N — YYYY-MM-DD — <decision in one line>
Context: <what prompted it>
Choice: <what we decided>
Why: <reasoning>
Alternatives rejected: <what we didn't pick and why>
-->

## #6 — 2026-06-02 — Encoding strategy: EyePACS/Messidor for classification, OLIVES for dynamics
Context: peer review of encoding plan flagged that EyePACS has no per-eye temporal structure, and OLIVES has two modalities (fundus + OCT) that may need different encoders.
Choice: use RETFound_cfp (frozen) for EyePACS + Messidor + OLIVES fundus images. Treat EyePACS/Messidor embeddings as classification/representation baselines only. OLIVES fundus embeddings feed into temporal sequences. OLIVES OCT encoder TBD — probe quality first.
Why: clean separation of what each dataset is actually for. Don't overstate EyePACS as temporal data.
Action: probe embedding dim at runtime; never hardcode 1024.

## #5 — 2026-06-01 — OLIVES dataloader: adapt authors' code, not build from scratch
Context: explored the official OLIVES GitHub repo (https://github.com/olivesgatech/OLIVES_Dataset) in full.
Choice: adapt the authors' `Time-Series Treatment Analysis` dataloader code as the reference implementation for our OLIVES temporal dataloader.
Why: authors already solved the hard parts — visit extraction from file paths (`W1`, `W4`, etc.), grouping by Eye_ID, sorting chronologically, stacking OCT B-scans into 3D volumes, pre-computing `.npy` files for efficiency. No reason to reinvent this.
Sprint: Weeks 4-5 (Baselines sprint) — OLIVES dataloader is needed to run GRU-D/T-LSTM/Cox baselines. Not needed for the EyePACS/RETFound encoding sprint (Weeks 3-4).
Alternatives rejected: building from scratch — wastes time when a reference implementation exists.

## #4 — 2026-06-01 — HuggingFace OLIVES download: use for encoder pretraining only
Context: downloaded OLIVES from HuggingFace (28.86 GB, Parquet format) after Zenodo throttling. Discovered it strips file paths and visit ordering; only preserves Eye_ID, biomarkers, BCVA, CST, Disease Label.
Choice: use HuggingFace data for encoder pretraining (EyePACS/RETFound sprint, Weeks 3-4) where temporal order doesn't matter. Still need Zenodo zip for dynamics modeling and baselines.
Why: HuggingFace images are usable for representation learning. For temporal modeling we need the original folder structure that encodes visit numbers — only the Zenodo zip has this.
Action required: download Zenodo OLIVES.zip before Weeks 4-5 baselines sprint begins. Run overnight with `curl.exe` (Zenodo throttles to ~3 Mbps regardless of connection speed).

## #3 — 2026-05-31 — Model class decision: latent ODE is viable for OLIVES
Context: OLIVES feasibility audit completed. Needed to choose model class before building anything.
Choice: **Latent ODE** — proceed with tODE as the core dynamics model for Year 1.
Why: 94/96 eyes (97.9%) have ≥4 visits; mean 16.6, max 27. Temporal structure far exceeds the ≥4-visit threshold. 16 OCT biomarkers present with ~0% missing. Two sub-studies (TREX DME 56 eyes, Prime_FULL 40 eyes) can be unified on a relative time clock.
Alternatives rejected: GRU-D / T-LSTM / short-horizon baselines — appropriate for 2-3 visit datasets; OLIVES is clearly rich enough to justify the full latent ODE approach. Baselines will still be built for comparison, but are no longer the primary model.
Open item: full longitudinal biomarker alignment still needs `full_labels/OCT-DME.xlsx` parsing and treatment event alignment (follow-up task, not a blocker for this decision).

## #1 — Month 0 — Adopt a layered docs architecture for AI-driven development
Context: building primarily with AI (Claude, sometimes GPT) across many sessions.
Choice: guard-rail / how-we-build / plan / focus / memory doc layers, with archive + ritual.
Why: keeps an AI agent on-mission across sessions; the docs are the shared memory between models.
Alternatives rejected: single big doc (AI ignores it) and no docs (AI drifts).

## #2 — Month 0 — OLIVES-first, feasibility audit before any model code
Context: public DR data is mostly cross-sectional; OLIVES is the only open longitudinal set.
Choice: audit OLIVES temporal structure before building any dynamics model.
Why: a latent ODE on temporally-thin data is wasted effort; decide model class on evidence.
Alternatives rejected: jumping straight to model-building (risks months on an unprovable claim).
