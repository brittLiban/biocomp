> Parent: CLAUDE.md · Related: docs/DECISIONS.md (each run has a matching decision entry)
> This doc: index of all W&B training runs. One row per run — name, ID, script, key metric, decision.
> Changes: append a row every time a run is logged to W&B. Never delete rows (mark superseded instead).
> How to use: find a result → get the run ID → open W&B or grep DECISIONS.md for the full context.

# W&B Run Index

W&B project: **synapse** (most runs) · legacy: **synapse-v1** (encoding run)
Dashboard: https://wandb.ai/models-biocomp/synapse

## Runs

| Run name | Run ID | Date | Script | Key metric | Decision | Notes |
|---|---|---|---|---|---|---|
| retfound-encoding | — | 2026-06-01 | experiments/encode_datasets.py | EyePACS (31,542×1024), Messidor partial (745×1024) | #6 | project=synapse-v1; Messidor encoding was incomplete (no filename index) — superseded by messidor_val_v1 encoding |
| grud_seed42 | — | 2026-06-07 | scripts/baseline_grud.py | RMSE 82.2 um (ordinal) | #9 | Ordinal delta_t baseline; canonical comparison |
| tlstm_seed42 | — | 2026-06-07 | scripts/baseline_tlstm.py | RMSE 82.0 um (ordinal) | #9 | Ordinal delta_t baseline; sets the bar for ODE |
| latent_ode_v1_seed42 | 942anp2d | 2026-06-07 | scripts/latent_ode.py | RMSE 81.96 um (ordinal) | #10 | Best checkpoint epoch 10/100; rapid overfitting on 77 eyes |
| ode_realdelta_seed42 | — | 2026-06-07 | scripts/latent_ode.py | RMSE 83.5 um (real-dt) | #11 | **SUPERSEDED** — batch-mean delta_t approximation (reproducibility defect); replaced by v2 |
| grud_realdelta_seed42 | — | 2026-06-07 | scripts/baseline_grud.py | RMSE 84.2 um (real-dt) | #11 | Real delta_t; +2.0 um vs ordinal — recurrent degrades |
| tlstm_realdelta_seed42 | — | 2026-06-07 | scripts/baseline_tlstm.py | RMSE 85.0 um (real-dt) | #11 | Real delta_t; +3.0 um vs ordinal — recurrent degrades |
| ode_realdelta_v2_seed42 | — | 2026-06-07 | scripts/latent_ode.py | RMSE 81.6 um (real-dt) | #11 | Corrected: grouped per-example odeint; -0.4 um vs ordinal — ODE benefits from real timing |
| messidor_val_v1 | fxkir659 | 2026-06-07 | scripts/validate_messidor.py | AUC 0.7699 OOD | #12 | Frozen logreg; train=EyePACS, test=Messidor-2 (1,744); cross-dataset DR signal confirmed |
| grud_ordinal_valsplit_seed42 | 7aaazk8q | 2026-06-10 | scripts/run_valsplit.py | ts-wt 88.21 μm / eye-wt 76.70 μm (ordinal) | #14 | 60/17/19 split, val-checkpoint; SUPERSEDES prior grud runs |
| grud_realdelta_valsplit_seed42 | yjk28y9q | 2026-06-10 | scripts/run_valsplit.py | ts-wt 85.15 μm / eye-wt 74.23 μm (real-dt) | #14 | 60/17/19 split, val-checkpoint |
| tlstm_ordinal_valsplit_seed42 | xis5opkq | 2026-06-10 | scripts/run_valsplit.py | ts-wt 83.35 μm / eye-wt 73.07 μm (ordinal) | #14 | 60/17/19 split, val-checkpoint; SUPERSEDES prior tlstm runs |
| tlstm_realdelta_valsplit_seed42 | 03hzdy9j | 2026-06-10 | scripts/run_valsplit.py | ts-wt 84.68 μm / eye-wt 73.63 μm (real-dt) | #14 | 60/17/19 split, val-checkpoint |
| ode_ordinal_valsplit_seed42 | xfsrrnp1 | 2026-06-10 | scripts/run_valsplit.py | ts-wt 81.63 μm / eye-wt 70.89 μm (ordinal) | #14 | 60/17/19 split, val-checkpoint; SUPERSEDES prior ODE runs |
| ode_realdelta_valsplit_seed42 | 0gwhx722 | 2026-06-10 | scripts/run_valsplit.py | ts-wt 81.69 μm / eye-wt 71.90 μm (real-dt) | #14 | 60/17/19 split, val-checkpoint |
| true_latent_ode_v1_seed42 | xhuntnop | 2026-06-27 | scripts/true_latent_ode.py | RMSE 87.0 μm / MAE 63.7 μm / CI-cov 0.816 / KL 0.363 nats | #23 | Best ckpt epoch 10/150; RMSE FAILS bar (85 μm), CI PASSES, KL PASSES at best ckpt; KL peaked epoch 10 (0.36 nats) then declined to ~0.02–0.04 post-warmup (partial collapse pattern); RMSE stagnated ~87 μm after epoch 10; no hard collapse (KL never < 0.01); 77 train / 19 test eyes; real delta_t |
