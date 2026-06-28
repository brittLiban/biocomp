> Parent: CLAUDE.md · Related: docs/MILESTONES.md, docs/DECISIONS.md
> This doc: live risk register. Review at each decision gate. Retire resolved risks; escalate new ones.
> Changes: at each gate (Month 3, 6, 12, 18, 24, 36) + when a risk materially changes.

# Risk Register

Status: [ ] active · [~] watch · [x] retired · [!] escalated

## Scientific

| # | Risk | Sev | Status | Mitigation |
|---|---|---|---|---|
| S1 | OLIVES temporal depth too thin for latent ODE | MED | [x] RETIRED 2026-06-01 | Feasibility audit complete: 97.9% eyes ≥4 visits, mean 16.6 — viable (Decision #3) |
| S2 | Models learn imaging artifacts not disease signal | HIGH | [ ] | External validation (Messidor-2); subgroup analysis; clinician review |
| S3 | Disease dynamics not learnable from available data | HIGH | [ ] | Falsify early and pivot — accept this is a real possibility |
| S4 | OLIVES too small to support strong claims | HIGH | [ ] | Scope claims to prototype + small-n caveats; controlled data for scale |
| S5 | Architecture overfits OLIVES, fails on held-out data | MED | [ ] | Strict train/val/test split; external validation datasets |

## Data Access

| # | Risk | Sev | Status | Mitigation |
|---|---|---|---|---|
| D1 | All controlled-data paths stall (UKB, EyePACS, ACCORD, clinic) | MED | [~] | GRAPE (free, Figshare, download today) + DRCR Protocol T (form request, no fee) added as unblocked routes 2026-06-16 (Decision #21). UK Biobank deferred (£9K). Multiple paths active. |
| D2 | UK Biobank application rejected or delayed | LOW-MED | [ ] | Grant funding buffers cost; not personal runway; pursue parallel routes |
| D3 | EyePACS partnership negotiations slow or fail | MED | [ ] | Start early (Month 2-3); narrow initial ask; ACCORD/UKB are alternatives |
| D4 | ACCORD EYE has no raw images (only derived data) | MED | [ ] | BioLINCC request to confirm — do not assume raw images exist |

## Business

| # | Risk | Sev | Status | Mitigation |
|---|---|---|---|---|
| B1 | Pharma sales cycles too slow for revenue before runway runs out | HIGH | [ ] | Non-dilutive grants bridge (SBIR, NEI, ARPA-H) before venture capital |
| B2 | Well-resourced academic/industry labs replicate approach | MED | [ ] | Data + validation moats compound over time; publication speed matters |
| B3 | Regulatory classification (SaMD) triggered prematurely | MED | [ ] | Position as research tool initially; no clinical decision-support claims |
| B4 | IP entanglement if UW affiliation pursued | MED | [ ] | Settle ownership via tech-transfer office BEFORE using any UW resources |

## Personal / Execution

| # | Risk | Sev | Status | Mitigation |
|---|---|---|---|---|
| P1 | Competing commitments dilute focus below critical threshold | HIGH | [ ] | Sustainable cadence; AI leverage; protect deep-work blocks |
| P2 | Burnout from solo founder pace | HIGH | [ ] | Celebrate rung-by-rung progress; realistic quarterly scope |
| P3 | Isolation reduces scientific quality and judgment | MED | [ ] | Community engagement; cultivate advisors and peer reviewers |
| P4 | Premature financial pressure forces bad decisions | LOW | [ ] | Near-zero burn M1-6; do not quit income until evidence justifies it |

## Notes on Review Cadence
- Gate 1 (Month 3): review S1, S4, D1
- Gate 2 (Month 6): review S2, S3, D1-D4, B1
- Gate 3 (Month 12): full register review; retire what's resolved; escalate what worsened
