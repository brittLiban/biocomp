> Parent: CLAUDE.md · Related: docs/Q1_PLAN.md
> This doc: dataset schemas, access status, and audit findings. AI reads this for real data shape.
> Changes: as datasets are acquired and audited. The OLIVES findings section is filled by the audit.

# Data

## Status Overview
| Dataset | Access | Status | Role |
|---|---|---|---|
| OLIVES | Free, CC BY 4.0 | [ ] downloaded [ ] audited | Dynamics PoC |
| EyePACS public | Free (Kaggle) | [ ] downloaded | Encoder pretraining |
| Messidor / -2 | Free (ADCIS) | [ ] downloaded | External validation |
| FGADR | Free | [ ] downloaded | Lesion-aware aux |
| UK Biobank | ~£9K | [ ] applied | Scale (Y2) |
| EyePACS private | Partnership | [ ] inquired | Scale (Y2) |

## OLIVES Feasibility Audit Findings
> FILL THIS IN after the Weeks 1-2 audit. Until then it is blank by design.

- File/folder structure: ____
- Eyes total: 96 (per dataset description)
- Distinct timestamped visits per eye (histogram): ____
- Eyes with ≥2 visits: ____  ≥4 visits: ____  ≥6 visits: ____
- Imaging ↔ biomarkers ↔ treatment alignment possible? ____
- Largest clean longitudinal subset: ____ eyes with ____ visits
- **MODEL CLASS DECISION:** ____ (latent ODE if ≥4 visits common; else simpler temporal baseline)
- Decision logged in DECISIONS.md? ____

## OLIVES Known Facts (from dataset description)
96 eyes; 1,268 near-IR fundus images; each fundus paired with ~49 OCT scans;
16 biomarkers; DR/DME diagnosis labels; treatment/injection timing;
avg follow-up ≥2 years; avg treatment duration 66 weeks; avg ~7 injections/eye.
License: CC BY 4.0 (redistribution/reuse with attribution).
