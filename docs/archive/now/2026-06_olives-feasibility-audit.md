> Parent: docs/Q1_PLAN.md (Weeks 1-2 chunk) · Constitution: CLAUDE.md
> This doc: THE current sprint. Fully specified. Rewritten each sprint via the closeout ritual.

# NOW — OLIVES Feasibility Audit

## Goal
Determine whether OLIVES has the temporal structure to support disease-dynamics modeling,
BEFORE writing any model code. This audit decides the model class for all downstream work.

## Why This First
A latent ODE needs real time-series per eye. If OLIVES is temporally thin, building an ODE
on it is wasted effort. We decide based on evidence, not hope.

## Inputs
- OLIVES dataset (download from Zenodo, DOI 10.5281/zenodo.7105232; OLIVES.zip ~33.9GB + labels zip)
- No prior code required. This is inspection, not modeling.

## Tasks
1. Download OLIVES + labels. Unzip into data/raw/olives/ (gitignored — large).
2. Inspect file/folder structure. Document how eyes, visits, fundus, OCT are organized.
3. Build a per-eye visit table: for each eye, list distinct timestamped visits.
4. Produce a histogram: how many eyes have 2, 3, 4, 5, 6+ visits.
5. Check alignment: can imaging timepoints align with the 16 biomarkers AND injection/treatment dates on one clock?
6. Determine the largest clean longitudinal subset (eyes with enough timestamped visits).

## Done When
- DATA.md "OLIVES Feasibility Audit Findings" section is filled in.
- The histogram is saved to results/figures/.
- The MODEL CLASS DECISION is made and logged in DECISIONS.md:
  - Most eyes have ≥4 visits → latent ODE viable.
  - Most eyes have 2-3 visits → scope to short-horizon + simpler temporal baselines.
- OPEN_QUESTIONS.md data items are checked off or updated.

## Hard Constraints
- NO model code this sprint. Inspection and decision only.
- This is a notebook task — notebooks/olives_feasibility_audit.ipynb.

## Next Up
Weeks 2-3: Reproducible infrastructure (W&B, seeds, env specs) — can run in parallel.
Then Weeks 3-4: EyePACS + Messidor pipeline + RETFound encoding.
