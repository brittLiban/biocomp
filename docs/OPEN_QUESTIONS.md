> Parent: CLAUDE.md · Related: docs/DECISIONS.md (resolved questions graduate there, then are deleted here)
> This doc: running list of unresolved unknowns. Append when one surfaces; delete when answered.
> Protocol: when a question is answered → log it in DECISIONS.md → DELETE it from this file entirely.
> Changes: continuously, as questions arise and resolve.
> **Last reviewed: 2026-06-11**

# Open Questions

Status: [ ] open only — answered questions are deleted, not marked [x]. Answers live in DECISIONS.md.

## Data

- [ ] Do OCT scans share a timestamp/clock with the paired fundus image?
- [ ] Can injection/treatment dates be aligned to the imaging timeline? (`full_labels/OCT-DR.xlsx` has treatment arm + injection column; needs per-visit mapping)
- [ ] Does `full_labels/OCT-DME.xlsx` (195-col wide table, VisitNums 1–28) contain full longitudinal biomarkers for all visits? If so, pivot to long format.
- [ ] Does ACCORD EYE include raw image files or only derived datasets?
- [ ] UK Biobank: exact independent-researcher eligibility criteria + current fee?

## Strategy

- [ ] Will a UW affiliation materialize, and is Dr. Dargahi open to a bridge thesis? (bonus, not critical-path)
- [ ] Which controlled-data route is most realistic first: UK Biobank, EyePACS partnership, or local clinic?

## Technical

- [ ] RETFound frozen vs fine-tuned — defer until after preprint is live and Year 2 data is in hand.
- [ ] OLIVES OCT modality: does RETFound_cfp (colour fundus pretrain) work well on OCT slices, or do we need a separate OCT encoder? Probe embedding quality when revisiting encoding.
