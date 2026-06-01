> Parent: CLAUDE.md · Related: docs/DECISIONS.md (resolved questions graduate into decisions)
> This doc: running list of unresolved unknowns. Append when one surfaces; check off when answered.
> Changes: continuously, as questions arise and resolve.

# Open Questions

Status: [ ] open · [x] answered (then move the answer to DECISIONS.md if it's a choice)

## Data
- [ ] OLIVES temporal depth — how many eyes have ≥4 timestamped visits? (the audit answers this)
- [ ] Do OCT scans share a timestamp/clock with the paired fundus image?
- [ ] Can injection/treatment dates be aligned to the imaging timeline?
- [ ] Does ACCORD EYE include raw image files or only derived datasets?
- [ ] UK Biobank: exact independent-researcher eligibility + current fee?

## Strategy
- [ ] Will a UW affiliation materialize, and is Dr. Dargahi open to a bridge thesis? (bonus, not critical-path)
- [ ] Which controlled-data route is most realistic first: UK Biobank, EyePACS partnership, or local clinic?

## Technical
- [ ] Is OLIVES temporal structure rich enough for a latent ODE, or only simpler baselines? (audit decides)
- [ ] RETFound frozen vs fine-tuned for v1 — defer until after encoding works.
