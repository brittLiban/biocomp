> Parent: CLAUDE.md · Related: docs/RISKS.md (B1), docs/MILESTONES.md
> This doc: track grant/funding opportunities, deadlines, and application status.
> Changes: update when a new opportunity is identified, a deadline passes, or status changes.
>          Reviewed every quarterly review.

# Funding Tracker

Status key: [ ] not started · [~] drafting · [>] submitted / waiting · [x] closed (won/lost/passed)

## Active Opportunities

| # | Mechanism | Agency | Next deadline | Status | Notes |
|---|---|---|---|---|---|
| F1 | SBIR Phase I (R43) | NIH / NEI | Check grants.nih.gov | [ ] | ~$300K, 6-month project. NEI funds DR/retinal AI. Requires US company (LLC/Inc). Most relevant mechanism for Year 1 science. |
| F2 | SBIR Phase I | NSF | Rolling / check NSF.gov | [ ] | Technology focus; slightly harder fit for biomedical AI than NIH. Backup to F1. |
| F3 | ARPA-H | ARPA-H | Watch for open BAAs | [ ] | High-risk/high-reward; good fit for disease dynamics framing if a relevant BAA opens. No set cycle — monitor. |
| F4 | UK Research and Innovation (UKRI) | UKRI / Innovate UK | Check UKRI website | [ ] | Relevant if UW affiliation or UK collaborator materialises. Low priority until then. |
| F5 | Wellcome Trust — Digital Technologies | Wellcome | Check wellcome.org | [ ] | Accepts international PIs; strong interest in AI for neglected disease. Worth a look for Y2. |

## Submitted / In Review

_None yet._

## Closed

_None yet._

## SBIR Phase I — Detailed Plan (F1)

**What the $300K buys (6 months):**
- Salary: ~$120-150K — go full time, stop building alongside other income
- UK Biobank data access + GPU compute: ~$50-80K — run the scale experiment (10,000+ eyes), resolve the timing question definitively
- Clinical collaborator (subcontract): ~$50-70K — retinal specialist to consult on clinical validity, co-author next paper, open doors to patient data

**What Phase I produces:**
- Timing experiment replicated at scale (Bet 1 confirmed or killed)
- True Latent ODE trained on UK Biobank data
- Phase II application ($1.75M, 2 years) to add treatment conditioning

**Phase II is where the company becomes real. Phase I buys the answer to whether it's worth building Phase II.**

---

## Gap Analysis — How Far From a Submitted Phase I?

**Done:**
- [x] Science: preprint submitted (MEDRXIV/2026/355647) — feasibility demonstrated
- [x] Clear next experiment: scale timing test on UK Biobank
- [x] Clear budget rationale

**Not done (in order):**
- [ ] Incorporate Synapse as LLC or C-Corp (~1-2 weeks, ~$100-500 via Stripe Atlas or state filing)
- [ ] SAM.gov registration — get UEI number (2-4 weeks processing, free, required for all federal grants)
- [ ] UK Biobank application submitted (shows reviewers you have a data plan)
- [ ] Specific Aims page — 1 page, the most important document (~1-2 weeks to draft)
- [ ] Research Strategy — 6 pages: Significance, Innovation, Approach (~3-4 weeks)
- [ ] NIH Biosketch (CV in NIH format) (~1 week)
- [ ] Budget + budget justification (~1 week)
- [ ] Email NEI Program Officer for informal fit feedback (do this early — before full draft)

**Realistic target: December 5, 2026 deadline**
August 5 is 7 weeks away — too tight given incorporation + SAM.gov alone takes 2-4 weeks.
December 5 gives ~5.5 months: comfortable timeline to incorporate, register, write, and submit properly.

**Total gap: ~3-4 months of focused work.**

---

## Notes on SBIR (F1)
- Company must be incorporated in the US before applying (LLC or C-corp).
- Phase I: up to ~$300K, 6 months, feasibility focus — fits exactly where we are.
- Phase II: up to ~$1.75M, 2 years — follows a successful Phase I.
- NEI (National Eye Institute) program officer contact is worth an email before submitting — they give informal feedback on fit. Find the right PO at nei.nih.gov/about/offices-branches/dera.
- Specific Aims page is the most important 1-pager — worth drafting during the preprint sprint since the science overlaps heavily.
- Key reuse: preprint abstract ≈ SBIR significance section. Draft them together.
