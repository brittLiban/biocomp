> Parent: CLAUDE.md · Related: docs/Q1_PLAN.md
> This doc: dataset schemas, access status, and audit findings. AI reads this for real data shape.
> Changes: as datasets are acquired and audited. The OLIVES findings section is filled by the audit.

# Data

## Large Dataset Download Protocol
Hard-won lessons — read before starting any large download:

- **Use `curl.exe -L -C - -o <dest> <url>`** — the `-C -` flag resumes interrupted downloads. Never use BITS (`Start-BitsTransfer`) for large files: it holds data in a temp location and loses everything if cancelled before `Complete-BitsTransfer` is called.
- **Zenodo throttles to ~3 Mbps** regardless of your connection speed. A 33 GB file takes ~23 hours. Plan for an overnight run. Leave machine on, don't sleep it.
- **HuggingFace mirrors may strip metadata.** The OLIVES HuggingFace version (gOLIVES/OLIVES_Dataset) has images but no file paths or visit ordering — unusable for temporal modeling. Always check what columns are present before assuming a mirror is equivalent to the original.
- **Check Zenodo status** at zenodo.org before starting if downloads fail with 504 errors — outages happen.
- **Resuming:** if curl drops, just rerun the exact same command with `-C -`. It picks up where it left off.

## Status Overview
| Dataset | Access | Status | Role |
|---|---|---|---|
| OLIVES | Free, CC BY 4.0 | [x] labels [x] Zenodo extracted (162,864 files) [x] audited [x] encoded (78,822 × 1024) | Dynamics PoC |
| EyePACS public | Free (Kaggle) | [x] downloaded [x] encoded (31,542 × 1024) | Encoder pretraining |
| Messidor / -2 | Free (ADCIS) | [x] downloaded + extracted (1,748 images, data/raw/messidor/images/) | External validation |
| FGADR | Free | [ ] downloaded | Lesion-aware aux |
| UK Biobank | ~£9K | [ ] applied | Scale (Y2) |
| EyePACS private | Partnership | [ ] inquired | Scale (Y2) |

## OLIVES Feasibility Audit Findings
> Completed: 2026-05-31. Source: `ml_centric_labels/Clinical_Data_Images.xlsx` + `Biomarker_Clinical_Data_Images.csv`.

### File/Folder Structure
Two sub-studies bundled together:
- **TREX DME** (56 eyes): paths like `/TREX DME/GILA/<PatientID>/V<n>/<eye>/image.tif` — visits indexed V1..V27
- **Prime_FULL** (40 eyes): paths like `/Prime_FULL/<PatientID>/W<week>/<eye>/image.png` — visits indexed by week (W0, W4, W8... up to W104)

Labels live in two folders:
- `full_labels/` — raw clinical tables (OCT-DME.xlsx wide format with per-visit biomarkers; OCT-DR.xlsx with patient demographics + treatment arm)
- `ml_centric_labels/` — image-indexed CSVs/XLSXs ready for dataloader use

### Visit Depth (from `Clinical_Data_Images.xlsx`, 78,185 image rows)
- Eyes total: **96**
- Eyes with ≥2 visits: **96 / 96 (100%)**
- Eyes with ≥4 visits: **94 / 96 (97.9%)**
- Eyes with ≥6 visits: **91 / 96 (94.8%)**
- Mean visits per eye: **16.6** · Median: **17** · Max: **27** · Min: **3**
- Histogram saved: `results/figures/olives_visit_histogram.png`

### 16 Biomarkers
All 16 OCT biomarkers present in `Biomarker_Clinical_Data_Images.csv`:
Atrophy/thinning, Disruption of EZ, DRIL, IR hemorrhages, IR HRF, Partially attached vitreous face, Fully attached vitreous face, Preretinal tissue/hemorrhage, Vitreous debris, VMT, DRT/ME, Fluid (IRF), Fluid (SRF), Disruption of RPE, PED (serous), SHRM.
Missing data rate: **~0%** (< 0.1% on two biomarkers).

### Alignment: Imaging ↔ Biomarkers ↔ Treatment
- **Imaging ↔ 16 biomarkers**: aligned at image level in `Biomarker_Clinical_Data_Images.csv` (keyed by file path). Confirmed for 9,408 rows × 96 eyes.
- **Note**: the `ml_centric` biomarker CSV appears to cover only ~2 visit timepoints per eye (baseline + one follow-up); the `full_labels/OCT-DME.xlsx` (195 columns, VisitNums 1–28) likely holds the full longitudinal biomarker table — **needs parsing**.
- **Treatment/injection**: `full_labels/OCT-DR.xlsx` has patient-level demographics (age, gender, HbA1c at W24/W52/W76/W104, baseline BCVA/CST) — **no per-visit week data**. See timing audit below.
- **Time clock**: TREX DME uses visit index V1..Vn; Prime_FULL uses weeks W0..W104. Both can be converted to a common relative timeline.

### Per-Visit Timing Audit (Real Delta-T Sprint, 2026-06-07)

| File | Contents | Usable for timing? |
|---|---|---|
| `full_labels/OCT-DR.xlsx` | 41×20: patient demographics (age, gender, HbA1c at W24/52/76/104, baseline BCVA/CST, treatment arm) | **No** — no per-visit data |
| `full_labels/OCT-DME.xlsx` | 43×195: VisitNums 1–28, each block has VisitDate+Week+ETDRS+Snellen+Oct columns | **No** — Week columns are ~98% NaN (only 10 non-null values across the entire file) |
| `ml_centric_labels/Clinical_Data_Images.xlsx` | 78185×5: File_Path, BCVA, CST, Eye_ID, Patient_ID | **No** — no timing columns |

**Conclusion**: Real per-visit week data exists only in the file paths themselves:
- **Prime_FULL (40 eyes)**: visit keys are `W{n}` → visit_nums ARE week numbers. Week gaps computed as `diff(visit_nums) / 4` (normalized to 4-week units). Gaps range 1.0–12.0 (min=4 wks, max=48 wks real). Genuine variation: 68% of gaps are 1.0, 25% are 2.0, 7% larger.
- **TREX DME (56 eyes)**: visit keys are `V{n}` → ordinal only. TREX is a treat-and-extend protocol (variable intervals, not fixed monthly). OCT-DME Week columns are empty. Real timing unavailable — `week_gaps` field set to 1.0 per step.

**Implementation**: `build_sequences()` (v2 cache, `olives_sequences_v2.pkl`) adds `week_gaps` field per eye using this logic. Normalized by 4 so 1.0 ≈ one 4-week interval, consistent across sub-studies.

### Largest Clean Longitudinal Subset
94 eyes with ≥4 visits; 56 eyes with ≥16 visits.

### **MODEL CLASS DECISION: Latent ODE — VIABLE**
97.9% of eyes have ≥4 visits; mean 16.6. Temporal structure is richer than required. Logged in DECISIONS.md #3.

## OLIVES Reference Code
Official repo: https://github.com/olivesgatech/OLIVES_Dataset
Key folder for us: `Time-Series Treatment Analysis/` — contains working temporal dataloaders, ResNet18+LSTM model, and visit-extraction logic. Adapt this for our Weeks 4-5 baselines sprint. Do NOT build OLIVES dataloader from scratch.

## OLIVES Known Facts (from dataset description)
96 eyes; 1,268 near-IR fundus images; each fundus paired with ~49 OCT scans;
16 biomarkers; DR/DME diagnosis labels; treatment/injection timing;
avg follow-up ≥2 years; avg treatment duration 66 weeks; avg ~7 injections/eye.
License: CC BY 4.0 (redistribution/reuse with attribution).
