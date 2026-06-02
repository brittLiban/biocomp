import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

base = Path('C:/Users/liban/Documents/biocomp/data/raw/olives/labels/OLIVES_Dataset_Labels')
FIGURES_DIR = Path('C:/Users/liban/Documents/biocomp/results/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

cdi = pd.read_excel(base / 'ml_centric_labels/Clinical_Data_Images.xlsx')
cdi['Visit'] = cdi['File_Path'].str.extract(r'/V(\d+)/')
cdi['Visit2'] = cdi['File_Path'].str.extract(r'[/_]W(\d+)[/_]', flags=re.IGNORECASE)
cdi['VisitFinal'] = cdi['Visit'].combine_first(cdi['Visit2'])

visits_per_eye = cdi.groupby('Eye_ID')['VisitFinal'].nunique()

fig, ax = plt.subplots(figsize=(9, 5))
counts = visits_per_eye.value_counts().sort_index()
ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white', width=0.8)
ax.axvline(3.5, color='red', linestyle='--', linewidth=1.5, label='4-visit threshold (latent ODE)')
ax.set_xlabel('Number of visits per eye', fontsize=12)
ax.set_ylabel('Number of eyes', fontsize=12)
ax.set_title('OLIVES — Visit depth per eye (96 eyes)', fontsize=13)
ax.legend(fontsize=10)
pct_4plus = (visits_per_eye >= 4).mean() * 100
ax.text(0.97, 0.95, f'{pct_4plus:.0f}% of eyes have ≥4 visits',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, color='darkred',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red', alpha=0.8))
plt.tight_layout()
out = FIGURES_DIR / 'olives_visit_histogram.png'
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
print(visits_per_eye.describe())
print(f'\nEyes >= 4 visits: {(visits_per_eye >= 4).sum()} / {len(visits_per_eye)}')

# --- Treatment alignment check ---
print('\n--- Treatment / injection alignment ---')
bio = pd.read_csv(base / 'ml_centric_labels/Biomarker_Clinical_Data_Images.csv')
bio['Visit'] = bio['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'].str.extract(r'/V(\d+)/')
bio['Visit2'] = bio['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'].str.extract(r'[/_]W(\d+)[/_]', flags=re.IGNORECASE)
bio['VisitFinal'] = bio['Visit'].combine_first(bio['Visit2'])

print(f'Biomarker rows: {len(bio)}, unique eyes: {bio["Eye_ID"].nunique()}')
print(f'Unique visits per eye (biomarker): {bio.groupby("Eye_ID")["VisitFinal"].nunique().describe()}')

# Check missing data in biomarkers
bio_cols = ['Atrophy / thinning of retinal layers', 'Disruption of EZ', 'DRIL',
            'IR hemorrhages', 'IR HRF', 'Partially attached vitreous face',
            'Fully attached vitreous face', 'Preretinal tissue/hemorrhage',
            'Vitreous debris', 'VMT', 'DRT/ME', 'Fluid (IRF)', 'Fluid (SRF)',
            'Disruption of RPE', 'PED (serous)', 'SHRM']
print(f'\nBiomarker missing data rate: {bio[bio_cols].isna().mean().mean():.1%}')
print('Per-biomarker NaN rate:')
print(bio[bio_cols].isna().mean().round(3))

# Sub-study breakdown
bio['Study'] = bio['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'].str.split('/').str[1]
print('\nSub-study breakdown:')
print(bio.groupby('Study')['Eye_ID'].nunique())
