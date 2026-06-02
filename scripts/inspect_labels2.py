import pandas as pd
import re
from pathlib import Path

base = Path('C:/Users/liban/Documents/biocomp/data/raw/olives/labels/OLIVES_Dataset_Labels')

# --- Visit depth for the 40 "0-visit" eyes ---
cdi = pd.read_excel(base / 'ml_centric_labels/Clinical_Data_Images.xlsx')
cdi['Visit'] = cdi['File_Path'].str.extract(r'/V(\d+)/')

zero_visit_eyes = cdi[cdi['Visit'].isna()]['Eye_ID'].unique()
print(f'Eyes with no /V#/ in path: {len(zero_visit_eyes)}')
sample_paths = cdi[cdi['Eye_ID'].isin(zero_visit_eyes[:3])]['File_Path'].head(6)
print('Sample paths:')
for p in sample_paths:
    print(' ', p)
print()

# Try alternate pattern (e.g. _W1_ or week number)
cdi['Visit2'] = cdi['File_Path'].str.extract(r'[/_]W(\d+)[/_]', flags=re.IGNORECASE)
print('Alt visit pattern (W#) values:', sorted(cdi['Visit2'].dropna().unique()[:10]))
print()

# Full visit distribution (combining both patterns)
cdi['VisitFinal'] = cdi['Visit'].combine_first(cdi['Visit2'])
visits_per_eye = cdi.groupby('Eye_ID')['VisitFinal'].nunique()
print('Final visit count distribution:')
print(visits_per_eye.value_counts().sort_index())
print(f'\nTotal eyes: {visits_per_eye.shape[0]}')
print(f'Eyes with >= 4 visits: {(visits_per_eye >= 4).sum()}')
print(f'Eyes with >= 2 visits: {(visits_per_eye >= 2).sum()}')
print()

# --- Biomarker CSV ---
bio = pd.read_csv(base / 'ml_centric_labels/Biomarker_Clinical_Data_Images.csv')
print(f'Biomarker CSV: {len(bio)} rows, {bio.shape[1]} cols')
print('Columns:', list(bio.columns))
print(bio.head(3).to_string())
print()

# How many unique eyes and visits in biomarker file?
if 'Eye_ID' in bio.columns:
    print(f'Unique eyes in biomarker file: {bio["Eye_ID"].nunique()}')
if 'Visit' in bio.columns or any('visit' in c.lower() for c in bio.columns):
    vcol = [c for c in bio.columns if 'visit' in c.lower()][0]
    print(f'Unique visits ({vcol}): {bio[vcol].nunique()}')

# Check biomarker column count (should be ~16)
non_id_cols = [c for c in bio.columns if c not in ['File_Path', 'Eye_ID', 'Patient_ID', 'BCVA', 'CST']]
print(f'\nNon-ID columns ({len(non_id_cols)}): {non_id_cols}')
