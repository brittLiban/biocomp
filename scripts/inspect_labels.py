import pandas as pd
import re
from pathlib import Path

base = Path('C:/Users/liban/Documents/biocomp/data/raw/olives/labels/OLIVES_Dataset_Labels')

# --- Clinical_Data_Images ---
cdi = pd.read_excel(base / 'ml_centric_labels/Clinical_Data_Images.xlsx')
print(f'Clinical_Data_Images: {len(cdi)} rows, {cdi["Eye_ID"].nunique()} unique eyes')
print(cdi.head(4).to_string())
print()

# Extract visit number from File_Path (e.g. /V1/, /V2/)
cdi['Visit'] = cdi['File_Path'].str.extract(r'/V(\d+)/')
print('Visit values found:', sorted(cdi['Visit'].dropna().unique().tolist()))

visits_per_eye = cdi.groupby('Eye_ID')['Visit'].nunique()
print('\nVisit count distribution (visits -> n_eyes):')
print(visits_per_eye.value_counts().sort_index())
print(f'\nTotal eyes: {visits_per_eye.shape[0]}')
print(f'Eyes with >= 4 visits: {(visits_per_eye >= 4).sum()}')
print(f'Eyes with >= 2 visits: {(visits_per_eye >= 2).sum()}')

# --- Biomarker file ---
bio_files = list((base / 'full_labels').glob('Biomarker*')) + list((base / 'ml_centric_labels').glob('Biomarker*'))
print('\n\nBiomarker files:', [f.name for f in bio_files])
for bf in bio_files[:1]:
    df = pd.read_excel(bf, nrows=5)
    print(f'\n{bf.parent.name}/{bf.name} columns:')
    print(list(df.columns))
    print(df.head(3).to_string())

# --- OCT-DME structure ---
dme = pd.read_excel(base / 'full_labels/OCT-DME.xlsx', header=None, nrows=10)
print('\n\nOCT-DME first 10 rows x 20 cols:')
print(dme.iloc[:, :20].to_string())
