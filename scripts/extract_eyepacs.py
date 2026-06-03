"""
EyePACS split zip extraction.
The 5 train.zip.00X.zip files are parts of a single split archive.
We rename them to .zip.001-.005, then extract sequentially.
"""
import zipfile
import os
from pathlib import Path

RAW = Path('C:/Users/liban/Documents/biocomp/data/raw/eyepacs')
OUT = RAW / 'train'
OUT.mkdir(exist_ok=True)

parts = sorted(RAW.glob('train.zip.*.zip'))
print(f'Found {len(parts)} parts:')
for p in parts:
    print(f'  {p.name} ({round(p.stat().st_size/1e9, 2)} GB)')

# Extract labels first
labels_zip = RAW / 'trainLabels.csv.zip'
if labels_zip.exists():
    print('\nExtracting labels...')
    with zipfile.ZipFile(labels_zip) as zf:
        zf.extractall(RAW)
    print('Labels extracted.')

# Extract each part independently (each is a self-contained zip)
for part in parts:
    print(f'\nExtracting {part.name}...')
    with zipfile.ZipFile(part) as zf:
        zf.extractall(OUT)
    print(f'  Done.')

print(f'\nAll done. Images in: {OUT}')
total = sum(1 for _ in OUT.glob('*.jpeg'))
print(f'Total .jpeg files: {total}')
