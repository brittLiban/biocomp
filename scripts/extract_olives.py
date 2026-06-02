import zipfile
from pathlib import Path

base = Path('C:/Users/liban/Documents/biocomp/data/raw/olives')

print('Extracting outer zip...')
with zipfile.ZipFile(base / 'OLIVES.zip') as zf:
    zf.extractall(base)
print('Outer zip extracted.')

for inner in (base / 'OLIVES').iterdir():
    size_gb = inner.stat().st_size / 1e9
    print(f'  {inner.name}: {size_gb:.2f} GB')

print('\nExtracting TREX_DME.zip...')
with zipfile.ZipFile(base / 'OLIVES' / 'TREX_DME.zip') as zf:
    zf.extractall(base / 'OLIVES')
print('TREX_DME extracted.')

print('\nExtracting Prime_FULL.zip...')
with zipfile.ZipFile(base / 'OLIVES' / 'Prime_FULL.zip') as zf:
    zf.extractall(base / 'OLIVES')
print('Prime_FULL extracted.')

print('\nFinal structure:')
for p in sorted((base / 'OLIVES').iterdir()):
    print(f'  {p.name}')
