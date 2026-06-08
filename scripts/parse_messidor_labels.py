"""
Parse Messidor-2 adjudicated label file → data/processed/messidor_labels.csv

Handles the Google/Krause adjudicated CSV (adjudicated_dr_grade column)
and common ADCIS XLS/CSV variants.

Usage:
  python scripts/parse_messidor_labels.py --input data/raw/messidor/annotations/<file>
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("C:/Users/liban/Documents/biocomp/data/processed/messidor_labels.csv")
IMAGES_DIR = Path("C:/Users/liban/Documents/biocomp/data/raw/messidor/images/IMAGES")


def read_label_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    print(f"Loaded {path.name}: {df.shape}, columns: {df.columns.tolist()}")
    return df


COLUMN_ALIASES = {
    # image id candidates
    "image_id": ["image_id", "Image name", "image", "filename", "Filename", "name"],
    # grade candidates
    "dr_grade": [
        "adjudicated_dr_grade",  # Google Krause standard
        "Retinopathy grade",     # ADCIS Messidor-1 XLS
        "retinopathy_grade",
        "dr_grade",
        "grade",
        "DR_grade",
    ],
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to the ADCIS/Google annotation file")
    args = parser.parse_args()

    src = Path(args.input)
    assert src.exists(), f"Not found: {src}"

    df = read_label_file(src)

    id_col    = find_column(df, COLUMN_ALIASES["image_id"])
    grade_col = find_column(df, COLUMN_ALIASES["dr_grade"])

    print(f"  image_id col : {id_col}")
    print(f"  dr_grade col : {grade_col}")

    # Drop ungradable images (adjudicated_gradable=0 have empty DR grades)
    if "adjudicated_gradable" in df.columns:
        n_before = len(df)
        df = df[df["adjudicated_gradable"] == 1].reset_index(drop=True)
        print(f"  Gradable only: {len(df)}/{n_before} (dropped {n_before - len(df)} ungradable)")

    df = df.dropna(subset=[grade_col])

    out = pd.DataFrame({
        "image_id": df[id_col].astype(str),
        "dr_grade": df[grade_col].astype(int),
    })

    # Sanity: grades should be 0-4 (or 0-3 for Messidor-1)
    grades = sorted(out["dr_grade"].unique())
    print(f"  Grade values : {grades}")
    assert max(grades) <= 4, f"Unexpected max grade {max(grades)}"

    # Cross-check: how many image_ids match files on disk
    on_disk = {p.stem for p in IMAGES_DIR.glob("*")
               if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    label_stems = out["image_id"].apply(lambda x: Path(str(x)).stem)
    matches = label_stems.isin(on_disk).sum()
    print(f"\nDisk match: {matches}/{len(out)} label rows matched to images")
    if matches < len(out) * 0.8:
        print("  WARNING: fewer than 80% matched — check image_id format")

    out.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT} ({len(out)} rows)")
    print(out["dr_grade"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
