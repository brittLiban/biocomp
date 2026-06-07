"""
OLIVES temporal sequence dataset.

Aligns 78 K per-B-scan embeddings (encoded via RETFound) to visit order
using (Eye_ID, BCVA, CST) as the join key to Clinical_Data_Images.xlsx.
Averages all B-scan embeddings within a visit into one per-visit embedding.

Usage:
    sequences = build_sequences()          # first call ~30s, cached after
    train_ds, test_ds = split_by_eye(sequences, test_frac=0.2, seed=42)
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_EMBEDDINGS_DIR = Path("data/processed/embeddings")
_LABELS_PATH = Path(
    "data/raw/olives/labels/OLIVES_Dataset_Labels/"
    "ml_centric_labels/Clinical_Data_Images.xlsx"
)
_CACHE_PATH = Path("data/processed/olives_sequences_v2.pkl")


# ── Alignment ──────────────────────────────────────────────────────────────

def _visit_num(visit_key: str) -> int:
    """V3 → 3, W12 → 12. Used for within-eye chronological sort."""
    return int(visit_key[1:])


def build_sequences(force: bool = False) -> dict:
    """
    Returns:
        sequences: {eye_id (int): {
            'embeddings': np.ndarray (n_visits, 1024),
            'visit_keys': np.ndarray (n_visits,)  e.g. ['V1','V2',...]
            'visit_nums': np.ndarray (n_visits,)  e.g. [1, 2, ...]
            'sub_study':  str  'TREX' or 'Prime'
            'bcva':       np.ndarray (n_visits,)
            'cst':        np.ndarray (n_visits,)
            'labels':     np.ndarray (n_visits,)  binary disease label
            'n_scans':    np.ndarray (n_visits,)  B-scans averaged per visit
            'n_visits':   int
        }}

    Drops ~4.9% of B-scan rows where (Eye_ID, BCVA, CST) maps to >1 visit
    (BCVA/CST did not change between consecutive visits for that eye).
    """
    if _CACHE_PATH.exists() and not force:
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("Building OLIVES sequences (first run)...")

    # Load per-B-scan data
    embeddings   = np.load(_EMBEDDINGS_DIR / "olives_retfound.npy")
    eye_ids      = np.load(_EMBEDDINGS_DIR / "olives_eye_ids.npy",  allow_pickle=True).astype(float)
    disease_lbls = np.load(_EMBEDDINGS_DIR / "olives_labels.npy").astype(float)
    bcva_raw     = np.load(_EMBEDDINGS_DIR / "olives_bcva.npy",     allow_pickle=True)
    cst_raw      = np.load(_EMBEDDINGS_DIR / "olives_cst.npy",      allow_pickle=True)

    # Load label file — has File_Path (visit key) + BCVA + CST + Eye_ID
    df_labels = pd.read_excel(_LABELS_PATH)
    df_labels["visit_key"] = df_labels["File_Path"].apply(
        lambda p: m.group(1) if (m := re.search(r"/(V\d+|W\d+)/", str(p))) else None
    )
    df_labels = df_labels.dropna(subset=["visit_key"])

    # Build unambiguous (Eye_ID, BCVA, CST) → visit_key lookup
    n_visits_per_group = df_labels.groupby(["Eye_ID", "BCVA", "CST"])["visit_key"].nunique()
    clean_keys = set(n_visits_per_group[n_visits_per_group == 1].index)
    df_clean = df_labels[
        df_labels.apply(lambda r: (r.Eye_ID, r.BCVA, r.CST) in clean_keys, axis=1)
    ]
    lookup_df = (
        df_clean.groupby(["Eye_ID", "BCVA", "CST"])["visit_key"]
        .first()
        .reset_index()
        .rename(columns={"Eye_ID": "eye_id", "BCVA": "bcva", "CST": "cst"})
    )

    # Per-embedding DataFrame
    emb_df = pd.DataFrame({
        "emb_idx": np.arange(len(embeddings)),
        "eye_id":  eye_ids,
        "bcva":    pd.to_numeric(bcva_raw, errors="coerce"),
        "cst":     pd.to_numeric(cst_raw,  errors="coerce"),
        "label":   disease_lbls,
    }).dropna(subset=["bcva", "cst"])

    # Join to get visit_key per embedding
    merged = emb_df.merge(lookup_df, on=["eye_id", "bcva", "cst"], how="left")
    matched = merged.dropna(subset=["visit_key"]).copy()
    matched["visit_key"] = matched["visit_key"].astype(str)

    n_total  = len(embeddings)
    n_matched = len(matched)
    n_dropped = n_total - n_matched
    print(f"  Matched: {n_matched:,} / {n_total:,} B-scans  ({n_dropped:,} dropped - ambiguous BCVA/CST)")

    # Aggregate: average B-scan embeddings within each (eye_id, visit_key)
    sequences = {}
    for (eye_id, visit_key), group in matched.groupby(["eye_id", "visit_key"]):
        eye_id = int(eye_id)
        if eye_id not in sequences:
            sequences[eye_id] = []
        sequences[eye_id].append({
            "visit_key": visit_key,
            "visit_num": _visit_num(visit_key),
            "sub_study": "TREX" if visit_key.startswith("V") else "Prime",
            "embedding": embeddings[group["emb_idx"].values].mean(axis=0),
            "bcva":      float(group["bcva"].iloc[0]),
            "cst":       float(group["cst"].iloc[0]),
            "label":     int(group["label"].mode().iloc[0]),
            "n_scans":   len(group),
        })

    # Sort each eye chronologically and pack into arrays
    result = {}
    for eye_id, visits in sequences.items():
        visits.sort(key=lambda v: v["visit_num"])
        sub_study = visits[0]["sub_study"]
        vnums = np.array([v["visit_num"] for v in visits], dtype=np.float32)
        # week_gaps: real week intervals between consecutive visits (length n_visits-1).
        # Prime: visit_nums ARE week numbers (W0=0, W4=4, ...). Normalize by 4 so
        #        1.0 = one 4-week interval, consistent with the TREX ordinal scale.
        # TREX: real timing unavailable (treat-and-extend protocol, OCT-DME Week
        #        columns ~98% empty). Use ordinal 1.0 per step.
        if sub_study == "Prime":
            week_gaps = np.diff(vnums) / 4.0
        else:
            week_gaps = np.ones(len(visits) - 1, dtype=np.float32)
        result[eye_id] = {
            "embeddings": np.stack([v["embedding"] for v in visits]),
            "visit_keys": np.array([v["visit_key"] for v in visits]),
            "visit_nums": vnums.astype(np.int32),
            "sub_study":  sub_study,
            "bcva":       np.array([v["bcva"] for v in visits], dtype=np.float32),
            "cst":        np.array([v["cst"]  for v in visits], dtype=np.float32),
            "labels":     np.array([v["label"] for v in visits], dtype=np.int64),
            "n_scans":    np.array([v["n_scans"] for v in visits], dtype=np.int32),
            "n_visits":   len(visits),
            "week_gaps":  week_gaps,
        }

    n_visits_list = [v["n_visits"] for v in result.values()]
    print(f"  Eyes: {len(result)}  |  visits per eye: "
          f"min={min(n_visits_list)}  max={max(n_visits_list)}  "
          f"mean={np.mean(n_visits_list):.1f}")

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"  Cached -> {_CACHE_PATH}")
    return result


# ── Train / test split ─────────────────────────────────────────────────────

def split_by_eye(
    sequences: dict,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Split sequence dict into train / test by Eye_ID. Never mixes scans from
    the same eye across splits."""
    rng = np.random.default_rng(seed)
    eye_ids = np.array(sorted(sequences.keys()))
    rng.shuffle(eye_ids)
    n_test = max(1, round(len(eye_ids) * test_frac))
    test_ids  = set(eye_ids[:n_test])
    train_ids = set(eye_ids[n_test:])
    return (
        {k: sequences[k] for k in train_ids},
        {k: sequences[k] for k in test_ids},
    )


# ── PyTorch Dataset ────────────────────────────────────────────────────────

class OLIVESSequenceDataset(Dataset):
    """
    Each item is one eye's full visit sequence.

    Returns:
        embeddings: FloatTensor  (n_visits, 1024)
        labels:     LongTensor   (n_visits,)
        bcva:       FloatTensor  (n_visits,)
        cst:        FloatTensor  (n_visits,)
        visit_nums: LongTensor   (n_visits,)
        eye_id:     int
    """

    def __init__(self, sequences: dict):
        self.eye_ids = sorted(sequences.keys())
        self.sequences = sequences

    def __len__(self):
        return len(self.eye_ids)

    def __getitem__(self, idx):
        eye_id = self.eye_ids[idx]
        seq = self.sequences[eye_id]
        return {
            "embeddings": torch.from_numpy(seq["embeddings"]).float(),
            "labels":     torch.from_numpy(seq["labels"]).long(),
            "bcva":       torch.from_numpy(seq["bcva"]).float(),
            "cst":        torch.from_numpy(seq["cst"]).float(),
            "visit_nums": torch.from_numpy(seq["visit_nums"]).long(),
            "eye_id":     eye_id,
        }

    def __repr__(self):
        n = len(self)
        total_visits = sum(self.sequences[e]["n_visits"] for e in self.eye_ids)
        return f"OLIVESSequenceDataset(eyes={n}, total_visits={total_visits})"


def collate_sequences(batch: list[dict]) -> dict:
    """Pad variable-length sequences to the longest in the batch."""
    max_len = max(item["embeddings"].shape[0] for item in batch)
    emb_dim = batch[0]["embeddings"].shape[1]

    padded_emb   = torch.zeros(len(batch), max_len, emb_dim)
    padded_lbl   = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_bcva  = torch.zeros(len(batch), max_len)
    padded_cst   = torch.zeros(len(batch), max_len)
    padded_vnum  = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths      = torch.zeros(len(batch), dtype=torch.long)
    eye_ids      = []

    for i, item in enumerate(batch):
        n = item["embeddings"].shape[0]
        padded_emb[i, :n]  = item["embeddings"]
        padded_lbl[i, :n]  = item["labels"]
        padded_bcva[i, :n] = item["bcva"]
        padded_cst[i, :n]  = item["cst"]
        padded_vnum[i, :n] = item["visit_nums"]
        lengths[i]         = n
        eye_ids.append(item["eye_id"])

    return {
        "embeddings": padded_emb,
        "labels":     padded_lbl,
        "bcva":       padded_bcva,
        "cst":        padded_cst,
        "visit_nums": padded_vnum,
        "lengths":    lengths,
        "eye_ids":    eye_ids,
    }
