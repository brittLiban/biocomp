"""
Task 6: Cox proportional hazards baseline — time to CST normalization.

Original spec called for time-to-first-injection from OCT-DR.xlsx, but that
data is degenerate (all 40 Prime patients injected at baseline; TREX is a
fixed injection protocol). Reframed to a non-degenerate clinical event:

  Event:    first visit where CST <= 300 um (treatment response / normalization)
  Censored: eyes that never reach CST <= 300, censored at last observed visit
  Time:     visit index (ordinal, 0 = baseline)

Result: 81/96 events, 15/96 censored. Time range 0-26 visits, mean 4.0.

Features (7 total — within 10 events/variable rule for 81 events):
  - PCA(5) of baseline-visit RETFound embedding (1024 -> 5 components)
  - Baseline BCVA
  - Baseline CST

Metric: C-index (concordance)
Library: lifelines
W&B run: cox_survival_seed42

Run:
    python scripts/baseline_cox.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from data.olives import build_sequences, split_by_eye

CST_THRESHOLD = 300   # um — clinical normalization threshold


def build_survival_df(sequences: dict) -> pd.DataFrame:
    """Build one row per eye: duration, event, and baseline features."""
    rows = []
    for eye_id, seq in sequences.items():
        cst = seq["cst"]
        bcva_0 = float(seq["bcva"][0])
        cst_0  = float(cst[0])
        emb_0  = seq["embeddings"][0]          # baseline visit embedding (1024-d)

        normalized = np.where(cst <= CST_THRESHOLD)[0]
        if len(normalized) > 0:
            duration = int(normalized[0])
            event    = 1
        else:
            duration = seq["n_visits"] - 1
            event    = 0

        rows.append({
            "eye_id":   eye_id,
            "duration": duration,
            "event":    event,
            "bcva_0":   bcva_0,
            "cst_0":    cst_0,
            "emb_0":    emb_0,
        })
    return pd.DataFrame(rows)


def main():
    np.random.seed(42)

    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=42)

    train_df = build_survival_df(train_seqs)
    test_df  = build_survival_df(test_seqs)

    print(f"Train: {len(train_df)} eyes  |  events: {train_df['event'].sum()}")
    print(f"Test:  {len(test_df)} eyes   |  events: {test_df['event'].sum()}")

    # PCA on baseline embeddings (fit on train only)
    emb_train = np.stack(train_df["emb_0"].values)
    emb_test  = np.stack(test_df["emb_0"].values)

    n_components = 5
    pca    = PCA(n_components=n_components, random_state=42)
    scaler = StandardScaler()

    emb_train_pca = pca.fit_transform(emb_train)
    emb_test_pca  = pca.transform(emb_test)

    var_explained = pca.explained_variance_ratio_.sum()
    print(f"\nPCA({n_components}) variance explained: {var_explained:.1%}")

    # Assemble feature DataFrames
    pca_cols = [f"pca_{i}" for i in range(n_components)]

    def make_features(df, pca_arr):
        feat = pd.DataFrame(pca_arr, columns=pca_cols, index=df.index)
        feat["bcva_0"] = df["bcva_0"].values
        feat["cst_0"]  = df["cst_0"].values
        feat["duration"] = df["duration"].values
        feat["event"]    = df["event"].values
        return feat

    train_feat = make_features(train_df, emb_train_pca)
    test_feat  = make_features(test_df,  emb_test_pca)

    # Scale clinical features (PCA outputs are already ~unit scale)
    for col in ["bcva_0", "cst_0"]:
        mu  = train_feat[col].mean()
        std = train_feat[col].std()
        train_feat[col] = (train_feat[col] - mu) / std
        test_feat[col]  = (test_feat[col]  - mu) / std

    # Fit Cox PH
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_feat, duration_col="duration", event_col="event",
            show_progress=False)

    print("\n--- Cox PH summary ---")
    cph.print_summary(columns=["coef", "exp(coef)", "p"], decimals=3)

    # C-index on test set
    risk_scores = cph.predict_partial_hazard(test_feat)
    c_index = concordance_index(
        test_feat["duration"], -risk_scores, test_feat["event"]
    )

    # Train C-index (overfit check)
    risk_train = cph.predict_partial_hazard(train_feat)
    c_train = concordance_index(
        train_feat["duration"], -risk_train, train_feat["event"]
    )

    print("\n=== Cox PH (time to CST normalization) ===")
    print(f"C-index (test) : {c_index:.4f}  (primary)")
    print(f"C-index (train): {c_train:.4f}  (overfit check)")
    print(f"Event rate: {test_df['event'].mean():.0%} test  |  "
          f"{train_df['event'].mean():.0%} train")
    print(f"\nNote: event = first visit with CST <= {CST_THRESHOLD} um. "
          f"Injection-timing data from OCT-DR.xlsx is degenerate "
          f"(all patients injected at baseline).")

    # W&B logging
    try:
        import wandb
        run = wandb.init(
            project="synapse-v1",
            name="cox_survival_seed42",
            config={
                "model": "CoxPH",
                "task": "time_to_cst_normalization",
                "cst_threshold_um": CST_THRESHOLD,
                "features": pca_cols + ["bcva_0", "cst_0"],
                "n_pca_components": n_components,
                "pca_var_explained": float(var_explained),
                "penalizer": 0.1,
                "seed": 42,
                "train_eyes": len(train_df),
                "test_eyes": len(test_df),
                "train_events": int(train_df["event"].sum()),
                "test_events": int(test_df["event"].sum()),
            },
        )
        wandb.log({
            "c_index_test":  c_index,
            "c_index_train": c_train,
        })
        wandb.finish()
        print(f"\nLogged to W&B: synapse-v1 / cox_survival_seed42")
    except Exception as e:
        print(f"\nW&B logging skipped: {e}")


if __name__ == "__main__":
    main()
