"""
Messidor external validation.

Pipeline:
  1. Load EyePACS embeddings + labels → train logistic regression (binary: grade >= 2)
  2. Load Messidor embeddings + filenames → join to messidor_labels.csv
  3. Evaluate on Messidor: AUC-ROC, accuracy, sensitivity, specificity
  4. Log to W&B as run messidor_val_v1

Usage:
  python scripts/validate_messidor.py --labels data/processed/messidor_labels.csv

messidor_labels.csv expected columns: image_id (stem, no ext), dr_grade (0-4 or 0-3)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, roc_curve)
import wandb

EMBED_DIR = Path("C:/Users/liban/Documents/biocomp/data/processed/embeddings")

REFERABLE_THRESHOLD = 2  # grade >= 2 → referable DR


def load_eyepacs():
    emb = np.load(EMBED_DIR / "eyepacs_retfound.npy")
    lbl = np.load(EMBED_DIR / "eyepacs_labels.npy")
    y   = (lbl >= REFERABLE_THRESHOLD).astype(int)
    print(f"EyePACS: {emb.shape}, referable={y.mean():.3f}")
    return emb, y


def load_messidor(labels_csv: Path):
    emb = np.load(EMBED_DIR / "messidor_retfound.npy")
    fn  = np.load(EMBED_DIR / "messidor_filenames.npy")

    df  = pd.read_csv(labels_csv)
    # Normalise: image_id column may have extension; strip to stem
    df["stem"] = df["image_id"].apply(lambda x: Path(str(x)).stem)

    stem_to_grade = dict(zip(df["stem"], df["dr_grade"]))

    matched_emb, matched_y, matched_ids = [], [], []
    for i, stem in enumerate(fn):
        if stem in stem_to_grade:
            matched_emb.append(emb[i])
            matched_y.append(int(stem_to_grade[stem] >= REFERABLE_THRESHOLD))
            matched_ids.append(stem)

    emb_arr = np.stack(matched_emb)
    y_arr   = np.array(matched_y)

    n_total = len(fn)
    n_match = len(matched_ids)
    print(f"Messidor: {n_match}/{n_total} images matched to labels")
    assert n_match > n_total * 0.8, (
        f"Only {n_match}/{n_total} matched — check join key / label source"
    )
    print(f"  referable={y_arr.mean():.3f}")
    return emb_arr, y_arr


def train_logreg(X_train, y_train):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_sc, y_train)
    return clf, scaler


def evaluate(clf, scaler, X_test, y_test):
    X_sc    = scaler.transform(X_test)
    probs   = clf.predict_proba(X_sc)[:, 1]
    preds   = clf.predict(X_sc)
    auc     = roc_auc_score(y_test, probs)
    acc     = accuracy_score(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    sens    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"auc": auc, "accuracy": acc, "sensitivity": sens, "specificity": spec,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "probs": probs, "y_true": y_test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True,
                        help="Path to messidor_labels.csv (image_id, dr_grade)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    assert labels_path.exists(), f"Labels file not found: {labels_path}"

    # Load data
    X_train, y_train = load_eyepacs()
    X_test,  y_test  = load_messidor(labels_path)

    # Train
    print("\nTraining logistic regression on EyePACS...")
    clf, scaler = train_logreg(X_train, y_train)

    # Evaluate
    print("Evaluating on Messidor...")
    metrics = evaluate(clf, scaler, X_test, y_test)

    print(f"\n{'='*40}")
    print(f"  AUC-ROC:     {metrics['auc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print(f"{'='*40}\n")

    if not args.no_wandb:
        run = wandb.init(project="synapse", name="messidor_val_v1",
                         config={"threshold": REFERABLE_THRESHOLD,
                                 "train_set": "eyepacs",
                                 "test_set": "messidor2",
                                 "model": "RETFound_MAE+LogReg"})
        wandb.log({k: v for k, v in metrics.items()
                   if k not in ("probs", "y_true")})

        # ROC curve
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["probs"])
        roc_data = [[x, y] for x, y in zip(fpr.tolist(), tpr.tolist())]
        table = wandb.Table(data=roc_data, columns=["fpr", "tpr"])
        wandb.log({"roc_curve": wandb.plot.line(table, "fpr", "tpr",
                   title=f"ROC (AUC={metrics['auc']:.3f})")})
        run.finish()
        print("Logged to W&B: messidor_val_v1")


if __name__ == "__main__":
    main()
