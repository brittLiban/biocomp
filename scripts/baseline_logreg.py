"""
Task 3: Logistic regression baseline.

Single-visit prediction: each visit embedding (1024-d) -> disease label (0/1).
This is a floor baseline — no temporal information, no sequence modeling.
Train/test split is by Eye_ID to prevent leakage.

Run:
    python scripts/baseline_logreg.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from data.olives import build_sequences, split_by_eye


def flatten_sequences(sequences: dict) -> tuple[np.ndarray, np.ndarray]:
    """Unroll per-eye visit sequences into flat (n_visits, 1024) + labels."""
    X, y = [], []
    for seq in sequences.values():
        X.append(seq["embeddings"])
        y.append(seq["labels"])
    return np.vstack(X), np.concatenate(y)


def main():
    print("Loading sequences...")
    seqs = build_sequences()
    train_seqs, test_seqs = split_by_eye(seqs, test_frac=0.2, seed=42)

    X_train, y_train = flatten_sequences(train_seqs)
    X_test,  y_test  = flatten_sequences(test_seqs)

    print(f"Train: {X_train.shape[0]} visits ({y_train.mean():.1%} positive)")
    print(f"Test:  {X_test.shape[0]} visits  ({y_test.mean():.1%} positive)")

    # Scale (logistic regression is sensitive to feature scale)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Fit — l2 penalty, saga solver handles high-dim well
    print("\nFitting logistic regression...")
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_prob  = clf.predict_proba(X_test_s)[:, 1]
    y_pred  = clf.predict(X_test_s)

    auc      = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Logistic Regression (single-visit, no temporal info) ===")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No disease", "Disease"]))

    print("Train AUC (overfit check):", round(
        roc_auc_score(y_train, clf.predict_proba(X_train_s)[:, 1]), 4
    ))


if __name__ == "__main__":
    main()
