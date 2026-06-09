"""
Generate all three paper figures.

Figure 1: ROC curves — OLIVES within-distribution (DME vs healthy) +
          Messidor-2 OOD (referable DR), both from frozen RETFound embeddings.
Figure 2: Bar chart — ordinal vs real-Δt RMSE for all three temporal models.
          Shows the divergence pattern: ODE improves, recurrents degrade.
Figure 3: ODE-RNN architecture diagram.

Usage:
    python scripts/make_figures.py --labels data/processed/messidor_labels.csv

Outputs (all to results/figures/):
    figure1_roc_curves.pdf  (+ .png)
    figure2_timing_bar.pdf  (+ .png)
    figure3_architecture.pdf (+ .png)
"""

import sys
sys.path.insert(0, "src")

import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

EMBED_DIR = pathlib.Path("data/processed/embeddings")
FIG_DIR   = pathlib.Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Published results (from W&B runs, see RUNS.md)
TABLE1 = {
    "models":        ["GRU-D",  "T-LSTM", "Latent ODE"],
    "ordinal_rmse":  [82.2,      82.0,      81.96],
    "realdt_rmse":   [84.2,      85.0,      81.6],
}
PERSISTENCE_RMSE = 91.7


# ── Figure 1: ROC curves ───────────────────────────────────────────────────

def make_figure1(labels_csv: pathlib.Path):
    print("Figure 1: ROC curves...")

    # OLIVES classification (DME vs healthy, within-distribution)
    olives_emb = np.load(EMBED_DIR / "olives_retfound.npy")
    olives_lbl = np.load(EMBED_DIR / "olives_labels.npy")   # 0=healthy, 1=DME

    scaler_ol = StandardScaler()
    X_ol = scaler_ol.fit_transform(olives_emb)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    probs_ol = cross_val_predict(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        X_ol, olives_lbl, cv=5, method="predict_proba")[:, 1]
    auc_ol = roc_auc_score(olives_lbl, probs_ol)
    fpr_ol, tpr_ol, _ = roc_curve(olives_lbl, probs_ol)

    # Messidor-2 OOD
    eyepacs_emb = np.load(EMBED_DIR / "eyepacs_retfound.npy")
    eyepacs_lbl = (np.load(EMBED_DIR / "eyepacs_labels.npy") >= 2).astype(int)
    mess_emb    = np.load(EMBED_DIR / "messidor_retfound.npy")
    mess_fn     = np.load(EMBED_DIR / "messidor_filenames.npy")

    import pandas as pd
    df = pd.read_csv(labels_csv)
    df["stem"] = df["image_id"].apply(lambda x: pathlib.Path(str(x)).stem)
    stem_to_grade = dict(zip(df["stem"], df["dr_grade"]))
    matched_emb, matched_y = [], []
    for i, stem in enumerate(mess_fn):
        if stem in stem_to_grade:
            matched_emb.append(mess_emb[i])
            matched_y.append(int(stem_to_grade[stem] >= 2))
    mess_emb_arr = np.stack(matched_emb)
    mess_y_arr   = np.array(matched_y)

    scaler_ep = StandardScaler()
    X_ep = scaler_ep.fit_transform(eyepacs_emb)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_ep, eyepacs_lbl)
    probs_mess = clf.predict_proba(scaler_ep.transform(mess_emb_arr))[:, 1]
    auc_mess = roc_auc_score(mess_y_arr, probs_mess)
    fpr_mess, tpr_mess, _ = roc_curve(mess_y_arr, probs_mess)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr_ol,   tpr_ol,   color="#2166ac", lw=2,
            label=f"OLIVES within-dist. (AUC = {auc_ol:.3f})")
    ax.plot(fpr_mess, tpr_mess, color="#d6604d", lw=2,
            label=f"Messidor-2 OOD (AUC = {auc_mess:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Frozen RETFound Representation", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"figure1_roc_curves.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved {p}")
    plt.close(fig)


# ── Figure 2: Timing bar chart ─────────────────────────────────────────────

def make_figure2():
    print("Figure 2: Ordinal vs real-dt bar chart...")

    models       = TABLE1["models"]
    ordinal_rmse = TABLE1["ordinal_rmse"]
    realdt_rmse  = TABLE1["realdt_rmse"]

    x      = np.arange(len(models))
    width  = 0.35
    colors = {"ord": "#4393c3", "rdt": "#d6604d"}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bars_ord = ax.bar(x - width/2, ordinal_rmse, width, label="Ordinal time",
                      color=colors["ord"], edgecolor="white", linewidth=0.5)
    bars_rdt = ax.bar(x + width/2, realdt_rmse,  width, label="Real Δt",
                      color=colors["rdt"], edgecolor="white", linewidth=0.5)

    # Annotate each bar with its value
    for bar in (*bars_ord, *bars_rdt):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    # Annotate direction arrows: show Δ for each model
    for i, (m, o, r) in enumerate(zip(models, ordinal_rmse, realdt_rmse)):
        delta = r - o
        color = "#b2182b" if delta > 0 else "#2166ac"
        sign  = "+" if delta >= 0 else ""
        ax.text(x[i], max(o, r) + 1.2, f"{sign}{delta:.1f}",
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Next-visit CST RMSE (μm)", fontsize=11)
    ax.set_title("Effect of Real Inter-Visit Timing on Model Performance\n"
                 "(19 held-out eyes; annotated Δ = real-Δt minus ordinal)",
                 fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(79, 89)
    ax.spines[["top", "right"]].set_visible(False)

    note = ("ODE benefits from real timing (−0.4 μm);\n"
            "recurrent models degrade (+2.0 to +3.0 μm).")
    ax.text(0.98, 0.04, note, transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.7))

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"figure2_timing_bar.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved {p}")
    plt.close(fig)


# ── Figure 3: ODE-RNN architecture diagram ────────────────────────────────

def make_figure3():
    print("Figure 3: Architecture diagram...")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel="", color="#cce5ff"):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.05", linewidth=1.2,
            edgecolor="#336699", facecolor=color)
        ax.add_patch(rect)
        ax.text(cx, cy + (0.12 if sublabel else 0), label,
                ha="center", va="center", fontsize=9, fontweight="bold")
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel,
                    ha="center", va="center", fontsize=7.5, color="#555")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#336699", lw=1.3))

    # Row 1: input pipeline
    box(1.0, 2.0, 1.6, 0.8, "OCT/Fundus", "visit t", color="#e8f4f8")
    box(2.9, 2.0, 1.6, 0.8, "RETFound", "frozen, 1024-d", color="#e8f4f8")
    box(4.8, 2.0, 1.6, 0.8, "Linear enc.", "1024 → 32", color="#ddeeff")
    box(6.7, 2.0, 1.6, 0.8, "GRU update", "obs. assimilation", color="#ddeeff")
    box(8.6, 2.0, 1.6, 0.8, "Decoder", "32 → CST_t+1", color="#e8f4f8")

    arrow(1.8, 2.0, 2.1, 2.0)
    arrow(3.7, 2.0, 4.0, 2.0)
    arrow(5.6, 2.0, 5.9, 2.0)
    arrow(7.5, 2.0, 7.8, 2.0)

    # ODE integration loop below GRU
    box(6.7, 0.8, 1.6, 0.7, "Neural ODE", "dopri5, exact per-example dt", color="#fff3cd")
    ax.annotate("", xy=(6.7, 1.15), xytext=(6.7, 1.6),
                arrowprops=dict(arrowstyle="->", color="#c07000", lw=1.3))
    ax.annotate("", xy=(7.5, 0.8), xytext=(8.0, 0.8),
                arrowprops=dict(arrowstyle="->", color="#c07000", lw=1.3))
    ax.annotate("", xy=(8.6, 1.6), xytext=(8.3, 0.8),
                arrowprops=dict(arrowstyle="->", color="#c07000", lw=1.3))

    ax.text(7.85, 0.22, "real inter-visit gap", fontsize=7.5,
            ha="center", color="#c07000", style="italic")

    # Recurrent loop arrow (top)
    ax.annotate("", xy=(5.9, 2.4), xytext=(7.5, 2.4),
                arrowprops=dict(arrowstyle="<-", color="#336699",
                                lw=1.0, connectionstyle="arc3,rad=-0.3"))
    ax.text(6.7, 3.05, "h_t carries to next visit", fontsize=7.5,
            ha="center", color="#336699", style="italic")

    ax.set_title("ODE-RNN Architecture (Rubanova et al. 2019)\n"
                 "GRU assimilates each observation; ODE integrates over exact biological interval",
                 fontsize=9.5, pad=8)

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"figure3_architecture.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=None,
                        help="Path to messidor_labels.csv (required for Figure 1)")
    parser.add_argument("--skip-fig1", action="store_true",
                        help="Skip Figure 1 (ROC curves) — useful if OLIVES embeddings unavailable")
    args = parser.parse_args()

    if not args.skip_fig1:
        if args.labels is None:
            print("WARNING: --labels not provided; skipping Figure 1 (ROC curves).")
            print("  Run with: --labels data/processed/messidor_labels.csv")
        else:
            try:
                make_figure1(pathlib.Path(args.labels))
            except FileNotFoundError as e:
                print(f"WARNING: Figure 1 skipped — {e}")

    make_figure2()
    make_figure3()
    print("\nAll figures written to results/figures/")


if __name__ == "__main__":
    main()
