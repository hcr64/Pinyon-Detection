"""
semi_supervised.py
────────────────────────────────────────────────────────────────────────────────
Semi-supervised species classification for Pinyon-Detection.

Motivation
──────────
The supervised classifier in train_tree_classifier.py trains on ~166 labeled
clusters and ignores the remaining ~3,400 unlabeled ones. LabelSpreading
treats the unlabeled clusters not as waste but as structural information:
it builds a similarity graph across ALL clusters (labeled + unlabeled) and
propagates species labels from labeled nodes through the graph to their
geometric neighbours.

The key assumption is that clusters which are geometrically similar in feature
space (similar height, color, PCA shape) are more likely to be the same species.
This is ecologically reasonable — ponderosa clusters should form a coherent
region of feature space regardless of whether they were hand-labeled.

How it works
────────────
1. Combine labeled and unlabeled clusters into one feature matrix X_all
2. Assign y = species label for labeled rows, y = -1 for unlabeled rows
3. Fit LabelSpreading with an RBF kernel — builds a fully-connected
   similarity graph where edge weight = exp(-gamma * ||xi - xj||^2)
4. Labels flow from labeled nodes to unlabeled neighbours iteratively
5. The alpha parameter controls how much a node can deviate from its
   initial label (alpha=0 → labels are fixed; alpha=1 → fully propagated)
6. Read off predicted species + confidence for every cluster

Usage
─────
Call run_label_spreading() after train_tree_classifier() in main.py:

    from semi_supervised import run_label_spreading

    df_deep_clusters = run_label_spreading(
        df_deep_clusters,
        df_clusters,
        save_path=PATHS['Images']
    )

The function adds two columns to df_deep_clusters in-place:
    "semi_label"       — propagated species string for every cluster
    "semi_confidence"  — probability of the assigned label [0, 1]

Requirements
────────────
    scikit-learn (LabelSpreading is in sklearn.semi_supervised)
    No additional installs needed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score


# must match FEATURES in train_tree_classifier.py exactly
FEATURES = [
    "height", "radius", "n_points",
    "obb_extent_x", "obb_extent_y", "obb_extent_z",
    "eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
    "linearity", "planarity", "sphericity",
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "height_to_radius",
    "green_dominance",
    "crown_volume",
    "color_saturation",
    "verticality",
    "flatness_ratio",
    "crown_base_ratio",
]


def run_label_spreading(df_deep, df_labels_matched, save_path=None,
                        alpha=0.2, gamma=0.5, max_iter=100,
                        confidence_threshold=0.80):
    """
    Propagate species labels from labeled clusters to unlabeled ones using
    LabelSpreading, then evaluate on the labeled held-out subset.

    Parameters
    ──────────
    alpha : float in (0, 1)
        Clamping factor. At alpha=0 labeled nodes never move from their
        initial label (fully clamped). At alpha=1 they propagate freely
        (no clamping). 0.2 is a conservative starting point that keeps
        labeled nodes close to their true labels while still propagating.
        Increase toward 0.4 if unlabeled clusters are being assigned
        confidently wrong labels.

    gamma : float > 0
        RBF kernel bandwidth. Controls how far labels spread in feature
        space. Higher gamma = tighter neighbourhoods = labels spread less
        far. Lower gamma = broader spread. 0.5 works well when features
        are StandardScaler-normalised. Tune if results look oversmoothed
        (all clusters get pinyon) or undersmoothed (very few propagations).

    confidence_threshold : float in (0, 1)
        Minimum spreading confidence to report a cluster as classified.
        Clusters below this are left as "uncertain". Default 0.80.

    Args:
        df_deep (pd.DataFrame):
            Full feature dataframe from make_deep_dataframe() +
            engineer_features(). Must contain all columns in FEATURES.
            Includes BOTH labeled and unlabeled clusters.
        df_labels_matched (pd.DataFrame):
            Output of match_labels_to_clusters(). Must have "file" and
            "Name". Labeled rows have a species string; unlabeled rows
            have "unknown" or NaN.
        save_path (str | None):
            Directory to save diagnostic plots. Pass None to skip.
        alpha (float): Clamping factor. Default 0.2.
        gamma (float): RBF bandwidth. Default 0.5.
        max_iter (int): Maximum spreading iterations. Default 100.
        confidence_threshold (float): Minimum confidence to classify.
            Default 0.80.

    Returns:
        df_deep (pd.DataFrame): Input DataFrame with two new columns:
            "semi_label"      — propagated species or "uncertain"
            "semi_confidence" — spreading confidence score [0, 1]
    """

    print(f"\n{'═' * 60}")
    print("  SEMI-SUPERVISED LABEL SPREADING")
    print(f"{'═' * 60}")
    print(f"  alpha={alpha}  gamma={gamma}  max_iter={max_iter}  "
          f"confidence_threshold={confidence_threshold}")
    print()

    # ── only use features that actually exist ─────────────────────────────────
    available = [f for f in FEATURES if f in df_deep.columns]
    missing   = [f for f in FEATURES if f not in df_deep.columns]
    if missing:
        print(f"  ⚠  Missing features (skipped): {missing}")

    # ── build full feature matrix across all clusters ─────────────────────────
    # LabelSpreading sees every cluster — labeled and unlabeled alike.
    # The similarity graph it builds uses ALL of them.
    X_all = df_deep[available].values.copy()

    # ── scale features ────────────────────────────────────────────────────────
    # LabelSpreading's RBF kernel is distance-based so features must be on
    # comparable scales. StandardScaler makes this stable regardless of
    # whether height is in metres and eigenvalues are in mm².
    scaler = StandardScaler()
    X_all  = scaler.fit_transform(X_all)

    # ── build label vector ────────────────────────────────────────────────────
    # -1 = unlabeled (LabelSpreading convention)
    # Merge on "file" index so labeled rows get their species, others get -1.
    label_map = (
        df_labels_matched[["file", "Name"]]
        .set_index("file")["Name"]
        .to_dict()
    )

    species = ["juniper", "pinyon", "ponderosa"]   # fixed order for encoding

    # encode species → integer; unknown/missing → -1
    le = LabelEncoder()
    le.fit(species)

    y_all = np.full(len(df_deep), -1, dtype=int)

    for idx, row in df_deep.iterrows():
        name = label_map.get(row["file"], None)
        if name in species:
            y_all[idx] = le.transform([name])[0]

    n_labeled   = (y_all >= 0).sum()
    n_unlabeled = (y_all == -1).sum()
    print(f"  Labeled clusters:   {n_labeled}")
    print(f"  Unlabeled clusters: {n_unlabeled}")
    print(f"  Total:              {len(df_deep)}")
    print()

    labeled_counts = {sp: (y_all == le.transform([sp])[0]).sum() for sp in species}
    print("  Labeled class distribution:")
    for sp, cnt in labeled_counts.items():
        print(f"    {sp:<12} {cnt}")
    print()

    # ── fit LabelSpreading ────────────────────────────────────────────────────
    print("  Fitting LabelSpreading...")
    ls = LabelSpreading(
        kernel="rbf",
        gamma=gamma,
        alpha=alpha,
        max_iter=max_iter,
        n_jobs=-1,
    )
    ls.fit(X_all, y_all)
    print("  Done.\n")

    # ── extract predictions and confidences ───────────────────────────────────
    # label_distributions_ is shape (n_samples, n_classes) — a soft assignment
    # that sums to 1 per row. The predicted label is the argmax; confidence is
    # the max probability.
    proba      = ls.label_distributions_          # (N, n_classes)
    pred_enc   = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1)

    pred_labels = le.inverse_transform(pred_enc)

    # apply confidence threshold — below threshold → "uncertain"
    final_labels = np.where(
        confidence >= confidence_threshold,
        pred_labels,
        "uncertain"
    )

    df_deep["semi_label"]      = final_labels
    df_deep["semi_confidence"] = confidence

    # ── summary of propagated labels ──────────────────────────────────────────
    print("  Propagated label distribution (all clusters):")
    label_counts = pd.Series(final_labels).value_counts()
    for label, cnt in label_counts.items():
        pct = 100 * cnt / len(final_labels)
        print(f"    {label:<12} {cnt:>5}  ({pct:.1f}%)")
    print()

    # ── evaluate on labeled subset ────────────────────────────────────────────
    # Use the labeled rows as a proxy test set to sanity-check propagation.
    # This is not a clean held-out evaluation (the model saw these labels),
    # but it tells you whether the graph structure makes sense — if propagation
    # accuracy on labeled nodes is low, gamma or alpha need tuning.
    labeled_mask  = y_all >= 0
    y_true_labels = le.inverse_transform(y_all[labeled_mask])
    y_pred_labels = pred_labels[labeled_mask]

    macro_f1 = f1_score(y_true_labels, y_pred_labels,
                        average="macro", zero_division=0)
    kappa    = cohen_kappa_score(y_true_labels, y_pred_labels)

    print("  ── Labeled-node recovery (sanity check, not held-out eval) ──")
    print(f"  Note: model saw these labels during fitting — use as a")
    print(f"  consistency check only, not a true accuracy estimate.\n")
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
    print(f"  Macro F1:      {macro_f1:.3f}")
    print(f"  Cohen's Kappa: {kappa:.3f}")
    print()

    if macro_f1 < 0.7:
        print("  ⚠  Low labeled-node recovery — try lowering alpha (e.g. 0.1)")
        print("     or increasing gamma (e.g. 1.0) to tighten neighbourhoods.\n")

    # ── diagnostic plots ──────────────────────────────────────────────────────
    if save_path is not None:
        _save_confidence_histogram(confidence, final_labels,
                                   confidence_threshold, save_path)
        _save_propagation_scatter(df_deep, df_labels_matched,
                                  final_labels, confidence,
                                  confidence_threshold, save_path)

    return df_deep


# ── diagnostic helpers ────────────────────────────────────────────────────────

def _save_confidence_histogram(confidence, final_labels,
                                threshold, save_path):
    """
    Histogram of spreading confidence scores, split by predicted species.
    A good result has most clusters piled up near 1.0 with a clear gap
    below the threshold. A bad result has a flat distribution — the graph
    is not separating well and gamma/alpha need tuning.
    """
    os.makedirs(save_path, exist_ok=True)

    colors = {"pinyon": "#e05c5c", "juniper": "#5b8dd9",
              "ponderosa": "#f5a623", "uncertain": "#aaaaaa"}

    fig, ax = plt.subplots(figsize=(9, 4))

    species_order = ["pinyon", "juniper", "ponderosa", "uncertain"]
    for sp in species_order:
        mask = final_labels == sp
        if mask.sum() == 0:
            continue
        ax.hist(confidence[mask], bins=30, alpha=0.6,
                label=f"{sp} (n={mask.sum()})",
                color=colors.get(sp, "grey"))

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"threshold={threshold}")
    ax.set_xlabel("Spreading confidence")
    ax.set_ylabel("Cluster count")
    ax.set_title("LabelSpreading — confidence distribution by species")
    ax.legend(fontsize=8)
    plt.tight_layout()

    out = os.path.join(save_path, "semi_confidence_histogram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def _save_propagation_scatter(df_deep, df_labels_matched,
                               final_labels, confidence,
                               threshold, save_path):
    """
    Scatter of height vs radius coloured by semi_label, with labeled
    clusters marked by a black ring. Shows whether propagated labels
    form coherent regions in the most discriminative feature pair.
    """
    os.makedirs(save_path, exist_ok=True)

    colors = {"pinyon": "#e05c5c", "juniper": "#5b8dd9",
              "ponderosa": "#f5a623", "uncertain": "#cccccc"}

    labeled_files = set(
        df_labels_matched.loc[
            df_labels_matched["Name"].isin(["pinyon", "juniper", "ponderosa"]),
            "file"
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    for sp in ["pinyon", "juniper", "ponderosa", "uncertain"]:
        mask = np.array(final_labels) == sp
        if mask.sum() == 0:
            continue
        ax.scatter(
            df_deep.loc[mask, "height"],
            df_deep.loc[mask, "radius"],
            c=colors[sp],
            label=f"{sp} (n={mask.sum()})",
            alpha=0.4 if sp == "uncertain" else 0.6,
            s=8,
            edgecolors="none",
        )

    # overlay black rings on originally-labeled clusters
    labeled_mask = df_deep["file"].isin(labeled_files).values
    ax.scatter(
        df_deep.loc[labeled_mask, "height"],
        df_deep.loc[labeled_mask, "radius"],
        facecolors="none", edgecolors="black",
        s=25, linewidths=0.8, label="labeled (ground truth)", zorder=5,
    )

    ax.set_xlabel("height (m)")
    ax.set_ylabel("radius (m)")
    ax.set_title("LabelSpreading — propagated labels in height/radius space\n"
                 "(black rings = originally labeled clusters)")
    ax.legend(fontsize=8, markerscale=1.5)
    plt.tight_layout()

    out = os.path.join(save_path, "semi_propagation_scatter.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")