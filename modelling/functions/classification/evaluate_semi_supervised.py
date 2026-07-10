"""
evaluate_semi_supervised.py
────────────────────────────────────────────────────────────────────────────────
Honest held-out evaluation for the LabelSpreading classifier in
semi_supervised.py.

Why this file exists
─────────────────────
run_label_spreading()'s built-in "sanity check" fits LabelSpreading with
alpha=0.2, which clamps labeled nodes close to their true label, then checks
whether those same labeled nodes were recovered correctly. That is close to
tautological — it mostly measures how strongly alpha clamps, not whether
propagation to the ~3,400 *unlabeled* clusters can be trusted. The function's
own docstring already warns about this ("not a clean held-out evaluation").

This module fixes that by masking each labeled cluster's label before fitting
(so the model genuinely never sees it), evaluating in stratified folds so
every labeled cluster gets one turn as a true held-out point, and comparing
against the supervised RF baseline on the *same* folds so the comparison is
apples-to-apples.

Four pieces:
    1. evaluate_label_spreading_holdout()  — the honest recovery metric
    2. compare_to_supervised_baseline()     — RF vs LabelSpreading, same folds
    3. run_degradation_curve()              — does LS degrade more gracefully
                                               than RF as labeled data shrinks?
    4. plot_reliability_diagram()           — is semi_confidence calibrated?

Usage
─────
    from evaluate_semi_supervised import run_full_evaluation

    run_full_evaluation(
        df_deep_clusters,
        df_clusters,
        save_path=PATHS['Images'],
        alpha=0.2,
        gamma=0.5,
    )

Requirements
────────────
    numpy, pandas, matplotlib, scikit-learn (already in the venv)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score

from functions.classification.semi_supervised import FEATURES as DEFAULT_FEATURES

SPECIES = ["juniper", "pinyon", "ponderosa"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _prep(df_deep, df_labels_matched, features=None):
    """
    Build the scaled feature matrix, integer label vector, and the boolean
    mask of which rows are actually labeled. Shared setup for every function
    below so folds line up against the same row ordering.

    Returns:
        X (np.ndarray): (N, F) scaled feature matrix, all clusters.
        y_enc (np.ndarray): (N,) int labels, -1 for unlabeled rows.
        labeled_idx (np.ndarray): row indices (into df_deep) that are labeled.
        le (LabelEncoder): fitted on SPECIES, for decoding predictions.
        available (list[str]): feature columns actually present and used.
    """
    available = [f for f in (features or DEFAULT_FEATURES) if f in df_deep.columns]
    missing   = [f for f in (features or DEFAULT_FEATURES) if f not in df_deep.columns]
    if missing:
        print(f"  ⚠  Missing features (skipped): {missing}")

    X = StandardScaler().fit_transform(df_deep[available].values)

    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file", how="left")
    label_map = df.set_index(df_deep.index)["Name"]

    le = LabelEncoder().fit(SPECIES)
    y_enc = np.full(len(df_deep), -1, dtype=int)
    labeled_mask = label_map.isin(SPECIES).values
    y_enc[labeled_mask] = le.transform(label_map[labeled_mask].values)

    labeled_idx = np.where(labeled_mask)[0]

    return X, y_enc, labeled_idx, le, available


# ── 1. honest held-out evaluation ─────────────────────────────────────────────

def evaluate_label_spreading_holdout(df_deep, df_labels_matched, features=None,
                                     alpha=0.2, gamma=0.5, max_iter=100,
                                     n_splits=5, random_state=42, verbose=True):
    """
    Evaluate LabelSpreading by genuinely masking each labeled cluster's label
    before fitting, rather than trusting alpha-clamped recovery.

    For each of n_splits stratified folds, the fold's labeled clusters are
    set to -1 (unlabeled) before fit(); LabelSpreading has to recover their
    species purely from graph structure built on the *other* labeled clusters
    plus every unlabeled one. Every labeled cluster gets exactly one turn as
    a true held-out point across the n_splits folds.

    This is the metric to trust over run_label_spreading()'s internal
    "sanity check" — that one clamps labeled nodes toward their true label at
    fit time (via alpha) and is barely different from checking whether alpha
    did its job.

    Args:
        df_deep (pd.DataFrame): Output of make_deep_dataframe() +
            engineer_features(). Must include all columns in `features`.
        df_labels_matched (pd.DataFrame): Output of match_labels_to_clusters().
            Must have "file" and "Name".
        features (list[str] | None): Feature columns to use. Defaults to
            semi_supervised.FEATURES.
        alpha (float): LabelSpreading clamping factor. Default 0.2.
        gamma (float): RBF kernel bandwidth. Default 0.5.
        max_iter (int): Max spreading iterations. Default 100.
        n_splits (int): Number of stratified folds over the labeled subset.
            Default 5.
        random_state (int): Fold shuffling seed. Default 42.
        verbose (bool): Print classification report and metrics. Default True.

    Returns:
        dict: {
            "y_true": np.ndarray of true species strings,
            "y_pred": np.ndarray of held-out predicted species strings,
            "confidence": np.ndarray of spreading confidence at prediction time,
            "macro_f1": float,
            "kappa": float,
        }

    Requirements:
        numpy, pandas, scikit-learn
    """
    X, y_enc, labeled_idx, le, available = _prep(df_deep, df_labels_matched, features)
    y_true_enc = y_enc[labeled_idx]

    preds       = np.full(len(labeled_idx), -1, dtype=int)
    confidences = np.full(len(labeled_idx), np.nan)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_pos, test_pos in skf.split(labeled_idx, y_true_enc):
        y_fold = y_enc.copy()
        # mask this fold's labeled rows — the model must never see them
        y_fold[labeled_idx[test_pos]] = -1

        ls = LabelSpreading(kernel="rbf", gamma=gamma, alpha=alpha,
                            max_iter=max_iter, n_jobs=-1)
        ls.fit(X, y_fold)

        proba = ls.label_distributions_[labeled_idx[test_pos]]
        preds[test_pos]       = np.argmax(proba, axis=1)
        confidences[test_pos] = proba.max(axis=1)

    y_true = le.inverse_transform(y_true_enc)
    y_pred = le.inverse_transform(preds)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  LABELSPREADING — HONEST HELD-OUT EVALUATION")
        print(f"  ({n_splits}-fold, labels masked at fit time)")
        print(f"{'═' * 60}")
        print(classification_report(y_true, y_pred, zero_division=0))
        print(f"  Macro F1:      {macro_f1:.3f}")
        print(f"  Cohen's Kappa: {kappa:.3f}")
        print()

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidences,
        "macro_f1": macro_f1,
        "kappa": kappa,
    }


# ── 2. supervised baseline on the same folds ──────────────────────────────────

def compare_to_supervised_baseline(df_deep, df_labels_matched, features=None,
                                   alpha=0.2, gamma=0.5, max_iter=100,
                                   n_splits=5, random_state=42):
    """
    Run LabelSpreading and a plain supervised RF on identical stratified
    folds over the labeled subset, so the comparison isolates the effect of
    using the unlabeled clusters — not a difference in what data each model
    happened to see.

    RF only ever sees the labeled training fold (no unlabeled clusters at
    all). LabelSpreading sees the same labeled training fold PLUS every
    unlabeled cluster, with the test fold masked to -1. If LabelSpreading
    doesn't beat RF here, the unlabeled data isn't earning its keep.

    Args:
        df_deep, df_labels_matched, features: same as
            evaluate_label_spreading_holdout().
        alpha, gamma, max_iter: LabelSpreading params, same meaning as above.
        n_splits (int): Number of folds. Default 5.
        random_state (int): Shared fold seed — same folds used for both
            models. Default 42.

    Returns:
        pd.DataFrame: One row per model ("RF_baseline", "LabelSpreading")
            with columns "macro_f1" and "kappa", sorted best first.

    Requirements:
        numpy, pandas, scikit-learn
    """
    X, y_enc, labeled_idx, le, available = _prep(df_deep, df_labels_matched, features)
    y_true_enc = y_enc[labeled_idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rf_preds = np.full(len(labeled_idx), -1, dtype=int)
    ls_preds = np.full(len(labeled_idx), -1, dtype=int)

    for train_pos, test_pos in skf.split(labeled_idx, y_true_enc):
        train_rows = labeled_idx[train_pos]
        test_rows  = labeled_idx[test_pos]

        # ── RF: only ever sees the labeled training rows ──────────────────
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=42)
        rf.fit(X[train_rows], y_enc[train_rows])
        rf_preds[test_pos] = rf.predict(X[test_rows])

        # ── LabelSpreading: sees train rows + all unlabeled, test masked ──
        y_fold = y_enc.copy()
        y_fold[test_rows] = -1
        ls = LabelSpreading(kernel="rbf", gamma=gamma, alpha=alpha,
                            max_iter=max_iter, n_jobs=-1)
        ls.fit(X, y_fold)
        ls_preds[test_pos] = np.argmax(ls.label_distributions_[test_rows], axis=1)

    y_true = le.inverse_transform(y_true_enc)

    results = []
    for name, preds in [("RF_baseline", rf_preds), ("LabelSpreading", ls_preds)]:
        y_pred = le.inverse_transform(preds)
        results.append({
            "model":    name,
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "kappa":    round(cohen_kappa_score(y_true, y_pred), 4),
        })

    df_results = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    print(f"\n{'═' * 60}")
    print(f"  RF BASELINE  vs  LABELSPREADING  ({n_splits}-fold, same folds)")
    print(f"{'═' * 60}")
    print(df_results.to_string(index=False))
    if df_results.iloc[0]["model"] == "RF_baseline":
        print("\n  ⚠  RF alone beats LabelSpreading — the unlabeled clusters "
              "are not adding useful signal at these hyperparameters.")
    print()

    return df_results


# ── 3. degradation curve ──────────────────────────────────────────────────────

def run_degradation_curve(df_deep, df_labels_matched, features=None,
                          alpha=0.2, gamma=0.5, max_iter=100,
                          fractions=(1.0, 0.75, 0.5, 0.25, 0.1),
                          n_splits=5, random_state=42, save_path=None):
    """
    Test whether LabelSpreading degrades more gracefully than supervised RF
    as labeled data shrinks — the actual claim behind using it (that the
    ~166-label / ~20-ponderosa bottleneck can be offset by unlabeled data).

    For each fraction, only that fraction of each fold's *training* labels
    is kept (stratified per class so rare species aren't wiped out first);
    the held-out test fold is untouched. Both RF and LabelSpreading are
    evaluated at each fraction on the same reduced training set, using the
    same folds as compare_to_supervised_baseline().

    If the two curves stay close together, or LabelSpreading doesn't pull
    ahead as fractions shrink, the unlabeled graph isn't compensating for
    scarce labels the way the motivation assumes.

    Args:
        df_deep, df_labels_matched, features: same as above.
        alpha, gamma, max_iter: LabelSpreading params.
        fractions (tuple of float): Fractions of training labels to retain
            per fold, largest first. Default (1.0, 0.75, 0.5, 0.25, 0.1).
        n_splits (int): Number of folds. Default 5.
        random_state (int): Shared seed for folds and subsampling.
            Default 42.
        save_path (str | None): Directory to save the comparison plot into.
            Pass None to skip saving. Default None.

    Returns:
        pd.DataFrame: One row per (fraction, model) with macro_f1 and kappa.

    Requirements:
        numpy, pandas, matplotlib, scikit-learn
    """
    X, y_enc, labeled_idx, le, available = _prep(df_deep, df_labels_matched, features)
    y_true_enc = y_enc[labeled_idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(labeled_idx, y_true_enc))

    rng = np.random.default_rng(random_state)
    rows = []

    for frac in fractions:
        rf_preds = np.full(len(labeled_idx), -1, dtype=int)
        ls_preds = np.full(len(labeled_idx), -1, dtype=int)

        for train_pos, test_pos in folds:
            train_rows = labeled_idx[train_pos]
            test_rows  = labeled_idx[test_pos]

            # stratified subsample of the training rows for this fraction —
            # keep at least 1 per class present so rare species (ponderosa)
            # aren't the first thing dropped as frac shrinks
            train_labels = y_enc[train_rows]
            keep = []
            for cls in np.unique(train_labels):
                cls_rows = train_rows[train_labels == cls]
                n_keep   = max(1, int(round(len(cls_rows) * frac)))
                keep.extend(rng.choice(cls_rows, size=n_keep, replace=False))
            keep = np.array(keep)

            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=42)
            rf.fit(X[keep], y_enc[keep])
            rf_preds[test_pos] = rf.predict(X[test_rows])

            y_fold = np.full(len(y_enc), -1, dtype=int)
            y_fold[keep] = y_enc[keep]
            ls = LabelSpreading(kernel="rbf", gamma=gamma, alpha=alpha,
                                max_iter=max_iter, n_jobs=-1)
            ls.fit(X, y_fold)
            ls_preds[test_pos] = np.argmax(ls.label_distributions_[test_rows], axis=1)

        y_true = le.inverse_transform(y_true_enc)
        for name, preds in [("RF_baseline", rf_preds), ("LabelSpreading", ls_preds)]:
            y_pred = le.inverse_transform(preds)
            rows.append({
                "fraction_labels_kept": frac,
                "model":                name,
                "macro_f1":             f1_score(y_true, y_pred, average="macro", zero_division=0),
                "kappa":                cohen_kappa_score(y_true, y_pred),
            })

        print(f"  fraction={frac:.2f} done")

    df_curve = pd.DataFrame(rows)

    print(f"\n{'═' * 60}")
    print("  DEGRADATION CURVE  (macro F1 vs. % labels retained)")
    print(f"{'═' * 60}")
    print(df_curve.pivot(index="fraction_labels_kept", columns="model", values="macro_f1")
          .to_string())
    print()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, color in [("RF_baseline", "#4C72B0"), ("LabelSpreading", "#DD8452")]:
            sub = df_curve[df_curve["model"] == name].sort_values("fraction_labels_kept")
            ax.plot(sub["fraction_labels_kept"], sub["macro_f1"],
                   marker="o", label=name, color=color)
        ax.set_xlabel("Fraction of training labels retained")
        ax.set_ylabel("Macro F1 (held-out)")
        ax.set_title("Does LabelSpreading degrade more gracefully than RF\n"
                     "as labeled data shrinks?")
        ax.invert_xaxis()  # scarce labels on the right, like a "less data →" story
        ax.legend()
        plt.tight_layout()
        out = os.path.join(save_path, "label_spreading_degradation_curve.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved → {out}")

    return df_curve


# ── 4. calibration check ──────────────────────────────────────────────────────

def plot_reliability_diagram(y_true, y_pred, confidence, n_bins=10,
                             confidence_threshold=0.80, save_path=None):
    """
    Check whether semi_confidence means what the 0.80 downstream threshold
    assumes it means. Bins held-out predictions by confidence and plots
    empirical accuracy per bin against the diagonal (perfect calibration).

    Feed this the output of evaluate_label_spreading_holdout() — using
    held-out confidences, not the in-sample ones from run_label_spreading(),
    since a model is trivially "confident and correct" about points it was
    fit on.

    Args:
        y_true (np.ndarray): True species strings, held-out.
        y_pred (np.ndarray): Predicted species strings, held-out.
        confidence (np.ndarray): Spreading confidence per prediction, held-out.
        n_bins (int): Number of confidence bins. Default 10.
        confidence_threshold (float): The threshold used downstream (e.g. in
            train_model.py) to mark a detection "confirmed". Drawn as a
            vertical line. Default 0.80.
        save_path (str | None): Directory to save the PNG into. Pass None
            to skip saving. Default None.

    Returns:
        pd.DataFrame: Per-bin table with mean confidence, empirical accuracy,
            and count.

    Requirements:
        numpy, pandas, matplotlib
    """
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(int)
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center":       (bins[b] + bins[b + 1]) / 2,
            "mean_confidence":  confidence[mask].mean(),
            "empirical_accuracy": correct[mask].mean(),
            "count":            int(mask.sum()),
        })
    df_bins = pd.DataFrame(rows)

    print(f"\n{'═' * 60}")
    print("  RELIABILITY CHECK — held-out confidence vs. empirical accuracy")
    print(f"{'═' * 60}")
    print(df_bins.to_string(index=False))

    at_thresh = df_bins[df_bins["bin_center"] >= confidence_threshold]
    if len(at_thresh):
        weighted_acc = np.average(at_thresh["empirical_accuracy"], weights=at_thresh["count"])
        print(f"\n  Empirical accuracy for confidence ≥ {confidence_threshold}: "
              f"{weighted_acc:.3f}")
        if weighted_acc < confidence_threshold:
            print(f"  ⚠  Predictions above the {confidence_threshold} threshold are "
                  f"less reliable than the threshold implies — downstream "
                  f"'confirmed detection' counts are optimistic.")
    print()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
        ax.plot(df_bins["mean_confidence"], df_bins["empirical_accuracy"],
               marker="o", color="#4C72B0", label="LabelSpreading (held-out)")
        ax.axvline(confidence_threshold, color="black", linestyle=":",
                  label=f"downstream threshold ({confidence_threshold})")
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Reliability Diagram — held-out LabelSpreading confidence")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(save_path, "label_spreading_reliability.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved → {out}")

    return df_bins


# ── convenience: run everything in one call ───────────────────────────────────

def run_full_evaluation(df_deep, df_labels_matched, save_path=None,
                        features=None, alpha=0.2, gamma=0.5, max_iter=100,
                        n_splits=5, random_state=42,
                        run_degradation=True, confidence_threshold=0.80):
    """
    Run the full evaluation suite in one call: honest holdout metrics, RF
    comparison, optional degradation curve, and reliability diagram.

    Intended as a drop-in addition to train_model.py, called instead of (or
    alongside) the in-sample sanity check inside run_label_spreading().

    Args:
        df_deep, df_labels_matched, features: same as above.
        save_path (str | None): Directory for diagnostic plots. Default None.
        alpha, gamma, max_iter: LabelSpreading params — pass the same values
            used in the real run_label_spreading() call so the eval reflects
            actual production settings.
        n_splits (int): Number of folds for all evaluations. Default 5.
        random_state (int): Shared seed. Default 42.
        run_degradation (bool): Whether to also run the (slower)
            degradation curve. Default True.
        confidence_threshold (float): Downstream confidence threshold to
            check calibration against. Default 0.80.

    Returns:
        dict: {
            "holdout":     dict from evaluate_label_spreading_holdout(),
            "comparison":  pd.DataFrame from compare_to_supervised_baseline(),
            "degradation": pd.DataFrame or None,
            "reliability": pd.DataFrame from plot_reliability_diagram(),
        }

    Requirements:
        numpy, pandas, matplotlib, scikit-learn
    """
    holdout = evaluate_label_spreading_holdout(
        df_deep, df_labels_matched, features=features,
        alpha=alpha, gamma=gamma, max_iter=max_iter,
        n_splits=n_splits, random_state=random_state,
    )

    comparison = compare_to_supervised_baseline(
        df_deep, df_labels_matched, features=features,
        alpha=alpha, gamma=gamma, max_iter=max_iter,
        n_splits=n_splits, random_state=random_state,
    )

    degradation = None
    if run_degradation:
        degradation = run_degradation_curve(
            df_deep, df_labels_matched, features=features,
            alpha=alpha, gamma=gamma, max_iter=max_iter,
            n_splits=n_splits, random_state=random_state,
            save_path=save_path,
        )

    reliability = plot_reliability_diagram(
        holdout["y_true"], holdout["y_pred"], holdout["confidence"],
        confidence_threshold=confidence_threshold, save_path=save_path,
    )

    return {
        "holdout":     holdout,
        "comparison":  comparison,
        "degradation": degradation,
        "reliability": reliability,
    }