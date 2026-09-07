"""
inspect_misclassifications.py
────────────────────────────────────────────────────────────────────────────────
Error analysis for the small number of misclassified labeled clusters (e.g.
"3 pinyons predicted as juniper, 1 as ponderosa") rather than another feature
engineering pass. At ~166-236 labeled clusters, a handful of errors is more
efficiently diagnosed by looking directly at those clusters than by adding
more columns to FEATURES.

Two things this answers that a single train/test split can't:
    1. Is each misclassification STABLE across different CV folds/seeds, or
       does it only show up sometimes? A cluster that gets flipped in 1/5
       seeds is probably CV noise; a cluster flipped in 5/5 seeds is a real
       geometric/color outlier worth a closer look.
    2. Does the misclassified cluster also show up in the reality-check
       flags (low n_points, extreme height_to_radius, near the label_distance
       edge)? That points to a bad cluster or a noisy GPS match rather than
       a feature-space gap the model could ever have closed.

Usage
─────
    from functions.classification.inspect_misclassifications import (
        get_out_of_fold_predictions, inspect_misclassified_clusters
    )

    df_errors = inspect_misclassified_clusters(
        df_deep_clusters,
        df_clusters,
        n_seeds=5,
        save_path=PATHS['Images'] + 'misclassification_report.csv'
    )
    print(df_errors)

Requirements
────────────
    numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from functions.feature_config import FEATURES


# ── out-of-fold predictions ───────────────────────────────────────────────────

def get_out_of_fold_predictions(df_deep, df_labels_matched, features=None,
                                 random_state=42, n_folds=None):
    """
    Run stratified cross_val_predict over the labeled subset so every labeled
    cluster gets a prediction from a fold it was NOT trained on — an honest
    proxy for "would this cluster be misclassified in production," rather
    than relying on whatever one train_test_split happened to hold out.

    Uses a plain balanced RandomForest (same baseline as train_tree_classifier
    without any of the enhancement toggles) so results are comparable across
    calls and not sensitive to which enhancements happen to be turned on.

    Args:
        df_deep (pd.DataFrame): Output of make_deep_dataframe() +
            engineer_features(). Must contain all FEATURES columns and "file".
        df_labels_matched (pd.DataFrame): Output of match_labels_to_clusters().
            Must have "file" and "Name".
        features (list[str] | None): Feature columns to use. Defaults to
            feature_config.FEATURES, restricted to columns actually present.
        random_state (int): Seed for both fold shuffling and the RF. Default 42.
        n_folds (int | None): Number of CV folds. Defaults to min(3, min_class)
            to match train_tree_classifier.py's fold-capping behaviour.

    Returns:
        pd.DataFrame: One row per labeled cluster with columns "file",
            "true_label", "predicted_label", "correct" (bool).

    Requirements:
        numpy, pandas, scikit-learn
    """
    df = df_deep.merge(df_labels_matched[["file", "Name", "label_distance"]],
                        on="file")
    df = df.dropna(subset=["Name"])
    df = df[df["Name"] != "unknown"].reset_index(drop=True)

    available = [f for f in (features or FEATURES) if f in df.columns]
    missing   = [f for f in (features or FEATURES) if f not in df.columns]
    if missing:
        print(f"  ⚠  Missing features (skipped): {missing}")

    X = df[available]
    y = df["Name"]

    min_class = y.value_counts().min()
    n_folds   = n_folds or min(3, int(min_class))

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
    )

    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    out = pd.DataFrame({
        "file":            df["file"].values,
        "true_label":      y.values,
        "predicted_label": y_pred,
        "correct":         y.values == y_pred,
    })

    return out


# ── multi-seed stability + full context ───────────────────────────────────────

def inspect_misclassified_clusters(df_deep, df_labels_matched, features=None,
                                    n_seeds=5, n_folds=None,
                                    true_label_filter=None,
                                    save_path=None):
    """
    Repeat get_out_of_fold_predictions() across n_seeds different fold
    shufflings, then build one row per cluster that was EVER misclassified,
    with:
        - how often it was misclassified (out of n_seeds) — stability check
        - the most common wrong label it was assigned
        - label_distance (is the GPS match itself borderline?)
        - the same reality-check flags plot_cluster_reality_check.py uses
          (low n_points, near-zero height, extreme height_to_radius)
        - the cluster's full feature row, for direct inspection

    A cluster flagged 5/5 times is a real, consistent error worth pulling
    the .ply for. A cluster flagged 1/5 or 2/5 times is most likely CV noise
    on a small fold and not worth chasing with more features.

    Args:
        df_deep (pd.DataFrame): Output of make_deep_dataframe() +
            engineer_features(). Must contain FEATURES columns and "file".
        df_labels_matched (pd.DataFrame): Output of match_labels_to_clusters().
            Must have "file", "Name", "label_distance".
        features (list[str] | None): Passed through to
            get_out_of_fold_predictions(). Default None (uses FEATURES).
        n_seeds (int): Number of different fold shufflings to run. Default 5.
        n_folds (int | None): Passed through to get_out_of_fold_predictions().
            Default None (auto: min(3, min_class)).
        true_label_filter (str | None): If given (e.g. "pinyon"), only rows
            whose TRUE label matches this are included in the report — use
            this to focus on "pinyons predicted as juniper" specifically.
            Pass None to include misclassifications of any true class.
            Default None.
        save_path (str | None): Full file path to write the report CSV to.
            Pass None to skip saving. Default None.

    Returns:
        pd.DataFrame: One row per misclassified cluster, sorted by
            misclassification frequency (most stable errors first), with
            columns: file, true_label, most_common_wrong_label,
            times_misclassified, n_seeds, label_distance, n_points, height,
            radius, height_to_radius, flag_low_points, flag_low_height,
            flag_extreme_ratio, plus every column in `features`.

    Requirements:
        numpy, pandas, scikit-learn
    """
    print(f"\n{'═' * 60}")
    print(f"  MISCLASSIFICATION STABILITY CHECK  ({n_seeds} seeds)")
    print(f"{'═' * 60}")

    all_runs = []
    for seed in range(n_seeds):
        run = get_out_of_fold_predictions(
            df_deep, df_labels_matched, features=features,
            random_state=seed, n_folds=n_folds
        )
        run["seed"] = seed
        all_runs.append(run)
        n_wrong = (~run["correct"]).sum()
        print(f"  seed={seed}: {n_wrong} misclassified / {len(run)} labeled")

    df_all = pd.concat(all_runs, ignore_index=True)

    wrong = df_all[~df_all["correct"]]
    if true_label_filter is not None:
        wrong = wrong[wrong["true_label"] == true_label_filter]

    if len(wrong) == 0:
        print(f"\n  No misclassifications found"
              f"{f' for true_label={true_label_filter}' if true_label_filter else ''}"
              f" across {n_seeds} seeds.\n")
        return pd.DataFrame()

    # ── aggregate stability per cluster ───────────────────────────────────
    summary_rows = []
    for file_idx, group in wrong.groupby("file"):
        true_label = group["true_label"].iloc[0]
        wrong_label_counts = group["predicted_label"].value_counts()
        summary_rows.append({
            "file":                     int(file_idx),
            "true_label":               true_label,
            "most_common_wrong_label":  wrong_label_counts.index[0],
            "times_misclassified":      len(group),
            "n_seeds":                  n_seeds,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(
        "times_misclassified", ascending=False
    ).reset_index(drop=True)

    # ── merge in full feature context ─────────────────────────────────────
    context_cols = ["file", "label_distance"]
    available = [f for f in (features or FEATURES) if f in df_deep.columns]

    df_context = df_deep[["file"] + [c for c in available if c not in ("file",)]].copy()
    df_context = df_context.merge(
        df_labels_matched[["file", "label_distance"]], on="file", how="left"
    )

    df_report = df_summary.merge(df_context, on="file", how="left")

    # ── reality-check style flags (same logic as plot_cluster_reality_check.py) ──
    if "n_points" in df_report.columns:
        df_report["flag_low_points"] = df_report["n_points"] < 20
    if "height" in df_report.columns:
        df_report["flag_low_height"] = df_report["height"] < 0.5
    if "height_to_radius" in df_report.columns:
        full_quantile_99 = df_deep["height_to_radius"].quantile(0.99)
        df_report["flag_extreme_ratio"] = df_report["height_to_radius"] > full_quantile_99

    print(f"\n  {len(df_report)} unique cluster(s) misclassified"
          f"{f' (true_label={true_label_filter})' if true_label_filter else ''}"
          f" at least once across {n_seeds} seeds:\n")

    display_cols = ["file", "true_label", "most_common_wrong_label",
                     "times_misclassified", "label_distance"]
    display_cols += [c for c in ("n_points", "height", "radius", "height_to_radius")
                      if c in df_report.columns]
    display_cols += [c for c in df_report.columns if c.startswith("flag_")]

    print(df_report[display_cols].to_string(index=False))
    print()

    stable_errors = df_report[df_report["times_misclassified"] == n_seeds]
    if len(stable_errors) > 0:
        print(f"  ⚠  {len(stable_errors)} cluster(s) misclassified in EVERY "
              f"seed — stable errors, worth pulling the .ply for a look: "
              f"{stable_errors['file'].tolist()}")
    noisy_errors = df_report[df_report["times_misclassified"] < n_seeds / 2]
    if len(noisy_errors) > 0:
        print(f"  ℹ  {len(noisy_errors)} cluster(s) misclassified in fewer "
              f"than half the seeds — likely CV noise, not a systematic "
              f"issue: {noisy_errors['file'].tolist()}")
    print()

    if save_path is not None:
        df_report.to_csv(save_path, index=False)
        print(f"  Saved → {save_path}\n")

    return df_report