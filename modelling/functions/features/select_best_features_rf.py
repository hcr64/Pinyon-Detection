"""
select_best_features_rf.py
────────────────────────────────────────────────────────────────────────────
Random-forest-based feature selection for the species classifier.

Ranks candidate features by cross-validated permutation importance (more
reliable than raw RF impurity importance, which is biased toward
high-cardinality/continuous features like n_points and eigenvalues), then
proposes a selected subset via a threshold on mean importance.

Why permutation importance over the impurity-based SelectFromModel already
used in train_tree_classifier.py's SELECTION enhancement: impurity
importance is computed on the training set and inflates importance for
features RF can overfit to, which matters more here given ~166-236 labeled
clusters. Permutation importance is computed on each fold's held-out test
set instead, so it reflects genuinely useful signal rather than
memorization capacity.

Also runs a direct full-features-vs-selected-features comparison (macro F1,
kappa, AND juniper recall) so a selection that trims macro-noise but quietly
hurts juniper recall — the project's primary failure mode — is visible
rather than hidden behind an aggregate metric.

Usage
─────
    from functions.features.select_best_features_rf import select_best_features_rf

    importance_df, selected_features, comparison_df = select_best_features_rf(
        df_deep_clusters, df_clusters
    )

Requirements
────────────
    numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

from functions.feature_config import FEATURES


def select_best_features_rf(df_deep, df_labels_matched, features=None,
                            importance_type="permutation",
                            threshold="mean", n_repeats=20,
                            n_splits=None, n_estimators=300,
                            random_state=42, verbose=True):
    """
    Rank candidate features by RF importance and propose a selected subset.

    Args:
        df_deep (pd.DataFrame): Output of make_deep_dataframe() +
            engineer_features(). Must contain a "file" column and all
            columns in `features`.
        df_labels_matched (pd.DataFrame): Output of match_labels_to_clusters().
            Must have "file" and "Name".
        features (list[str] | None): Candidate feature columns to rank.
            Defaults to FEATURES from feature_config.py.
        importance_type ("permutation" | "impurity"): "permutation"
            (default) computes importance on held-out folds via
            sklearn.inspection.permutation_importance, scored on macro F1.
            "impurity" uses RF's built-in feature_importances_ (faster,
            but computed on training data — prone to overstating importance
            for features RF can overfit to on a dataset this small).
        threshold ("mean" | "median" | float): Cutoff on mean importance
            for a feature to be marked "selected". "mean" (default) keeps
            features above the average importance across all candidates,
            matching the threshold semantics already used by
            SelectFromModel(threshold="mean") in train_tree_classifier.py's
            SELECTION enhancement, so results are comparable side-by-side.
        n_repeats (int): Number of shuffles per feature in permutation
            importance (only used when importance_type="permutation").
            Default 20.
        n_splits (int | None): CV folds. Defaults to min(3, min_class),
            same cap used everywhere else in this project to handle the
            ~20-sample ponderosa class. Default None.
        n_estimators (int): Trees per RF fit. Default 300.
        random_state (int): Shared seed for CV folds and RF. Default 42.
        verbose (bool): Print ranking table and comparison. Default True.

    Returns:
        importance_df (pd.DataFrame): One row per feature — "feature",
            "mean_importance", "std_importance" (across folds), "selected"
            (bool), sorted by mean_importance descending.
        selected_features (list[str]): Features where "selected" is True.
        comparison_df (pd.DataFrame): Cross-validated macro_f1, kappa, and
            juniper recall for the full feature set vs. the selected subset,
            using the SAME cv folds as the ranking step.

    Requirements:
        numpy, pandas, scikit-learn
    """
    candidate = features or FEATURES
    available = [f for f in candidate if f in df_deep.columns]
    missing = [f for f in candidate if f not in df_deep.columns]
    if missing and verbose:
        print(f"⚠  Missing features (skipped, regenerate deep_clusters.csv?): {missing}")

    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file")
    df = df.dropna(subset=["Name"])
    df = df[df["Name"] != "unknown"]

    X = df[available]
    y = df["Name"]

    class_counts = y.value_counts()
    min_class = int(class_counts.min())
    if verbose:
        print(f"Ranking {len(available)} candidate features on {len(df)} "
              f"labeled clusters")
        print(class_counts.to_string())
        if min_class < 10:
            print(f"⚠  Smallest class has only {min_class} samples — "
                  f"treat rankings with caution, especially fold-to-fold "
                  f"std_importance.")
        print()

    n_splits = n_splits or min(3, min_class)
    if n_splits < 2:
        raise ValueError(
            f"Smallest class ({min_class} samples) can't support even a "
            f"2-fold split — need more labeled data for this class before "
            f"feature selection is meaningful."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ── per-fold importance ────────────────────────────────────────────────
    records = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=random_state
        )
        rf.fit(X_train, y_train)

        if importance_type == "permutation":
            result = permutation_importance(
                rf, X_test, y_test, n_repeats=n_repeats,
                random_state=random_state, scoring="f1_macro", n_jobs=-1
            )
            imp = result.importances_mean
        elif importance_type == "impurity":
            imp = rf.feature_importances_
        else:
            raise ValueError('importance_type must be "permutation" or "impurity"')

        for feat, val in zip(available, imp):
            records.append({"fold": fold, "feature": feat, "importance": val})

    df_imp = pd.DataFrame(records)
    importance_df = (
        df_imp.groupby("feature")["importance"]
        .agg(mean_importance="mean", std_importance="std")
        .reset_index()
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── threshold + selection ────────────────────────────────────────────
    if threshold == "mean":
        cutoff = importance_df["mean_importance"].mean()
    elif threshold == "median":
        cutoff = importance_df["mean_importance"].median()
    elif isinstance(threshold, (int, float)):
        cutoff = threshold
    else:
        raise ValueError('threshold must be "mean", "median", or a float')

    importance_df["selected"] = importance_df["mean_importance"] >= cutoff
    selected_features = importance_df.loc[importance_df["selected"], "feature"].tolist()

    if verbose:
        print(f"{'─' * 55}")
        print(f"  Feature importance ({importance_type}, {n_splits}-fold CV)")
        print(f"  cutoff ({threshold}): {cutoff:.4f}")
        print(f"{'─' * 55}")
        print(importance_df.to_string(index=False))
        print(f"\n  Selected {len(selected_features)}/{len(available)} features:")
        print(f"    {selected_features}\n")

    # ── full vs. selected comparison, same cv folds ──────────────────────
    def _cv_eval(feat_list):
        rf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=random_state
        )
        y_pred = cross_val_predict(rf, X[feat_list], y, cv=cv)
        macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        juniper_recall = report.get("juniper", {}).get("recall", np.nan)
        return macro_f1, kappa, juniper_recall

    full_f1, full_kappa, full_jrec = _cv_eval(available)
    if selected_features:
        sel_f1, sel_kappa, sel_jrec = _cv_eval(selected_features)
    else:
        sel_f1, sel_kappa, sel_jrec = np.nan, np.nan, np.nan

    comparison_df = pd.DataFrame([
        {"feature_set": f"all ({len(available)})", "macro_f1": round(full_f1, 4),
         "kappa": round(full_kappa, 4), "juniper_recall": round(full_jrec, 4)},
        {"feature_set": f"selected ({len(selected_features)})", "macro_f1": round(sel_f1, 4),
         "kappa": round(sel_kappa, 4), "juniper_recall": round(sel_jrec, 4)},
    ])

    if verbose:
        print(f"{'─' * 55}")
        print("  Full vs. selected (same CV folds)")
        print(f"{'─' * 55}")
        print(comparison_df.to_string(index=False))
        if not np.isnan(sel_jrec) and sel_jrec < full_jrec - 0.02:
            print(f"\n  ⚠  Selected subset drops juniper recall "
                  f"({full_jrec:.3f} → {sel_jrec:.3f}) — try threshold="
                  f"'median' or a lower explicit cutoff before adopting "
                  f"this subset for juniper-recall-focused runs.")
        print()

    return importance_df, selected_features, comparison_df