# ── imports ───────────────────────────────────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless — safe for SLURM

from functions.feature_config import FEATURES


# ── default enhancement switches ──────────────────────────────────────────────
# These are the DEFAULTS used when train_tree_classifier() is called without
# an `enhancements` argument (e.g. direct interactive use, or old callers).
# Sweep callers (train_model.py) should pass an `enhancements` dict instead of
# mutating this module-level constant — mutating it from another module does
# nothing, since Python dicts imported via `from X import Y` for a *function*
# don't expose this module-level name at all unless explicitly imported, and
# even when imported, mutating it here doesn't affect a call already in
# progress if the function captured its own copy. Passing enhancements
# explicitly avoids all of that.
DEFAULT_ENHANCEMENTS = {
    "SMOTE":     False,   # synthetic oversampling for juniper/ponderosa
    "WEIGHTING": True,   # manual 2× upweight on ponderosa beyond "balanced"
    "SELECTION": True,   # drop features below mean RF importance
    "TUNING":    False,   # RandomizedSearchCV over RF hyperparameters (slowest)
}


def train_tree_classifier(df_deep, df_labels_matched,
                          save_confusion_matrix_path=None,
                          enhancements=None):
    """
    Train and compare species classifiers with four enhancements over the
    baseline, then return the best-performing model.

    Enhancements vs baseline
    ────────────────────────
    1. SMOTE oversampling  — synthetic minority samples for ponderosa/juniper
       generated *inside* each CV fold so validation folds stay clean.
    2. Manual class weights — ponderosa upweighted 2× beyond "balanced" so
       the 20-sample class is not drowned by the 108-sample pinyon majority.
    3. Feature selection   — SelectFromModel drops features below mean RF
       importance; selected feature list is printed for inspection.
    4. Hyperparameter tuning — RandomizedSearchCV over the RF parameter space
       most likely to matter on small datasets (depth, leaf size, features).

    Each variant runs alongside the existing baseline models so the comparison
    table shows exactly what each intervention buys.

    Args:
        df_deep (pd.DataFrame):
            Output of make_deep_dataframe() + engineer_features(). Must
            contain all columns in FEATURES.
        df_labels_matched (pd.DataFrame):
            Output of match_labels_to_clusters(). Must have "file" and "Name".
        save_confusion_matrix_path (str | None):
            Directory (or full path) to save the best-model confusion matrix
            PNG. Pass None to skip. Default None.
        enhancements (dict | None):
            Overrides for DEFAULT_ENHANCEMENTS — keys "SMOTE", "WEIGHTING",
            "SELECTION", "TUNING". Pass a partial dict to override just some
            keys; anything not given falls back to DEFAULT_ENHANCEMENTS.
            This is how sweep callers (train_model.py --smote/--weighting/
            --selection/--tuning) control these toggles per-run. Default
            None (use DEFAULT_ENHANCEMENTS unchanged).

    Returns:
        best_model: Fitted estimator or Pipeline with highest test macro F1.
        features (list[str]): Feature columns actually used after selection.
        best_metrics (dict): {"macro_f1": float, "kappa": float,
            "cv_macro_f1": float, "cv_std": float, "model_name": str} for
            the winning model — so sweep callers can log the score that
            actually resulted from a given enhancement combination, not
            just the combination itself.
    """

    ENHANCEMENTS = {**DEFAULT_ENHANCEMENTS, **(enhancements or {})}

    # ── merge features with labels ────────────────────────────────────────────
    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file")
    df = df.dropna(subset=["Name"])
    df = df[df["Name"] != "unknown"]

    print(f"\nTraining on {len(df)} labeled clusters")
    class_counts = df["Name"].value_counts()
    print("Class distribution:")
    print(class_counts.to_string())
    print()

    min_class = int(class_counts.min())
    if min_class < 10:
        print(f"⚠  Smallest class has only {min_class} samples — "
              f"treat CV scores with caution.\n")

    # ── only keep FEATURES columns that actually exist in df ─────────────────
    # guards against engineer_features() not having been called yet
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"⚠  Missing engineered features (call engineer_features() first): "
              f"{missing}\n")

    X = df[available]
    y = df["Name"]

    # ── train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── CV setup ──────────────────────────────────────────────────────────────
    n_folds = min(3, min_class)
    cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)


    # ═════════════════════════════════════════════════════════════════════════
    # ENHANCEMENT 1 — SMOTE
    # Generated inside ImbPipeline so synthetic samples never leak into the
    # validation fold. k_neighbors capped at min_class-1 to avoid a crash
    # when ponderosa has fewer than 6 samples in a fold.
    # ═════════════════════════════════════════════════════════════════════════
    if ENHANCEMENTS["SMOTE"]:
        print("Enhancement: SMOTE enabled")
        smote_k = min(4, min_class - 1)

        pinyon_count  = int(class_counts.get("pinyon", 108))
        smote_targets = {
            sp: min(pinyon_count, max(int(cnt * 2), 40))
            for sp, cnt in class_counts.items()
            if sp != "pinyon"
        }

        smote = SMOTE(
            sampling_strategy=smote_targets,
            k_neighbors=smote_k,
            random_state=42,
        )

        rf_smote = ImbPipeline([
            ("smote", smote),
            ("rf",    RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
            )),
        ])
    else:
        print("Enhancement: SMOTE disabled")
        rf_smote = None


    # ═════════════════════════════════════════════════════════════════════════
    # ENHANCEMENT 2 — Manual class weights
    # Start from sklearn's "balanced" weights then multiply ponderosa by 2×
    # so the 20-sample class is pushed harder than balanced alone achieves.
    # ═════════════════════════════════════════════════════════════════════════
    if ENHANCEMENTS["WEIGHTING"]:
        print("Enhancement: WEIGHTING enabled")
        classes_arr = np.array(sorted(y.unique()))
        bal_weights = compute_class_weight("balanced", classes=classes_arr, y=y)
        weight_dict = dict(zip(classes_arr, bal_weights))
        weight_dict["ponderosa"] = weight_dict.get("ponderosa", 1.0) * 2.0

        rf_weighted = RandomForestClassifier(
            n_estimators=200,
            class_weight=weight_dict,
            random_state=42,
        )
    else:
        print("Enhancement: WEIGHTING disabled")
        rf_weighted = None


    # ═════════════════════════════════════════════════════════════════════════
    # ENHANCEMENT 3 — Feature selection
    # Fit a quick RF on the training set, drop features below mean importance,
    # then retrain a clean RF on the reduced feature set. The selected feature
    # list is printed so you can hardcode survivors into FEATURES permanently.
    # ═════════════════════════════════════════════════════════════════════════
    if ENHANCEMENTS["SELECTION"]:
        print("Enhancement: SELECTION enabled")
        selector_rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
        selector = SelectFromModel(selector_rf, threshold="mean")
        selector.fit(X_train, y_train)

        selected_features = [f for f, keep
                             in zip(available, selector.get_support()) if keep]
        dropped_features  = [f for f, keep
                             in zip(available, selector.get_support()) if not keep]

        print(f"  {len(selected_features)} kept:    {selected_features}")
        print(f"  {len(dropped_features)} dropped: {dropped_features}")
        print()

        X_train_sel = X_train[selected_features]
        X_test_sel  = X_test[selected_features]
        X_sel       = X[selected_features]

        rf_selected = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        )
    else:
        print("Enhancement: SELECTION disabled")
        selected_features = None
        rf_selected       = None


    # ═════════════════════════════════════════════════════════════════════════
    # ENHANCEMENT 4 — Hyperparameter tuning
    # RandomizedSearchCV over the RF params most likely to matter on a small
    # dataset. n_iter=30 gives good coverage without a full grid search.
    # ═════════════════════════════════════════════════════════════════════════
    if ENHANCEMENTS["TUNING"]:
        print("Enhancement: TUNING enabled")
        param_dist = {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [None, 5, 10, 15, 20],
            "min_samples_leaf":  [1, 2, 4, 6],
            "min_samples_split": [2, 5, 10],
            "max_features":      ["sqrt", "log2", 0.5],
        }

        print(f"  Running hyperparameter search (30 iterations × {n_folds} folds)...")
        tuned_search = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=42),
            param_distributions=param_dist,
            n_iter=30,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        tuned_search.fit(X_train, y_train)
        rf_tuned = tuned_search.best_estimator_

        print(f"  Best params: {tuned_search.best_params_}")
        print(f"  Best CV F1:  {tuned_search.best_score_:.3f}")
        print()
    else:
        print("Enhancement: TUNING disabled")
        rf_tuned = None


    # ── model zoo ─────────────────────────────────────────────────────────────
    # baseline models + the four enhanced variants all in one comparison table
    models = {
        # ── baselines (unchanged from previous version) ────────────────────
        "RF_baseline": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", class_weight="balanced",
                           probability=True, random_state=42)),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("knn",    KNeighborsClassifier(n_neighbors=5)),
        ]),
    }

    # ── add enabled enhanced variants to the model zoo ────────────────────
    if ENHANCEMENTS["SMOTE"]     and rf_smote    is not None:
        models["RF_SMOTE"]    = rf_smote
    if ENHANCEMENTS["WEIGHTING"] and rf_weighted is not None:
        models["RF_weighted"] = rf_weighted
    if ENHANCEMENTS["TUNING"]    and rf_tuned    is not None:
        models["RF_tuned"]    = rf_tuned

    # ── evaluate each model ───────────────────────────────────────────────────
    results_summary = []
    fitted_models   = {}

    for name, model in models.items():
        print(f"{'─' * 55}")
        print(f"  {name}")
        print(f"{'─' * 55}")

        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa    = cohen_kappa_score(y_test, y_pred)

        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"  Macro F1:      {macro_f1:.3f}")
        print(f"  Cohen's Kappa: {kappa:.3f}")

        cv_scores = cross_val_score(model, X, y, cv=cv,
                                    scoring="f1_macro", n_jobs=-1)
        print(f"  CV Macro F1 ({n_folds}-fold): "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
              f"(folds: {', '.join(f'{s:.3f}' for s in cv_scores)})")
        print()

        results_summary.append({
            "model":       name,
            "macro_f1":    round(macro_f1, 4),
            "kappa":       round(kappa, 4),
            "cv_macro_f1": round(cv_scores.mean(), 4),
            "cv_std":      round(cv_scores.std(), 4),
            "features":    "all",
        })
        fitted_models[name] = (model, available)

    # ── enhancement 3 evaluated separately (reduced feature set) ─────────────
    if ENHANCEMENTS["SELECTION"] and rf_selected is not None:
        print(f"{'─' * 55}")
        print(f"  RF_selected  ({len(selected_features)} features)")
        print(f"{'─' * 55}")

        rf_selected.fit(X_train_sel, y_train)
        y_pred_sel   = rf_selected.predict(X_test_sel)
        macro_f1_sel = f1_score(y_test, y_pred_sel, average="macro", zero_division=0)
        kappa_sel    = cohen_kappa_score(y_test, y_pred_sel)

        print(classification_report(y_test, y_pred_sel, zero_division=0))
        print(f"  Macro F1:      {macro_f1_sel:.3f}")
        print(f"  Cohen's Kappa: {kappa_sel:.3f}")

        cv_sel = cross_val_score(rf_selected, X_sel, y, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        print(f"  CV Macro F1 ({n_folds}-fold): "
              f"{cv_sel.mean():.3f} ± {cv_sel.std():.3f}  "
              f"(folds: {', '.join(f'{s:.3f}' for s in cv_sel)})")
        print()

        results_summary.append({
            "model":       "RF_selected",
            "macro_f1":    round(macro_f1_sel, 4),
            "kappa":       round(kappa_sel, 4),
            "cv_macro_f1": round(cv_sel.mean(), 4),
            "cv_std":      round(cv_sel.std(), 4),
            "features":    f"{len(selected_features)} selected",
        })
        fitted_models["RF_selected"] = (rf_selected, selected_features)


    # ── comparison table ──────────────────────────────────────────────────────
    df_results = pd.DataFrame(results_summary).sort_values(
        "macro_f1", ascending=False
    ).reset_index(drop=True)

    print(f"\n{'═' * 55}")
    print("  Model comparison (sorted by test Macro F1)")
    print(f"{'═' * 55}")
    print(df_results.to_string(index=False))
    print()

    # ── pick best model ───────────────────────────────────────────────────────
    best_name              = df_results.iloc[0]["model"]
    best_model, best_feats = fitted_models[best_name]

    print(f"  Best model: {best_name}  "
          f"(Macro F1={df_results.iloc[0]['macro_f1']:.3f}, "
          f"Kappa={df_results.iloc[0]['kappa']:.3f})")

    # ── confusion matrix for best model ──────────────────────────────────────
    y_pred_best = best_model.predict(X_test[best_feats])
    classes     = sorted(y.unique())
    cm          = confusion_matrix(y_test, y_pred_best, labels=classes)

    print(f"\n  Confusion matrix — {best_name}:")
    print(f"  (rows = true label, cols = predicted)")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())
    print()

    if save_confusion_matrix_path is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {best_name}\n"
                     f"(Macro F1={df_results.iloc[0]['macro_f1']:.3f}, "
                     f"Kappa={df_results.iloc[0]['kappa']:.3f})")
        plt.tight_layout()
        plt.savefig(save_confusion_matrix_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved to {save_confusion_matrix_path}")

    # ── feature importances (RF and GB only) ─────────────────────────────────
    estimator = best_model
    if hasattr(best_model, "named_steps"):
        # unwrap Pipeline or ImbPipeline to get the actual estimator
        estimator = list(best_model.named_steps.values())[-1]

    if hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=best_feats)
        print(f"\n  Feature importances — {best_name}:")
        print(importances.sort_values(ascending=False).to_string())
        print()

    # ── predict on all clusters (including unlabeled) ─────────────────────────
    df_deep["predicted_label"] = best_model.predict(df_deep[best_feats])

    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(df_deep[best_feats])
        # classes_ may live on the pipeline's last step
        if hasattr(best_model, "classes_"):
            out_classes = best_model.classes_
        else:
            out_classes = estimator.classes_
        for i, cls in enumerate(out_classes):
            df_deep[f"prob_{cls}"] = proba[:, i]

    best_metrics = {
        "model_name":  best_name,
        "macro_f1":    float(df_results.iloc[0]["macro_f1"]),
        "kappa":       float(df_results.iloc[0]["kappa"]),
        "cv_macro_f1": float(df_results.iloc[0]["cv_macro_f1"]),
        "cv_std":      float(df_results.iloc[0]["cv_std"]),
    }

    return best_model, best_feats, best_metrics