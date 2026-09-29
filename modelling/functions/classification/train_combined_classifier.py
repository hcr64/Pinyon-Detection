"""
train_combined_classifier.py
────────────────────────────────────────────────────────────────────────────
Combined species + drought-tolerance classifier: predicts one of four
classes — juniper, ponderosa, pinyon_tolerant, pinyon_susceptible — in a
single model rather than chaining species classification into a separate
drought-tolerance model.

Pinyon clusters that were GPS-matched but never received a T/S field tag
(df_clusters["drought_class"] is NaN) cannot be assigned to either pinyon
subclass, so they're excluded from training. Juniper and ponderosa never
carry a drought tag at all — every matched juniper/ponderosa cluster is
used as-is.

Requirements
────────────
    numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from functions.feature_config import FEATURES


def build_combined_label(df_clusters):
    """
    Build the 4-class species+drought target column.

    Maps:
        "juniper"                          -> "juniper"
        "ponderosa"                        -> "ponderosa"
        "pinyon" + drought_class=tolerant   -> "pinyon_tolerant"
        "pinyon" + drought_class=susceptible-> "pinyon_susceptible"
        "pinyon" + drought_class=NaN        -> NaN (excluded — ambiguous)
        "unknown" / anything else           -> NaN

    Args:
        df_clusters (pd.DataFrame): Must have "Name" and, if any pinyon
            rows are present, "drought_class" (output of
            match_labels_to_clusters()).

    Returns:
        pd.Series: Object dtype, same index as df_clusters. NaN where the
            cluster can't be assigned to one of the four classes.

    Requirements:
        pandas, numpy
    """
    name = df_clusters["Name"]
    drought = df_clusters["drought_class"] if "drought_class" in df_clusters.columns \
        else pd.Series(np.nan, index=df_clusters.index)

    combined = pd.Series(np.nan, index=df_clusters.index, dtype=object)
    combined[name == "juniper"] = "juniper"
    combined[name == "ponderosa"] = "ponderosa"
    combined[(name == "pinyon") & (drought == "tolerant")] = "pinyon_tolerant"
    combined[(name == "pinyon") & (drought == "susceptible")] = "pinyon_susceptible"

    return combined


def train_combined_classifier(df_deep, df_labels_matched,
                               save_confusion_matrix_path=None,
                               features=None, min_per_class=6,
                               random_state=42):
    """
    Train and compare classifiers on the combined 4-class target
    (juniper / ponderosa / pinyon_tolerant / pinyon_susceptible).

    Args:
        df_deep (pd.DataFrame): Output of make_deep_dataframe() +
            engineer_features(). Must contain "file" and the columns in
            `features`.
        df_labels_matched (pd.DataFrame): Output of
            match_labels_to_clusters(). Must have "file", "Name", and
            "drought_class".
        save_confusion_matrix_path (str | None): Full path to save the
            best-model confusion matrix PNG. Default None.
        features (list[str] | None): Feature columns to use. Defaults to
            feature_config.FEATURES, restricted to columns present.
        min_per_class (int): Minimum labeled samples required in EACH of
            the 4 classes to attempt training. Below this, prints a
            warning and returns (None, None, None). Default 6.
        random_state (int): Shared seed for the split, CV folds, and every
            model. Default 42.

    Returns:
        best_model: Fitted estimator or Pipeline with highest test macro
            F1, or None if there wasn't enough labeled data.
        best_feats (list[str] | None): Feature columns used.
        best_metrics (dict | None): {"model_name", "macro_f1", "kappa",
            "cv_macro_f1", "cv_std"}.

    Requirements:
        numpy, pandas, matplotlib, scikit-learn
    """
    available = [f for f in (features or FEATURES) if f in df_deep.columns]
    missing   = [f for f in (features or FEATURES) if f not in df_deep.columns]
    if missing:
        print(f"⚠  Missing features (skipped): {missing}\n")

    combined_label = build_combined_label(df_labels_matched)
    df = df_deep.merge(
        pd.DataFrame({"file": df_labels_matched["file"], "combined": combined_label}),
        on="file"
    )
    df = df.dropna(subset=["combined"])

    print(f"\n{'═' * 60}")
    print("  COMBINED SPECIES + DROUGHT CLASSIFIER")
    print("  (juniper / ponderosa / pinyon_tolerant / pinyon_susceptible)")
    print(f"{'═' * 60}")
    print(f"  Labeled clusters usable for this target: {len(df)}")
    print(f"  (pinyon clusters without a T/S tag are excluded — see "
          f"build_combined_label())")

    class_counts = df["combined"].value_counts()
    print(class_counts.to_string())
    print()

    if len(class_counts) < 2 or class_counts.min() < min_per_class:
        print(f"  Not enough labeled data to train — need at least "
              f"{min_per_class} samples in each class, got:\n"
              f"{class_counts.to_string()}\n")
        return None, None, None

    min_class = int(class_counts.min())

    X = df[available]
    y = df["combined"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    n_folds = min(3, min_class)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    models = {
        "RF": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=random_state
        ),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced",
                        probability=True, random_state=random_state)),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=min(5, min_class))),
        ]),
    }

    results_summary = []
    fitted_models = {}

    for name, model in models.items():
        print(f"{'─' * 55}")
        print(f"  {name}")
        print(f"{'─' * 55}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)

        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"  Macro F1:      {macro_f1:.3f}")
        print(f"  Cohen's Kappa: {kappa:.3f}")

        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"  CV Macro F1 ({n_folds}-fold): "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
              f"(folds: {', '.join(f'{s:.3f}' for s in cv_scores)})")
        print()

        results_summary.append({
            "model": name,
            "macro_f1": round(macro_f1, 4),
            "kappa": round(kappa, 4),
            "cv_macro_f1": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
        })
        fitted_models[name] = model

    df_results = pd.DataFrame(results_summary).sort_values(
        "macro_f1", ascending=False
    ).reset_index(drop=True)

    print(f"\n{'═' * 55}")
    print("  Combined classifier comparison (sorted by test Macro F1)")
    print(f"{'═' * 55}")
    print(df_results.to_string(index=False))
    print()

    best_name = df_results.iloc[0]["model"]
    best_model = fitted_models[best_name]

    print(f"  Best model: {best_name}  "
          f"(Macro F1={df_results.iloc[0]['macro_f1']:.3f}, "
          f"Kappa={df_results.iloc[0]['kappa']:.3f})")

    classes = sorted(y.unique())
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=classes)

    print(f"\n  Confusion matrix — {best_name}:")
    print("  (rows = true label, cols = predicted)")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    print()

    if save_confusion_matrix_path is not None:
        dirname = os.path.dirname(save_confusion_matrix_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
        ax.set_title(f"Combined Species + Drought — {best_name}\n"
                     f"(Macro F1={df_results.iloc[0]['macro_f1']:.3f}, "
                     f"Kappa={df_results.iloc[0]['kappa']:.3f})")
        plt.tight_layout()
        plt.savefig(save_confusion_matrix_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved to {save_confusion_matrix_path}")

    estimator = best_model
    if hasattr(best_model, "named_steps"):
        estimator = list(best_model.named_steps.values())[-1]

    if hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=available)
        print(f"\n  Feature importances — {best_name}:")
        print(importances.sort_values(ascending=False).to_string())
        print()

    # ── predict on every cluster (labeled + unlabeled) ────────────────────
    df_deep["predicted_combined"] = best_model.predict(df_deep[available])

    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(df_deep[available])
        out_classes = best_model.classes_ if hasattr(best_model, "classes_") \
            else estimator.classes_
        for i, cls in enumerate(out_classes):
            df_deep[f"prob_combined_{cls}"] = proba[:, i]

    best_metrics = {
        "model_name": best_name,
        "macro_f1": float(df_results.iloc[0]["macro_f1"]),
        "kappa": float(df_results.iloc[0]["kappa"]),
        "cv_macro_f1": float(df_results.iloc[0]["cv_macro_f1"]),
        "cv_std": float(df_results.iloc[0]["cv_std"]),
    }

    return best_model, available, best_metrics