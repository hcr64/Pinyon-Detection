"""
train_drought_classifier.py
────────────────────────────────────────────────────────────────────────────
Binary drought-tolerance classifier (tolerant vs. susceptible) trained on
the pinyon subset of labeled clusters. Field labels are "pinyon T", "pinyon
T4", "pinyon S", "pinyon S7", etc. — match_labels_to_clusters() parses the
letter (T/S) into "drought_class" and the optional 1-10 severity number
into "drought_level" on df_clusters. This module trains on drought_class
only; drought_level is carried through but unused for now. Juniper and
ponderosa clusters are excluded — only pinyon received a drought tag.

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


def train_drought_classifier(df_deep, df_labels_matched,
                              save_confusion_matrix_path=None,
                              features=None, min_per_class=6,
                              random_state=42):
    """
    Train and compare binary tolerant/susceptible classifiers on the
    drought-labeled pinyon subset.

    Args:
        df_deep (pd.DataFrame): Output of make_deep_dataframe() +
            engineer_features(). Must contain "file" and the columns in
            `features`.
        df_labels_matched (pd.DataFrame): Output of
            match_labels_to_clusters(). Must have "file", "Name",
            "drought_class".
        save_confusion_matrix_path (str | None): Full path to save the
            best-model confusion matrix PNG. Default None.
        features (list[str] | None): Feature columns to use. Defaults to
            feature_config.FEATURES, restricted to columns present.
        min_per_class (int): Minimum labeled samples required in EACH of
            tolerant/susceptible to attempt training. Below this, prints a
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

    df = df_deep.merge(
        df_labels_matched[["file", "Name", "drought_class"]], on="file"
    )
    df = df[df["Name"] == "pinyon"]
    df = df.dropna(subset=["drought_class"])

    print(f"\n{'═' * 60}")
    print("  DROUGHT TOLERANCE CLASSIFIER  (pinyon: tolerant vs. susceptible)")
    print(f"{'═' * 60}")
    print(f"  Labeled pinyon clusters with a drought tag: {len(df)}")

    class_counts = df["drought_class"].value_counts()
    print(class_counts.to_string())
    print()

    if len(class_counts) < 2 or class_counts.min() < min_per_class:
        print(f"  Not enough labeled data to train — need at least "
              f"{min_per_class} samples in each of tolerant/susceptible, "
              f"got:\n{class_counts.to_string()}\n")
        return None, None, None

    min_class = int(class_counts.min())

    X = df[available]
    y = df["drought_class"]

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
    print("  Drought classifier comparison (sorted by test Macro F1)")
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
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Drought Tolerance — {best_name}\n"
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

    pinyon_files = df_labels_matched.loc[
        df_labels_matched["Name"] == "pinyon", "file"
    ]
    pinyon_mask = df_deep["file"].isin(pinyon_files)

    df_deep.loc[pinyon_mask, "predicted_drought_class"] = best_model.predict(
        df_deep.loc[pinyon_mask, available]
    )

    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(df_deep.loc[pinyon_mask, available])
        out_classes = best_model.classes_ if hasattr(best_model, "classes_") \
            else estimator.classes_
        for i, cls in enumerate(out_classes):
            df_deep.loc[pinyon_mask, f"prob_drought_{cls}"] = proba[:, i]

    best_metrics = {
        "model_name": best_name,
        "macro_f1": float(df_results.iloc[0]["macro_f1"]),
        "kappa": float(df_results.iloc[0]["kappa"]),
        "cv_macro_f1": float(df_results.iloc[0]["cv_macro_f1"]),
        "cv_std": float(df_results.iloc[0]["cv_std"]),
    }

    return best_model, available, best_metrics