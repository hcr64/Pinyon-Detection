from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless — safe for SLURM


# ── feature columns used by all models ───────────────────────────────────────

FEATURES = [
    "height", "radius", "n_points",
    "obb_extent_x", "obb_extent_y", "obb_extent_z",
    "eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
    "linearity", "planarity", "sphericity",
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
]


def train_tree_classifier(df_deep, df_labels_matched,
                          save_confusion_matrix_path=None):
    """
    Train and evaluate species classifiers on labeled cluster features.
 
    Merges df_deep with df_labels_matched on the "file" index column, drops
    unlabeled and "unknown" rows, then trains multiple classifier types
    (Gradient Boosting, SVM/RBF, Logistic Regression, KNN, and an ensemble)
    each wrapped in a Pipeline with StandardScaler. Cross-validation uses
    StratifiedKFold with fold count capped at min(3, min_class_size) to
    handle small juniper and ponderosa sample sizes.
 
    Class imbalance is expected — pinyon likely dominates. Raw accuracy is
    misleading; evaluate using per-class precision/recall/F1, macro F1, and
    Cohen's Kappa which are all reported. Confusion matrix PNGs and
    predict_proba columns are also produced.
 
    After training, the best model predicts species for *all* clusters in
    df_deep (including unlabeled ones) and appends a "predicted_label" column.
    In main.py the predictions are additionally filtered by confidence
    (pinyon_confidence > 0.80) before counting confirmed detections.
 
    Args:
        df_deep (pd.DataFrame): Feature DataFrame from make_deep_dataframe().
            Must contain the feature columns listed below.
        df_labels_matched (pd.DataFrame): Cluster DataFrame returned by
            match_labels_to_clusters(). Must have "file" and "Name" columns.
 
    Returns:
        model: Trained best-performing classifier (sklearn Pipeline).
        features (list of str): Feature column names used for training:
            height, radius, n_points, obb_extent_x/y/z,
            eigenvalue_1/2/3, linearity, planarity, sphericity,
            mean_r/g/b, std_r/g/b.
 
    Requirements:
        sklearn.pipeline, sklearn.preprocessing, sklearn.ensemble,
        sklearn.svm, sklearn.linear_model, sklearn.neighbors,
        sklearn.model_selection, sklearn.metrics, pandas
    """

    # ── merge features with labels ────────────────────────────────────────────
    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file")
    df = df.dropna(subset=["Name"])
    df = df[df["Name"] != "unknown"]

    print(f"\nTraining on {len(df)} labeled clusters")
    class_counts = df["Name"].value_counts()
    print("Class distribution:")
    print(class_counts.to_string())
    print()

    # warn early if any class is very small — CV scores will be noisy
    min_class = class_counts.min()
    if min_class < 10:
        print(f"⚠  Smallest class has only {min_class} samples — "
              f"treat CV scores with caution.\n")

    X = df[FEATURES]
    y = df["Name"]

    # ── train / test split ────────────────────────────────────────────────────
    # stratify so every split has all species represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── model zoo ─────────────────────────────────────────────────────────────
    # SVM and LR need scaled features; wrap them in a Pipeline so the scaler
    # is fitted only on training data (no leakage into CV folds).
    models = {
        "RandomForest": RandomForestClassifier(
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

    # ── cross-validation setup ────────────────────────────────────────────────
    # use min(3, min_class) folds so every fold has at least 1 sample per class
    n_folds = min(3, int(min_class))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # ── evaluate each model ───────────────────────────────────────────────────
    results_summary = []
    fitted_models   = {}

    for name, model in models.items():
        print(f"{'─' * 55}")
        print(f"  {name}")
        print(f"{'─' * 55}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa    = cohen_kappa_score(y_test, y_pred)

        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"  Macro F1:      {macro_f1:.3f}")
        print(f"  Cohen's Kappa: {kappa:.3f}")

        # per-fold CV scores
        cv_scores = cross_val_score(model, X, y, cv=cv,
                                    scoring="f1_macro", n_jobs=-1)
        print(f"  CV Macro F1 ({n_folds}-fold): "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
              f"(folds: {', '.join(f'{s:.3f}' for s in cv_scores)})")
        print()

        results_summary.append({
            "model":         name,
            "macro_f1":      round(macro_f1, 4),
            "kappa":         round(kappa, 4),
            "cv_macro_f1":   round(cv_scores.mean(), 4),
            "cv_std":        round(cv_scores.std(), 4),
        })

        fitted_models[name] = model

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
    best_name  = df_results.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"  Best model: {best_name}  "
          f"(Macro F1={df_results.iloc[0]['macro_f1']:.3f}, "
          f"Kappa={df_results.iloc[0]['kappa']:.3f})")

    # ── confusion matrix for best model ──────────────────────────────────────
    y_pred_best = best_model.predict(X_test)
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
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURES)
        print(f"\n  Feature importances — {best_name}:")
        print(importances.sort_values(ascending=False).to_string())
        print()

    # ── predict on all clusters (including unlabeled) ─────────────────────────
    df_deep["predicted_label"] = best_model.predict(df_deep[FEATURES])

    # attach predict_proba if the model supports it
    if hasattr(best_model, "predict_proba"):
        proba      = best_model.predict_proba(df_deep[FEATURES])
        classes_   = best_model.classes_ if hasattr(best_model, "classes_") \
                     else best_model.named_steps[list(best_model.named_steps)[-1]].classes_
        for i, cls in enumerate(classes_):
            df_deep[f"prob_{cls}"] = proba[:, i]

    return best_model, FEATURES