"""
advanced_classifiers.py
────────────────────────────────────────────────────────────────────────────────
Additional species classifiers for Pinyon-Detection.

Models included:
    1. XGBoost          — gradient boosting with native class-weight handling;
                          often the strongest single model on small tabular data
    2. LightGBM         — faster gradient boosting, handles class imbalance via
                          is_unbalance; good when pinyon heavily dominates
    3. Soft-voting      — ensemble of your existing best models (RF, GB, SVM);
                          averages predict_proba so agreement boosts confidence
    4. MLP              — shallow neural net; captures non-linear interactions
                          between color and geometry features the tree models miss
    5. Calibrated SVM   — SVM with Platt scaling; predict_proba from plain SVM
                          is unreliable, calibration fixes that so confidence
                          scores on pinyon detections are actually meaningful

Usage (drop-in replacement, same signature as train_tree_classifier):

    from advanced_classifiers import run_advanced_classifiers

    best_model, features = run_advanced_classifiers(
        df_deep_clusters,
        df_clusters,
        save_path=PATHS['Images']      # optional — saves confusion matrix PNGs
    )

Requirements (add to Monsoon venv):
    pip install xgboost lightgbm
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ── feature set (must match make_deep_dataframe output) ──────────────────────

FEATURES = [
    "height", "radius", "n_points",
    "obb_extent_x", "obb_extent_y", "obb_extent_z",
    "eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
    "linearity", "planarity", "sphericity",
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
]


# ── model definitions ─────────────────────────────────────────────────────────

def build_models(class_weights: dict) -> dict:
    """
    Instantiate all advanced models with appropriate class-imbalance handling.

    class_weights is a dict mapping species name → weight (higher = rarer class
    gets more attention). Computed automatically from the training set inside
    run_advanced_classifiers().

    Returns a dict of {name: estimator_or_pipeline}.
    """

    # XGBoost requires integer class labels, so LabelEncoder is applied outside
    # this function. sample_weight is passed at fit() time — see run_advanced_classifiers().
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    # LightGBM — is_unbalance reweights internally, no need to pass sample_weight
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        is_unbalance=True,          # handles pinyon-heavy imbalance automatically
        random_state=42,
        verbose=-1,                 # suppress LightGBM's own output
    )

    # Soft-voting ensemble — averages predict_proba from RF, GB, and SVM.
    # Each sub-model has class_weight="balanced" so minority species aren't drowned.
    # Weights (2,2,1) give tree ensembles slightly more say than SVM.
    rf  = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)

    voting = Pipeline([
        ("scaler", StandardScaler()),
        ("ensemble", VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("svm", svm)],
            voting="soft",
            weights=[2, 2, 1],
        )),
    ])

    # MLP — two hidden layers sized to the feature count.
    # Color (6 features) and geometry (12 features) interact non-linearly;
    # a shallow net can capture cross-feature patterns the tree models miss.
    # Note: early_stopping is disabled — it triggers a sklearn bug when labels
    # are integer-encoded strings. max_iter=1000 is sufficient for this dataset size.
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=1000,
            random_state=42,
        )),
    ])

    # Calibrated SVM — plain SVC predict_proba uses Platt scaling which is
    # unreliable with small datasets. CalibratedClassifierCV with cv=3 refits
    # the calibration on held-out folds, giving trustworthy confidence scores
    # for the >80% pinyon threshold used in main.py.
    calibrated_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("cal_svm", CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=42),
            method="isotonic",      # isotonic is more flexible than sigmoid/Platt
            cv=3,
        )),
    ])

    return {
        "XGBoost":        xgb,
        "LightGBM":       lgbm,
        "SoftVoting":     voting,
        "MLP":            mlp,
        "CalibratedSVM":  calibrated_svm,
    }


# ── main entry point ──────────────────────────────────────────────────────────

def run_advanced_classifiers(df_deep, df_labels_matched, save_path=None):
    """
    Train and compare advanced species classifiers, return the best one.

    Identical call signature to train_tree_classifier() so it can be swapped in
    directly in main.py.

    Args:
        df_deep (pd.DataFrame):
            Feature dataframe from make_deep_dataframe(). Must contain all
            columns in FEATURES plus a "file" column.
        df_labels_matched (pd.DataFrame):
            Output of match_labels_to_clusters(). Must have "file" and "Name".
        save_path (str | None):
            Directory to write confusion matrix PNGs into. One PNG per model,
            plus a combined comparison plot. Pass None to skip. Default None.

    Returns:
        best_model: Fitted estimator (or Pipeline) with the highest macro F1
                    on the hold-out test set.
        features (list[str]): Feature columns used — same as FEATURES constant.
    """

    # ── merge and clean ───────────────────────────────────────────────────────
    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file")
    df = df.dropna(subset=["Name"])
    df = df[df["Name"] != "unknown"]

    print(f"\n{'═' * 60}")
    print("  ADVANCED CLASSIFIER COMPARISON")
    print(f"{'═' * 60}")
    print(f"  Labeled clusters: {len(df)}")

    class_counts = df["Name"].value_counts()
    print("  Class distribution:")
    for species, count in class_counts.items():
        print(f"    {species:<12} {count}")
    print()

    min_class = int(class_counts.min())
    if min_class < 10:
        print(f"  ⚠  Smallest class has only {min_class} samples — "
              f"CV scores will be noisy.\n")

    # keep as DataFrame so LightGBM retains feature names at predict time
    X_df = df[FEATURES]
    X    = X_df.values          # numpy array for models that don't need names
    y = df["Name"].values

    # ── label encoding for XGBoost (needs integer targets) ───────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)         # "juniper"→0, "pinyon"→1, "ponderosa"→2 etc.

    # ── class weights for XGBoost sample_weight ───────────────────────────────
    # inverse-frequency weighting: rarer species get higher weight
    class_freq   = class_counts / class_counts.sum()
    class_weight = {le.transform([sp])[0]: 1.0 / freq
                    for sp, freq in class_freq.items()}
    sample_weight = np.array([class_weight[label] for label in y_enc])

    # ── train / test split ────────────────────────────────────────────────────
    # track indices so LightGBM can receive DataFrame slices (preserves feature names)
    idx = np.arange(len(df))
    (X_train, X_test,
     y_train, y_test,
     y_train_enc, y_test_enc,
     sw_train, _,
     idx_train, idx_test) = train_test_split(
        X, y, y_enc, sample_weight, idx,
        test_size=0.2, random_state=42, stratify=y
    )

    # DataFrame slices for LightGBM — same rows as the numpy splits above
    X_df_train = X_df.iloc[idx_train]
    X_df_test  = X_df.iloc[idx_test]

    # ── CV setup ──────────────────────────────────────────────────────────────
    n_folds = min(3, min_class)
    cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # ── build and evaluate all models ─────────────────────────────────────────
    models         = build_models(class_weight)
    results        = []
    fitted_models  = {}

    for name, model in models.items():
        print(f"{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")

        # XGBoost — encoded integer labels + sample_weight
        if name == "XGBoost":
            model.fit(X_train, y_train_enc, sample_weight=sw_train)
            y_pred_enc = model.predict(X_test)
            y_pred     = le.inverse_transform(y_pred_enc)
            cv_scores  = cross_val_score(
                model, X, y_enc, cv=cv, scoring="f1_macro", n_jobs=-1
            )

        # LightGBM — pass DataFrame so feature names are preserved at predict time
        elif name == "LightGBM":
            model.fit(X_df_train, y_train)
            y_pred    = model.predict(X_df_test)
            cv_scores = cross_val_score(
                model, X_df, y, cv=cv, scoring="f1_macro", n_jobs=-1
            )

        # MLP — early_stopping with string labels triggers a sklearn bug;
        # encode to integers and decode predictions back to species names
        elif name == "MLP":
            model.fit(X_train, y_train_enc)
            y_pred_enc = model.predict(X_test)
            y_pred     = le.inverse_transform(y_pred_enc)
            cv_scores  = cross_val_score(
                model, X, y_enc, cv=cv, scoring="f1_macro", n_jobs=-1
            )

        # all other models (SoftVoting, CalibratedSVM) — string labels fine
        else:
            model.fit(X_train, y_train)
            y_pred    = model.predict(X_test)
            cv_scores = cross_val_score(
                model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1
            )

        # ── metrics ───────────────────────────────────────────────────────────
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa    = cohen_kappa_score(y_test, y_pred)

        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"  Macro F1:      {macro_f1:.3f}")
        print(f"  Cohen's Kappa: {kappa:.3f}")
        print(f"  CV Macro F1 ({n_folds}-fold): "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
              f"(folds: {', '.join(f'{s:.3f}' for s in cv_scores)})")
        print()

        results.append({
            "model":       name,
            "macro_f1":    round(macro_f1, 4),
            "kappa":       round(kappa, 4),
            "cv_macro_f1": round(cv_scores.mean(), 4),
            "cv_std":      round(cv_scores.std(), 4),
        })
        fitted_models[name] = (model, y_pred)

        # ── per-model confusion matrix PNG ────────────────────────────────────
        if save_path is not None:
            _save_confusion_matrix(
                y_test, y_pred,
                sorted(np.unique(y)),
                name, macro_f1, kappa,
                save_path
            )

    # ── comparison table ──────────────────────────────────────────────────────
    df_results = (
        pd.DataFrame(results)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n{'═' * 60}")
    print("  RESULTS  (sorted by test Macro F1)")
    print(f"{'═' * 60}")
    print(df_results.to_string(index=False))
    print()

    # ── comparison bar chart ──────────────────────────────────────────────────
    if save_path is not None:
        _save_comparison_chart(df_results, save_path)

    # ── best model ────────────────────────────────────────────────────────────
    best_name        = df_results.iloc[0]["model"]
    best_model, _    = fitted_models[best_name]

    print(f"  Best model: {best_name}  "
          f"(Macro F1={df_results.iloc[0]['macro_f1']:.3f}, "
          f"Kappa={df_results.iloc[0]['kappa']:.3f})")

    # ── feature importances if available ─────────────────────────────────────
    _print_feature_importances(best_model, best_name)

    # ── predict on all clusters (including unlabeled) ─────────────────────────
    if best_name == "XGBoost":
        df_deep["predicted_label"] = le.inverse_transform(
            best_model.predict(df_deep[FEATURES].values)
        )
        if hasattr(best_model, "predict_proba"):
            proba   = best_model.predict_proba(df_deep[FEATURES].values)
            classes = le.classes_
            for i, cls in enumerate(classes):
                df_deep[f"prob_{cls}"] = proba[:, i]

    elif best_name == "LightGBM":
        # LightGBM needs a DataFrame with matching feature names
        X_all = df_deep[FEATURES]
        df_deep["predicted_label"] = best_model.predict(X_all)
        if hasattr(best_model, "predict_proba"):
            proba   = best_model.predict_proba(X_all)
            classes = best_model.classes_
            for i, cls in enumerate(classes):
                df_deep[f"prob_{cls}"] = proba[:, i]

    elif best_name == "MLP":
        # MLP was trained on encoded labels — decode predictions back to strings
        df_deep["predicted_label"] = le.inverse_transform(
            best_model.predict(df_deep[FEATURES].values)
        )
        if hasattr(best_model, "predict_proba"):
            proba   = best_model.predict_proba(df_deep[FEATURES].values)
            classes = le.classes_
            for i, cls in enumerate(classes):
                df_deep[f"prob_{cls}"] = proba[:, i]

    else:
        # SoftVoting, CalibratedSVM — string labels throughout
        df_deep["predicted_label"] = best_model.predict(df_deep[FEATURES].values)
        if hasattr(best_model, "predict_proba"):
            proba   = best_model.predict_proba(df_deep[FEATURES].values)
            classes = best_model.classes_ if hasattr(best_model, "classes_") \
                      else _get_pipeline_classes(best_model)
            for i, cls in enumerate(classes):
                df_deep[f"prob_{cls}"] = proba[:, i]

    return best_model, FEATURES


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_confusion_matrix(y_true, y_pred, classes, model_name,
                            macro_f1, kappa, save_path):
    """Save a single confusion matrix PNG."""
    os.makedirs(save_path, exist_ok=True)
    cm  = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(
        f"{model_name}\n"
        f"Macro F1={macro_f1:.3f}   Kappa={kappa:.3f}"
    )
    plt.tight_layout()
    out = os.path.join(save_path, f"cm_{model_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {out}")


def _save_comparison_chart(df_results, save_path):
    """Save a grouped bar chart comparing macro F1 and kappa across models."""
    os.makedirs(save_path, exist_ok=True)

    x     = np.arange(len(df_results))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, df_results["macro_f1"], width, label="Macro F1",  color="#4C72B0")
    ax.bar(x + width / 2, df_results["kappa"],    width, label="Kappa",     color="#DD8452")

    # CV error bars on the macro F1 bars
    ax.errorbar(x - width / 2, df_results["macro_f1"],
                yerr=df_results["cv_std"],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(df_results["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Advanced Model Comparison\n(error bars = CV Macro F1 std)")
    ax.legend()
    ax.axhline(0.745, color="grey", linestyle="--", linewidth=0.8,
               label="baseline kappa")    # your current result
    ax.axhline(0.854, color="grey", linestyle=":",  linewidth=0.8,
               label="baseline F1")

    plt.tight_layout()
    out = os.path.join(save_path, "advanced_model_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Comparison chart saved → {out}")


def _print_feature_importances(model, name):
    """Print feature importances for tree-based models."""
    # unwrap Pipeline if needed
    estimator = model
    if hasattr(model, "named_steps"):
        last_step = list(model.named_steps.values())[-1]
        # VotingClassifier doesn't expose importances directly
        if hasattr(last_step, "feature_importances_"):
            estimator = last_step
        else:
            return

    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_, index=FEATURES)
        print(f"\n  Feature importances — {name}:")
        print(imp.sort_values(ascending=False).to_string())
        print()


def _get_pipeline_classes(pipeline):
    """Extract classes_ from the last step of a Pipeline."""
    last = list(pipeline.named_steps.values())[-1]
    if hasattr(last, "classes_"):
        return last.classes_
    # VotingClassifier stores classes_ on itself
    if hasattr(pipeline, "classes_"):
        return pipeline.classes_
    return []