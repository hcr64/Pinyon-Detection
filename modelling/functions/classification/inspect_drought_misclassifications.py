import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from functions.feature_config import FEATURES


def get_drought_out_of_fold_predictions(df_deep, df_labels_matched, features=None,
                                        random_state=42, n_folds=None):
    """
    Out-of-fold tolerant/susceptible predictions for every drought-labeled
    pinyon cluster, from a balanced RF (same baseline as
    inspect_misclassifications.get_out_of_fold_predictions).

    Returns:
        pd.DataFrame: "file", "true_label", "predicted_label", "correct".
    """
    df = df_deep.merge(
        df_labels_matched[["file", "Name", "drought_class", "drought_level",
                           "label_distance"]],
        on="file",
    )
    df = df[df["Name"] == "pinyon"].dropna(subset=["drought_class"]).reset_index(drop=True)

    available = [f for f in (features or FEATURES) if f in df.columns]
    X, y = df[available], df["drought_class"]

    min_class = int(y.value_counts().min())
    n_folds = n_folds or min(3, min_class)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=random_state)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    return pd.DataFrame({
        "file":            df["file"].values,
        "true_label":      y.values,
        "predicted_label": y_pred,
        "correct":         y.values == y_pred,
    })


def inspect_drought_misclassified_clusters(df_deep, df_labels_matched, features=None,
                                           n_seeds=5, n_folds=None,
                                           true_label_filter=None, save_path=None):
    """
    Record pinyon clusters whose drought class (tolerant/susceptible) is
    misclassified, with stability across n_seeds fold shufflings, the
    field drought_level, label_distance, reality-check flags, and features.

    Args:
        df_deep, df_labels_matched, features: as in inspect_misclassified_clusters().
            df_labels_matched must have "drought_class" and "drought_level".
        n_seeds (int): Number of fold shufflings. Default 5.
        true_label_filter ("tolerant" | "susceptible" | None): Restrict to one
            true class. Default None.
        save_path (str | None): CSV path to write. Default None.

    Returns:
        pd.DataFrame sorted by times_misclassified (empty if no errors).
    """
    print(f"\n{'═' * 60}")
    print(f"  DROUGHT MISCLASSIFICATION STABILITY CHECK  ({n_seeds} seeds)")
    print(f"{'═' * 60}")

    runs = []
    for seed in range(n_seeds):
        run = get_drought_out_of_fold_predictions(
            df_deep, df_labels_matched, features=features,
            random_state=seed, n_folds=n_folds,
        )
        run["seed"] = seed
        runs.append(run)
        print(f"  seed={seed}: {(~run['correct']).sum()} misclassified / {len(run)} labeled")

    df_all = pd.concat(runs, ignore_index=True)
    wrong = df_all[~df_all["correct"]]
    if true_label_filter is not None:
        wrong = wrong[wrong["true_label"] == true_label_filter]

    if len(wrong) == 0:
        print("\n  No drought misclassifications found.\n")
        return pd.DataFrame()

    rows = []
    for file_idx, g in wrong.groupby("file"):
        rows.append({
            "file":                    int(file_idx),
            "true_label":              g["true_label"].iloc[0],
            "most_common_wrong_label": g["predicted_label"].value_counts().index[0],
            "times_misclassified":     len(g),
            "n_seeds":                 n_seeds,
        })
    df_summary = (pd.DataFrame(rows)
                  .sort_values("times_misclassified", ascending=False)
                  .reset_index(drop=True))

    available = [f for f in (features or FEATURES) if f in df_deep.columns]
    df_context = df_deep[["file"] + available].merge(
        df_labels_matched[["file", "drought_level", "label_distance"]],
        on="file", how="left",
    )
    df_report = df_summary.merge(df_context, on="file", how="left")

    if "n_points" in df_report.columns:
        df_report["flag_low_points"] = df_report["n_points"] < 20
    if "height" in df_report.columns:
        df_report["flag_low_height"] = df_report["height"] < 0.5
    if "height_to_radius" in df_report.columns:
        q99 = df_deep["height_to_radius"].quantile(0.99)
        df_report["flag_extreme_ratio"] = df_report["height_to_radius"] > q99

    display = ["file", "true_label", "most_common_wrong_label",
               "times_misclassified", "drought_level", "label_distance"]
    display += [c for c in ("n_points", "height", "radius") if c in df_report.columns]
    display += [c for c in df_report.columns if c.startswith("flag_")]

    print(f"\n  {len(df_report)} unique pinyon cluster(s) misclassified at least once:\n")
    print(df_report[display].to_string(index=False))
    print()

    stable = df_report[df_report["times_misclassified"] == n_seeds]
    if len(stable):
        print(f"  ⚠  {len(stable)} misclassified in EVERY seed: {stable['file'].tolist()}")
    print()

    if save_path is not None:
        df_report.to_csv(save_path, index=False)
        print(f"  Saved → {save_path}\n")

    return df_report