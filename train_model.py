"""
train_model.py
────────────────────────────────────────────────────────────────────────────────
Trains the species classifier on already-labeled clusters.

Reads df_clusters / df_deep_clusters straight from disk via load_dataframes()
— no point cloud, no clustering, no GPS matching. Run this whenever you want
to iterate on the model (feature selection, enhancements, advanced
classifiers, label spreading) without re-running run_clustering.py.

Requires run_clustering.py to have been run at least once for the given
trial (so PATHS['Dataframes'] exists and df_clusters already has the "Name"
column populated by match_labels_to_clusters).
"""

print("Loading packages...")

import argparse
from datetime import datetime

from constants import *
from functions import *

print("Successfully loaded all packages.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_name", type=str, required=True)
    parser.add_argument(
        "--advanced", action="store_true",
        help="Use run_advanced_classifiers() + label spreading instead of "
             "train_tree_classifier(). Replaces the old `if True: ... else: ...` "
             "toggle in main.py."
    )
    args = parser.parse_args()

    TRIAL_NAME = args.trial_name
    PATHS = get_paths(TRIAL_NAME)

    print()
    print("TRIAL_NAME:", TRIAL_NAME)
    print("Program Start Time: " + datetime.now().strftime("%H:%M:%S"))
    print()

    print("Reading in feature dataframes...")
    df_clusters, df_deep_clusters = load_dataframes(PATHS['Dataframes'])
    print("Dataframes read in.\n")

    if "Name" not in df_clusters.columns:
        print("df_clusters has no 'Name' column — run_clustering.py needs to "
              "run (and complete GPS label matching) before training.")
        return 1

    plot_feature_separability(
        df_deep_clusters,
        df_clusters,
        save_path=PATHS['Images']
    )

    if not args.advanced:
        model, features = train_tree_classifier(
            df_deep_clusters,
            df_clusters,
            save_confusion_matrix_path=PATHS['Images'] + 'confusion_matrix.png'
        )
    else:
        model, features = run_advanced_classifiers(
            df_deep_clusters,
            df_clusters,
            save_path=PATHS['Images']
        )

        print("Running semi-supervised label spreading...")
        df_deep_clusters = run_label_spreading(
            df_deep_clusters,
            df_clusters,
            save_path=PATHS['Images'],
            alpha=0.2,
            gamma=0.5,
            confidence_threshold=0.80
        )
        print("Label spreading complete.")

    if "prob_pinyon" in df_deep_clusters.columns:
        confirmed_pinyons = df_deep_clusters[
            (df_deep_clusters["predicted_label"] == "pinyon") &
            (df_deep_clusters["prob_pinyon"] > 0.80)
        ]
        print(f"High confidence pinyons: {len(confirmed_pinyons)}")

    print(df_deep_clusters[["file", "predicted_label"]])

    print("Program complete.")
    print("Time at Completion: " + datetime.now().strftime("%H:%M:%S"))


main()