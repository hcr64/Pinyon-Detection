"""
train_model.py
────────────────────────────────────────────────────────────────────────────────
Trains the species classifier on already-labeled clusters.

Reads df_clusters / df_deep_clusters straight from disk via load_dataframes()
— no CHM, no watershed, no GPS matching. Run this whenever you want to
iterate on the model (feature selection, enhancements, advanced classifiers,
label spreading, embeddings) without re-running run_clustering.py.

Requires run_clustering.py to have been run at least once for the given
trial (so PATHS['Dataframes'] exists and df_clusters already has the "Name"
column populated by match_labels_to_clusters).

Why modelling/functions/__init__.py imports from clustering.functions.io
─────────────────────────────────────────────────────────────────────────────
save_clusters / load_clusters / save_dataframes / load_dataframes all live in
clustering/functions/io/save_clusters.py, not anywhere under modelling/,
because run_clustering.py is what originally writes clusters and dataframes
to disk — modelling/ only ever needed to read the CACHED CSV dataframes
back via load_dataframes(), so historically train_model.py never touched
the raw .ply clusters at all.

That changed with pointcloud_autoencoder.py (--embeddings below): its
extract_embeddings() needs the actual point clouds (o3d.geometry.PointCloud
objects), not the flattened geometry columns in df_clusters/df_deep_clusters.
Rather than duplicate load_clusters() as a second copy inside modelling/,
modelling/functions/__init__.py reaches across and imports the existing one
from clustering.functions.io — hence the sys.path.insert() + cross-package
import there. Side effect worth knowing: importing modelling.functions now
transitively imports the entire clustering.functions package (rasterio,
scikit-image, pyproj, etc.) even when --embeddings isn't used, since Python
has no way to import "just load_clusters" without running
clustering/functions/__init__.py in full first.
"""

print("Loading packages...")

import argparse
import os
from datetime import datetime

# import global files, includes constants
# include the dataframe making files
import sys
sys.path.insert(0, "/home/hcr64/Pinyon-Detection")

from global_files import * 

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
    parser.add_argument(
        "--embeddings", action="store_true",
        help="Train (or load, if a checkpoint already exists) a PointNet "
             "autoencoder over the raw clusters and merge its embedding "
             "columns onto df_deep_clusters before classification. See "
             "pointcloud_autoencoder.py for why this exists instead of "
             "PointMLP/PointNeXt/Point Transformer/KPConv."
    )
    parser.add_argument(
        "--retrain_embeddings", action="store_true",
        help="Force retraining the autoencoder even if a checkpoint already "
             "exists at PATHS['Models']. Ignored unless --embeddings is set."
    )
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--embedding_n_points", type=int, default=256)
    parser.add_argument("--embedding_epochs", type=int, default=100)
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

    # ── optional: point cloud embeddings ──────────────────────────────────
    # Only reload the raw .ply clusters if actually needed — this is the
    # one thing in this script that reaches past the cached dataframes,
    # and it's exactly why modelling/functions/__init__.py imports
    # load_clusters from clustering.functions.io (see module docstring).
    if args.embeddings:
        checkpoint_path = os.path.join(PATHS['Models'], 'pointnet_ae.pt')

        print("Loading raw clusters for embedding extraction...")
        clusters = load_clusters(PATHS['Clusters'])
        print(f"Loaded {len(clusters)} clusters.\n")

        if os.path.exists(checkpoint_path) and not args.retrain_embeddings:
            print(f"Found existing autoencoder checkpoint, loading instead "
                  f"of retraining (pass --retrain_embeddings to force)...")
            encoder, n_points = load_autoencoder(checkpoint_path)
        else:
            print("Training PointNet autoencoder on all clusters "
                  "(labeled + unlabeled, self-supervised, no labels used)...")
            encoder, _ = train_autoencoder(
                clusters,
                n_points=args.embedding_n_points,
                embedding_dim=args.embedding_dim,
                epochs=args.embedding_epochs,
                save_path=checkpoint_path,
            )
            n_points = args.embedding_n_points

        print("Extracting embeddings...")
        df_embeddings = extract_embeddings(clusters, encoder, n_points=n_points)

        before_cols = set(df_deep_clusters.columns)
        df_deep_clusters = df_deep_clusters.merge(df_embeddings, on="file")

        new_cols = sorted(set(df_deep_clusters.columns) - before_cols)
        print(f"Merged {len(new_cols)} embedding columns onto "
              f"df_deep_clusters: {new_cols[0]} ... {new_cols[-1]}\n")

        # NOTE: emb_0 ... emb_<embedding_dim-1> won't be picked up by
        # train_tree_classifier.py / advanced_classifiers.py / semi_supervised.py
        # automatically — each defines its own hardcoded FEATURES list (see the
        # "FEATURES is duplicated, not shared" quirk in modelling/README.md).
        # Add the emb_* columns to whichever FEATURES list(s) you want them
        # considered by, or rely on ENHANCEMENTS["SELECTION"] in
        # train_tree_classifier.py to drop them automatically if they don't
        # clear mean RF importance.

    plot_feature_separability(
        df_deep_clusters,
        df_clusters,
        save_path=PATHS['Images']
    )

    if not args.advanced:
        model, features, *_ = train_tree_classifier(
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

        # evaluate semi supervised model
        from functions.classification.evaluate_semi_supervised import run_full_evaluation
        run_full_evaluation(
            df_deep_clusters,
            df_clusters,
            save_path=PATHS['Images'],
            alpha=0.2,
            gamma=0.5,
            confidence_threshold=0.80,
        )

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