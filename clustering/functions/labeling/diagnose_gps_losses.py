import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def _centroids(clusters, k):
    """XY centroids using the same density-weighted centre as clusters_to_dataframe(),
    so the stage-by-stage distances match what the real matcher sees."""
    # imported lazily: global_files is only on sys.path after run_clustering.py
    # inserts it, which happens after `from functions import *`
    from global_files.get_pointcloud_stats import density_weighted_center

    out = []
    for pcd in clusters:
        pts = np.asarray(pcd.points)
        if len(pts) < 3:
            out.append(pts[:, :2].mean(axis=0))
        else:
            out.append(density_weighted_center(pts, k=k)[:2])
    return np.array(out)


def diagnose_gps_losses(csv_coords, df_labels, stage_clusters, max_distance,
                        k=50, save_path=None):
    """
    For every GPS point, record the nearest-cluster distance (d_<stage>) and the
    number of clusters within max_distance (n_<stage>) after each pipeline stage.

    Adds per-point columns:
        final_status  perfect / no_match / multi (matches calculate_matching_score)
        lost_at       for no_match points: the first stage where no cluster was in
                      range any more ("never_<first stage>" if none ever was)
        multi_from    for multi points: the first stage with 2+ clusters in range

    Args:
        csv_coords (np.ndarray): (M, 2) UTM GPS coords AFTER the coverage filter.
        df_labels (pd.DataFrame): label frame aligned to csv_coords; needs "Name".
        stage_clusters (dict[str, list]): ordered {stage name: cluster list}.
            The last stage must be the final cluster list.
        max_distance (float): same value used for matching.
        k (int): neighbour count for the centroid, same as K in run_clustering.
        save_path (str | None): CSV path to write. None = print only.

    Returns:
        pd.DataFrame: one row per GPS point.
    """
    names = list(stage_clusters)
    df = pd.DataFrame({
        "gps_idx": np.arange(len(csv_coords)),
        "species": df_labels["Name"].values,
        "x": csv_coords[:, 0],
        "y": csv_coords[:, 1],
    })

    for name in names:
        cents = _centroids(stage_clusters[name], k)
        if len(cents) == 0:
            df[f"d_{name}"] = np.inf
            df[f"n_{name}"] = 0
            continue
        tree = KDTree(cents)
        dist, _ = tree.query(csv_coords)
        within = tree.query_ball_point(csv_coords, r=max_distance)
        df[f"d_{name}"] = dist.round(2)
        df[f"n_{name}"] = [len(m) for m in within]

    final = names[-1]
    df["final_status"] = np.select(
        [df[f"n_{final}"] == 0, df[f"n_{final}"] == 1],
        ["no_match", "perfect"], default="multi")

    def lost_at(row):
        if row["final_status"] != "no_match":
            return ""
        last_ok = None
        for i, name in enumerate(names):
            if row[f"n_{name}"] > 0:
                last_ok = i
        if last_ok is None:
            return f"never_{names[0]}"
        return names[last_ok + 1] if last_ok + 1 < len(names) else final

    def multi_from(row):
        if row["final_status"] != "multi":
            return ""
        for name in names:
            if row[f"n_{name}"] >= 2:
                return name
        return ""

    df["lost_at"] = df.apply(lost_at, axis=1)
    df["multi_from"] = df.apply(multi_from, axis=1)

    # ── summary ──
    print(f"\n{'═' * 60}\n  GPS LOSS DIAGNOSIS  (max_distance={max_distance} m)\n{'═' * 60}")
    print("  GPS points with 1+ cluster in range, per stage:")
    for name in names:
        print(f"    {name:<16} {(df[f'n_{name}'] > 0).sum():>4} / {len(df)}")

    nm = df[df["final_status"] == "no_match"]
    print(f"\n  Final no-match: {len(nm)}")
    if len(nm):
        print("  Lost at stage:")
        print(nm["lost_at"].value_counts().to_string())
        d0, d1 = nm[f"d_{names[0]}"], nm[f"d_{final}"]
        print(f"\n  Nearest-cluster distance for these points (m):")
        print(f"    at {names[0]}: median {d0.median():.1f}, "
              f"{(d0 <= max_distance + 2).sum()} within {max_distance + 2:.0f} m")
        print(f"    at {final}: median {d1.median():.1f}")
        print("  By species:")
        print(nm["species"].value_counts().to_string())

    mu = df[df["final_status"] == "multi"]
    print(f"\n  Final multi-match: {len(mu)}")
    if len(mu):
        print("  First stage with 2+ clusters in range:")
        print(mu["multi_from"].value_counts().to_string())
    print()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"  Saved → {save_path}\n")

    return df