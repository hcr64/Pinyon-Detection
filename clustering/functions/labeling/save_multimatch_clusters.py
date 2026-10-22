import os
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import KDTree


def save_multimatch_clusters(csv_coords, cluster_coords, clusters, df_labels,
                              df_clusters, save_path=None, max_distance=3.0,
                              stats_csv_path=None, verbose=True):
    """
    Report (and optionally save) the clusters behind 'multiple clusters
    nearby' GPS matches.

    For every GPS label with two or more cluster centroids within
    max_distance, prints each nearby cluster's stats (distance to the GPS
    point, n_points, height, radius, density), sorted nearest-first, and
    optionally writes the .ply files and a stats CSV.

    Args:
        csv_coords (np.ndarray): (M, 2) UTM XY GPS coordinates, same order
            as df_labels.
        cluster_coords (np.ndarray): (N, 2) UTM XY cluster centroids, same
            row order as df_clusters.
        clusters (list of o3d.geometry.PointCloud): Full cluster list,
            indexed via df_clusters["file"]. Only needed if save_path is set.
        df_labels (pd.DataFrame): GPS label DataFrame with a "Name" column.
        df_clusters (pd.DataFrame): Cluster DataFrame from
            clusters_to_dataframe(); needs "file" and, for the printout,
            "n_points", "height", "radius", "density".
        save_path (str | None): Directory for per-label .ply subfolders
            (wiped and recreated). None skips .ply writing. Default None.
        max_distance (float): Search radius in metres. Default 3.0.
        stats_csv_path (str | None): If given, writes one row per
            (ambiguous GPS label, nearby cluster) here. Default None.
        verbose (bool): Print the per-label stats. Default True.

    Returns:
        int: Number of ambiguous GPS labels found.

    Requirements:
        numpy, pandas, open3d, os, shutil, scipy.spatial.KDTree
    """

    tree = KDTree(cluster_coords)
    results = tree.query_ball_point(csv_coords, r=max_distance)

    if save_path is not None:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)

    stat_cols = [c for c in ("n_points", "height", "radius", "density")
                 if c in df_clusters.columns]

    rows = []
    n_ambiguous = 0

    for i, matches in enumerate(results):
        if len(matches) < 2:
            continue

        n_ambiguous += 1
        species = df_labels["Name"].iloc[i]
        gps_xy = csv_coords[i]

        # nearest first
        dists = {r: np.linalg.norm(gps_xy - cluster_coords[r]) for r in matches}
        ordered = sorted(matches, key=lambda r: dists[r])

        if verbose:
            print(f"\nGPS label {i:03d} ({species}) — {len(matches)} clusters "
                  f"within {max_distance}m:")

        folder = None
        if save_path is not None:
            folder = os.path.join(save_path, f"{i:03d}_{species}")
            os.makedirs(folder, exist_ok=True)

        for row_idx in ordered:
            row = df_clusters.iloc[row_idx]
            file_idx = int(row["file"])
            dist = dists[row_idx]

            rec = {"gps_idx": i, "species": species, "file": file_idx,
                   "distance_m": round(dist, 2)}
            for c in stat_cols:
                rec[c] = row[c]
            rows.append(rec)

            if verbose:
                stats_str = "  ".join(
                    f"{c}={row[c]:.0f}" if c == "n_points" else f"{c}={row[c]:.2f}"
                    for c in stat_cols
                )
                print(f"   cluster {file_idx:>5}  dist={dist:.2f}m  {stats_str}")

            if folder is not None and file_idx < len(clusters):
                out_path = os.path.join(folder, f"cluster_{file_idx}_{dist:.2f}m.ply")
                o3d.io.write_point_cloud(out_path, clusters[file_idx])

        # quick hint: a very lopsided n_points split suggests a fragment of
        # one tree; a similar split suggests two real crowns
        if verbose and "n_points" in stat_cols:
            npts = [df_clusters.iloc[r]["n_points"] for r in ordered]
            print(f"   → smallest/largest n_points ratio: {min(npts) / max(npts):.2f}")

    df_stats = pd.DataFrame(rows)

    if stats_csv_path is not None and len(df_stats):
        os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)
        df_stats.to_csv(stats_csv_path, index=False)
        print(f"Saved multi-match stats → {stats_csv_path}")

    where = f", .ply saved to '{save_path}'" if save_path else ""
    print(f"\nMulti-match diagnostic: {n_ambiguous} GPS labels had 2+ clusters "
          f"within {max_distance}m{where}")

    return n_ambiguous