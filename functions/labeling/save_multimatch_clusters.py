import os
import shutil
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def save_multimatch_clusters(csv_coords, cluster_coords, clusters, df_labels,
                              df_clusters, save_path, max_distance=3.0):
    """
    Save the clusters responsible for 'multiple clusters nearby' GPS matches.

    For every GPS label with two or more cluster centroids within
    max_distance (an ambiguous, non-1:1 match — see calculate_matching_score()),
    all nearby clusters are written to their own subfolder for visual
    inspection. Surfaces over-segmentation: one tagged tree producing several
    detected clusters, or two adjacent trees' clusters both falling within
    range of a single GPS tag.

    Subfolder naming: "<idx>_<species>/", e.g. "014_pinyon/". Each cluster
    inside is saved as "cluster_<file_idx>_<distance>m.ply", where file_idx
    is the index into df_clusters/clusters and distance is the GPS-to-
    centroid distance in metres.

    Args:
        csv_coords (np.ndarray): (M, 2) UTM XY GPS coordinates, same order
            as df_labels (post-coverage-filter array from
            match_labels_to_clusters()).
        cluster_coords (np.ndarray): (N, 2) UTM XY cluster centroids, same
            row order as df_clusters.
        clusters (list of o3d.geometry.PointCloud): Full cluster list —
            indexed via df_clusters["file"].
        df_labels (pd.DataFrame): GPS label DataFrame, same order as
            csv_coords. Must have a "Name" column.
        df_clusters (pd.DataFrame): Cluster DataFrame from
            clusters_to_dataframe(); row order must match cluster_coords.
            Must have a "file" column mapping to indices in `clusters`.
        save_path (str): Directory to write subfolders into. Wiped and
            recreated on every call.
        max_distance (float): Search radius in metres — pass the same value
            used for match_labels_to_clusters(). Default 3.0.

    Returns:
        int: Number of ambiguous GPS labels found (== folders written).

    Requirements:
        numpy, open3d, os, shutil, scipy.spatial.KDTree
    """

    tree = KDTree(cluster_coords)
    results = tree.query_ball_point(csv_coords, r=max_distance)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    n_ambiguous = 0

    for i, matches in enumerate(results):
        if len(matches) < 2:
            continue

        n_ambiguous += 1
        species = df_labels["Name"].iloc[i]
        folder  = os.path.join(save_path, f"{i:03d}_{species}")
        os.makedirs(folder, exist_ok=True)

        gps_xy = csv_coords[i]

        for row_idx in matches:
            file_idx = int(df_clusters.iloc[row_idx]["file"])
            if file_idx >= len(clusters):
                continue

            dist = np.linalg.norm(gps_xy - cluster_coords[row_idx])
            out_path = os.path.join(folder, f"cluster_{file_idx}_{dist:.2f}m.ply")
            o3d.io.write_point_cloud(out_path, clusters[file_idx])

    print(f"Multi-match diagnostic: {n_ambiguous} GPS labels had 2+ clusters "
          f"within {max_distance}m, saved to '{save_path}'")

    return n_ambiguous