import open3d as o3d
import pandas as pd
import os


def save_clusters(clusters, save_path):
    """
    Save a list of point cloud clusters to disk as individual .ply files.

    Clears the destination folder before writing so it always reflects the
    current cluster list. Files are named cluster0.ply, cluster1.ply, …

    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to save.
        save_path (str): Directory to write .ply files into. Created
            automatically if it does not exist.

    Returns:
        None

    Requirements:
        open3d, os
    """

    # delete everything in the folder before saving
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))
    else:
        os.makedirs(save_path)

    for index in range(len(clusters)):
        o3d.io.write_point_cloud(f"{save_path}cluster{index}.ply", clusters[index])

    print(f"All {len(clusters)} clusters saved to {save_path}")


def load_clusters(save_path):
    """
    Load all .ply cluster files from a directory into a list of PointClouds.

    Args:
        save_path (str): Directory containing .ply cluster files written by
            save_clusters().

    Returns:
        list of o3d.geometry.PointCloud: One entry per .ply file found,
            sorted by the integer index embedded in the filename so the order
            matches the original cluster list.

    Requirements:
        open3d, os
    """

    clusters = []

    ply_files = [f for f in os.listdir(save_path) if f.endswith(".ply")]

    # sort by embedded integer so cluster10 doesn't come before cluster2
    def _cluster_index(filename):
        name = os.path.splitext(filename)[0]          # e.g. "cluster7"
        digits = "".join(c for c in name if c.isdigit())
        return int(digits) if digits else 0

    ply_files.sort(key=_cluster_index)

    for file in ply_files:
        pcd = o3d.io.read_point_cloud(os.path.join(save_path, file))
        clusters.append(pcd)

    return clusters


def save_dataframes(df_clusters, df_deep_clusters, save_path):
    """
    Save df_clusters and df_deep_clusters to disk as CSV files.

    Writes two files into save_path:
        clusters.csv      — geometry / position DataFrame from clusters_to_dataframe()
        deep_clusters.csv — shape / colour / PCA DataFrame from make_deep_dataframe()

    Avoids re-running the (expensive) feature extraction stages on subsequent
    runs. Load back with load_dataframes().

    Args:
        df_clusters (pd.DataFrame): Output of clusters_to_dataframe().
        df_deep_clusters (pd.DataFrame): Output of make_deep_dataframe() and,
            optionally, engineer_features().
        save_path (str): Directory to write CSV files into. Created
            automatically if it does not exist.

    Returns:
        None

    Requirements:
        pandas, os
    """

    os.makedirs(save_path, exist_ok=True)

    clusters_path      = os.path.join(save_path, "clusters.csv")
    deep_clusters_path = os.path.join(save_path, "deep_clusters.csv")

    df_clusters.to_csv(clusters_path, index=False)
    df_deep_clusters.to_csv(deep_clusters_path, index=False)

    print(f"Saved df_clusters      ({len(df_clusters)} rows) → {clusters_path}")
    print(f"Saved df_deep_clusters ({len(df_deep_clusters)} rows) → {deep_clusters_path}")


def load_dataframes(save_path):
    """
    Load df_clusters and df_deep_clusters previously saved by save_dataframes().

    Args:
        save_path (str): Directory containing clusters.csv and
            deep_clusters.csv written by save_dataframes().

    Returns:
        df_clusters (pd.DataFrame): Geometry / position DataFrame.
        df_deep_clusters (pd.DataFrame): Shape / colour / PCA DataFrame.

    Raises:
        FileNotFoundError: If either expected CSV is absent from save_path.

    Requirements:
        pandas, os
    """

    clusters_path      = os.path.join(save_path, "clusters.csv")
    deep_clusters_path = os.path.join(save_path, "deep_clusters.csv")

    for path in (clusters_path, deep_clusters_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"load_dataframes: expected file not found: {path}\n"
                f"Run save_dataframes() first, or set STEPS['Make_Clusters'] = True."
            )

    df_clusters      = pd.read_csv(clusters_path)
    df_deep_clusters = pd.read_csv(deep_clusters_path)

    print(f"Loaded df_clusters      ({len(df_clusters)} rows) ← {clusters_path}")
    print(f"Loaded df_deep_clusters ({len(df_deep_clusters)} rows) ← {deep_clusters_path}")

    return df_clusters, df_deep_clusters