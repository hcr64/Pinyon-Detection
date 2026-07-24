# import libraries
import numpy as np
import pandas as pd
import open3d as o3d
import os
from scipy.spatial import KDTree


# a function to put all the pinyon stats into a df
def clusters_to_dataframe( clusters, k=50 ):
    """
    Build a DataFrame of geometry and position statistics from a cluster list.
 
    Each row corresponds to one cluster. The "file" column stores the
    cluster's integer index in the input list so it can be joined back to
    df_deep_clusters and df_labels after label matching.
 
    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to summarise.
        k (int): Number of neighbours used by density_weighted_center() when
            computing the XY centroid. Default 50.
 
    Returns:
        pd.DataFrame: One row per cluster with columns: n_points, height,
            width_x, width_y, radius, x_pos, y_pos, density, file.
 
    Requirements:
        numpy, pandas, open3d, os, scipy.spatial.KDTree
    """

    rows = []      

    for i, cluster in enumerate(clusters):
        stats = get_pointcloud_stats( cluster, k=k )
        stats["file"] = i
        rows.append( stats )

    df = pd.DataFrame( rows )

    # sort by file name
    df = df.sort_values("file").reset_index(drop=True)

    return df

# a function to get stats from a pointcloud, returns array
def get_pointcloud_stats(pcd, silent=True, k=50):
    """
    Compute geometry and position statistics for a single cluster.
 
    Used internally by clusters_to_dataframe(). The XY centroid is computed
    with density_weighted_center() rather than the AABB centre so that sparse
    crown edges do not pull the centroid away from the trunk.
 
    Args:
        pcd (o3d.geometry.PointCloud): A single cluster.
        silent (bool): Suppress per-stat print output. Default True.
        k (int): Neighbour count for density_weighted_center(). Default 50.
 
    Returns:
        dict: Keys: n_points, height, width_x, width_y, radius, x_pos,
            y_pos, density.
 
    Requirements:
        numpy, pandas, open3d, scipy.spatial.KDTree
    """

    # read the pointcloud
    # pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound  # [x, y, z]
    max_bound = aabb.max_bound

    # height (z axis)
    height = max_bound[2] - min_bound[2]

    # width/radius (x and y axes)
    width_x = max_bound[0] - min_bound[0]
    width_y = max_bound[1] - min_bound[1]
    radius = max(width_x, width_y) / 2

    # center position (x, y)
    # center = aabb.get_center()

    # get the center based off all the points 
    center = density_weighted_center(points, k=k)
    x_pos = center[0]
    y_pos = center[1]

    # number of points
    n_points = len(points)

    # point density (points per cubic meter)
    volume = width_x * width_y * height
    density = n_points / volume if volume > 0 else 0

    stats = {
        "n_points":  n_points,
        "height":    round(height, 3),
        "width_x":   round(width_x, 3),
        "width_y":   round(width_y, 3),
        "radius":    round(radius, 3),
        "x_pos":     round(x_pos, 3),
        "y_pos":     round(y_pos, 3),
        "density":   round(density, 3),
    }
    if not silent:
      for k, v in stats.items():
          print(f"{k}: {v}")

    return stats

# get center values wewighted on the amount of points, to reduce impacts of grass, bushes, etc.
def density_weighted_center(points, k=10):
    """
    Compute a density-weighted centroid for a point array.
 
    Weights each point by its local density (inverse mean distance to k
    neighbours). Denser regions — such as the trunk core — pull the centre
    more strongly than sparse crown edges. This gives a more stable XY
    position estimate for GPS label matching than the AABB centre.
 
    Args:
        points (np.ndarray): (N, 3) XYZ array.
        k (int): Neighbour count for local density estimation. Capped at
            len(points) - 1. Default 10.
 
    Returns:
        np.ndarray: (3,) weighted centroid [x, y, z].
 
    Requirements:
        numpy, scipy.spatial.KDTree
    """
    
    # cap k so it doenst go over the clsuter size
    k = min(k, len(points) - 1)

    tree = KDTree(points)
    distances, _ = tree.query(points, k=k)
    # local density = inverse of mean distance to k neighbors
    density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
    weights = density / density.sum()
    return (points * weights[:, np.newaxis]).sum(axis=0)


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