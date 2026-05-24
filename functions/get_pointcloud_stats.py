# import libraries
import numpy as np
import pandas as pd
import open3d as o3d
import os
from scipy.spatial import KDTree


# a function to put all the pinyon stats into a df
def clusters_to_dataframe( clusters ):
    """ 
    Desc:
        Creates a .csv/spreadsheet of generic metrics on clusters to make some graphs. Takes a folder path with all the clusters.
    Args:
        clusters_folder, str: Folder path with all of the clusters (.ply) in it. make sure nothing else is there.

    Returns:
        df: A Pandas dataframe with info on all clusters in the folder.
        
    Requirements:
        numpy, pandas, open3d, os
    """

    rows = []      

    for i, cluster in enumerate(clusters):
        stats = get_pointcloud_stats( cluster )
        stats["file"] = i
        rows.append( stats )

    df = pd.DataFrame( rows )

    # sort by file name
    df = df.sort_values("file").reset_index(drop=True)

    return df

# a function to get stats from a pointcloud, returns array
def get_pointcloud_stats(pcd, silent=True):
    """ 
    Desc:
        Supplental function to clusters_to_dataframe. Creates a single row of cluster statistics.
    Args:
        pcd_path, str: location of the single cluster to read. Do not pass a cluster itself, just a location.

    Returns:
        stats: An array of stats for the given pointcloud. Wil luse to make final csv.
        
    Requirements:
        numpy, pandas, open3d, os
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

    center = density_weighted_center(points)

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
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k)
    # local density = inverse of mean distance to k neighbors
    density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
    weights = density / density.sum()
    return (points * weights[:, np.newaxis]).sum(axis=0)