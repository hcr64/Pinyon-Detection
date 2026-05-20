import numpy as np
import pandas as pd
import open3d as o3d
import os


def make_deep_dataframe( clusters ):
    """ 
    Desc:
        Creates a .csv/spreadsheet of advanced metrics on clusters to train models on. Takes a folder path with all the clusters.
    Args:
        clusters, list of pcd: A list with clusters (pointclouds) in it. Stats are gathered from these pcds.

    Returns:
        df_deep: A Pandas dataframe with info on all clusters in the list.
        
    Requirements:
        numpy, pandas, open3d, os, make_deep_dataframe
    """

    # build dataframe from clusters
    rows = []

    for i, cluster in enumerate( clusters ):
        features = get_deep_cluster_features( cluster )
        features["file"] = i
        rows.append(features)


    df_deep = pd.DataFrame(rows)

    return df_deep

# support function
def get_deep_cluster_features(pcd):
    """ 
    Desc:
        Supplemental function to make_deep_dataframe. Creates an dict of stats for a single cluster. Will be a row in the final csv.
    Args:
        pcd, Open3D pointcloud: A single cluster from the list (open3D object).

    Returns:
        features: Array of features on the given cluster.
        
    Requirements:
        numpy, pandas, open3d, os
    """

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # geometry features
    aabb      = pcd.get_axis_aligned_bounding_box()
    obb       = pcd.get_oriented_bounding_box()  # rotation-aware bounding box
    center    = aabb.get_center()

    # PCA -- captures the shape/orientation of the cluster
    cov       = np.cov(points.T)
    eigvals   = np.linalg.eigvals(cov)
    eigvals   = np.sort(eigvals)[::-1]  # sort descending

    # color features
    mean_color = colors.mean(axis=0)  # average RGB
    std_color  = colors.std(axis=0)   # color variation

    features = {
        # shape
        "height":            (aabb.max_bound - aabb.min_bound)[2],
        "radius":            max((aabb.max_bound - aabb.min_bound)[:2]) / 2,
        "n_points":          len(points),
        "obb_extent_x":      obb.extent[0],  # oriented bounding box dimensions
        "obb_extent_y":      obb.extent[1],
        "obb_extent_z":      obb.extent[2],

        # PCA eigenvalues -- capture how flat/round/elongated the cluster is
        "eigenvalue_1":      eigvals[0],
        "eigenvalue_2":      eigvals[1],
        "eigenvalue_3":      eigvals[2],
        "linearity":         (eigvals[0] - eigvals[1]) / eigvals[0],
        "planarity":         (eigvals[1] - eigvals[2]) / eigvals[0],
        "sphericity":        eigvals[2] / eigvals[0],

        # color
        "mean_r":            mean_color[0],
        "mean_g":            mean_color[1],
        "mean_b":            mean_color[2],
        "std_r":             std_color[0],
        "std_g":             std_color[1],
        "std_b":             std_color[2],
    }

    return features