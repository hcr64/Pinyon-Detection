import numpy as np
import pandas as pd
import open3d as o3d
import os


def make_deep_dataframe( clusters ):
    """
    Build a DataFrame of shape, PCA, and colour features from a cluster list.
 
    Each row corresponds to one cluster. The "file" column stores the integer
    index so rows can be joined to df_clusters after label matching. This
    DataFrame is used as the feature matrix for train_tree_classifier().
 
    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to featurise.
 
    Returns:
        pd.DataFrame: One row per cluster with columns produced by
            get_deep_cluster_features() plus a "file" index column.
 
    Requirements:
        numpy, pandas, open3d
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
    Extract shape, PCA, and colour features from a single cluster.
 
    Used internally by make_deep_dataframe(). Features are chosen to
    discriminate between pinyon, juniper, and ponderosa pine based on crown
    geometry and colour:
 
    Shape features:
        height       — Z range of the axis-aligned bounding box
        radius       — max XY half-width of the AABB
        n_points     — total point count
        obb_extent_* — oriented bounding box dimensions (x, y, z);
                       rotation-aware so they capture true crown shape
                       regardless of scan orientation
 
    PCA features (eigenvalues of the 3×3 covariance matrix):
        eigenvalue_1/2/3 — raw eigenvalues (descending order)
        linearity        — (λ1 − λ2) / λ1  — how rod-like the cluster is
        planarity        — (λ2 − λ3) / λ1  — how flat/disc-like
        sphericity       — λ3 / λ1          — how spherical / isotropic
 
    Colour features (normalised RGB, 0.0–1.0):
        mean_r/g/b  — average colour per channel
        std_r/g/b   — colour variation per channel
 
    Args:
        pcd (o3d.geometry.PointCloud): A single cluster with colour data.
 
    Returns:
        dict: Feature name → scalar value for all features listed above.
 
    Requirements:
        numpy, open3d
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