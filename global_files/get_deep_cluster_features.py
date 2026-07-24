import numpy as np
import pandas as pd
import open3d as o3d
import os


def make_deep_dataframe(clusters):
    """
    Build a DataFrame of shape, PCA, colour, and derived features from a
    cluster list.

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
    rows = []

    for i, cluster in enumerate(clusters):
        features = get_deep_cluster_features(cluster)
        features["file"] = i
        rows.append(features)

    df_deep = pd.DataFrame(rows)
    return df_deep


def get_deep_cluster_features(pcd):
    """
    Extract shape, PCA, colour, and structural features from a single cluster.

    Used internally by make_deep_dataframe(). Features are chosen to
    discriminate between pinyon, juniper, and ponderosa pine based on crown
    geometry, point distribution, and colour.

    Shape features:
        height       — Z range of the axis-aligned bounding box
        radius       — max XY half-width of the AABB
        n_points     — total point count
        obb_extent_* — oriented bounding box dimensions (x, y, z)

    PCA features (eigenvalues of the 3x3 covariance matrix):
        eigenvalue_1/2/3 — raw eigenvalues (descending order)
        linearity        — (λ1 - λ2) / λ1
        planarity        — (λ2 - λ3) / λ1
        sphericity       — λ3 / λ1

    Colour features (normalised RGB, 0.0-1.0):
        mean_r/g/b  — average colour per channel
        std_r/g/b   — colour variation per channel

    Structural features (new — target species-specific crown architecture):
        verticality      — height / max horizontal OBB extent; ponderosa is
                           tall and columnar, pinyon is wide and squat
        flatness_ratio   — planarity / height; juniper sprawls flat relative
                           to its height, ponderosa does not
        crown_base_ratio — point density in upper third of crown vs lower
                           third; ponderosa has a high canopy with a sparse
                           lower trunk, pinyon fills more evenly

    Args:
        pcd (o3d.geometry.PointCloud): A single cluster with colour data.

    Returns:
        dict: Feature name -> scalar value for all features listed above.

    Requirements:
        numpy, open3d
    """

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # ── bounding boxes ────────────────────────────────────────────────────────
    aabb = pcd.get_axis_aligned_bounding_box()
    obb  = pcd.get_oriented_bounding_box()

    bounds = aabb.max_bound - aabb.min_bound
    height = bounds[2]
    radius = max(bounds[:2]) / 2

    # ── PCA ───────────────────────────────────────────────────────────────────
    cov     = np.cov(points.T)
    eigvals = np.linalg.eigvals(cov)
    eigvals = np.sort(np.abs(eigvals))[::-1]   # descending, guard against tiny negatives

    λ1, λ2, λ3 = eigvals[0], eigvals[1], eigvals[2]
    linearity  = (λ1 - λ2) / (λ1 + 1e-6)
    planarity  = (λ2 - λ3) / (λ1 + 1e-6)
    sphericity = λ3 / (λ1 + 1e-6)

    # ── colour ────────────────────────────────────────────────────────────────
    mean_color = colors.mean(axis=0)
    std_color  = colors.std(axis=0)

    # ── structural features ───────────────────────────────────────────────────

    # verticality: ponderosa is tall relative to its horizontal footprint;
    # pinyon is wide and squat (low ratio); juniper is intermediate
    max_horiz_extent = max(obb.extent[0], obb.extent[1])
    verticality = height / (max_horiz_extent + 1e-6)

    # flatness_ratio: juniper sprawls laterally relative to its height —
    # high planarity AND low height gives a high ratio
    flatness_ratio = planarity / (height + 1e-6)

    # crown_base_ratio: split the cluster into vertical thirds by Z value.
    # ponderosa has dense upper canopy and sparse lower trunk, so upper/lower
    # density is high. Pinyon and juniper fill more evenly top-to-bottom.
    z         = points[:, 2]
    z_min     = z.min()
    z_max     = z.max()
    z_range   = z_max - z_min + 1e-6

    lower_mask = z < (z_min + z_range / 3)
    upper_mask = z > (z_max - z_range / 3)

    n_lower = lower_mask.sum()
    n_upper = upper_mask.sum()

    # density proxy: points per unit height in that third
    lower_density = n_lower / (z_range / 3 + 1e-6)
    upper_density = n_upper / (z_range / 3 + 1e-6)
    crown_base_ratio = upper_density / (lower_density + 1e-6)

    # ── assemble feature dict ─────────────────────────────────────────────────
    features = {
        # shape
        "height":           height,
        "radius":           radius,
        "n_points":         len(points),
        "obb_extent_x":     obb.extent[0],
        "obb_extent_y":     obb.extent[1],
        "obb_extent_z":     obb.extent[2],

        # PCA
        "eigenvalue_1":     λ1,
        "eigenvalue_2":     λ2,
        "eigenvalue_3":     λ3,
        "linearity":        linearity,
        "planarity":        planarity,
        "sphericity":       sphericity,

        # colour
        "mean_r":           mean_color[0],
        "mean_g":           mean_color[1],
        "mean_b":           mean_color[2],
        "std_r":            std_color[0],
        "std_g":            std_color[1],
        "std_b":            std_color[2],

        # structural (new)
        "verticality":      verticality,
        "flatness_ratio":   flatness_ratio,
        "crown_base_ratio": crown_base_ratio,
    }

    return features


def engineer_features(df):
    """
    Add derived ratio features to the deep cluster DataFrame.

    These are computed from existing columns rather than raw point cloud
    geometry, so they are cheaper to recompute and easier to iterate on than
    the structural features in get_deep_cluster_features(). Call this after
    make_deep_dataframe() and before train_tree_classifier().

    New columns added:
        height_to_radius — ponderosa tends tall+narrow (high), pinyon
                           short+wide (low)
        green_dominance  — mean green minus average of red and blue; juniper
                           tends lower than pinyon
        crown_volume     — radius² × height; proxy for total canopy mass
        color_saturation — sum of per-channel std; ponderosa bark pulls
                           std_r higher

    Args:
        df (pd.DataFrame): Output of make_deep_dataframe().

    Returns:
        pd.DataFrame: Same DataFrame with four additional columns appended.

    Requirements:
        pandas
    """
    df = df.copy()

    df["height_to_radius"] = df["height"] / (df["radius"] + 1e-6)
    df["green_dominance"]  = df["mean_g"] - (df["mean_r"] + df["mean_b"]) / 2
    df["crown_volume"]     = df["radius"] ** 2 * df["height"]
    df["color_saturation"] = df["std_r"] + df["std_g"] + df["std_b"]

    return df