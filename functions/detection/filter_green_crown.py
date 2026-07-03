import numpy as np
import open3d as o3d


def has_green_crown(pcd, top_fraction=0.20, min_exg=0.05):
    """
    Return True if the most-green points in a cluster are genuinely green.

    Ranks all points by their Excess Green index (ExG = 2g - r - b), takes
    the top top_fraction by greenness, and checks whether their mean ExG
    exceeds min_exg. A real tree's greenest points will clear this easily;
    rock or bare-ground clusters that slipped through the vegetation filter
    will have weak top-greenness even if a handful of points incidentally
    passed it.

    Ranking by greenness rather than by Z avoids any assumption about where
    the green points are spatially — the question being asked is simply
    "are the greenest points in this cluster actually green?", not "is the
    top of the cluster green?"

    Args:
        pcd (o3d.geometry.PointCloud): Cluster to evaluate. Must have colour
            data in normalised RGB (0.0–1.0) as produced by
            las_folder_to_pointcloud().
        top_fraction (float): Fraction of points to inspect, ranked by ExG
            descending. 0.20 means the greenest 20% of all points.
            Default 0.20.
        min_exg (float): Minimum mean ExG for the top fraction to pass.
            ExG = 2*g - r - b; range is roughly -2.0 to 2.0 in normalised
            RGB space. A single point needs ExG > 0.05 to pass the green
            dominance test in clean_up_pointcloud() (green > red + 0.025
            AND green > blue + 0.025 implies ExG > 0.05), so this threshold
            asks the mean of the top-fraction to at least clear that same
            bar. Real tree crowns typically average 0.10–0.25 in their top
            20%; non-vegetation clusters typically sit below 0.05.
            Default 0.05.

    Returns:
        bool: True if the cluster passes the green check, False otherwise.

    Requirements:
        numpy, open3d
    """
    if not pcd.has_colors():
        # no colour data at all — cannot evaluate, assume it passes so we
        # don't silently drop clusters from point clouds without RGB
        return True

    colors = np.asarray(pcd.colors)   # (N, 3), normalised 0.0–1.0

    if len(colors) < 5:
        return False

    # Excess Green index per point: ExG = 2g - r - b
    exg = 2 * colors[:, 1] - colors[:, 0] - colors[:, 2]

    # select the top top_fraction by ExG value
    # np.partition is O(n) vs O(n log n) for sort — no need to order the
    # top-k values, just retrieve them
    n_top   = max(1, int(len(exg) * top_fraction))
    top_exg = np.partition(exg, -n_top)[-n_top:]

    return float(top_exg.mean()) >= min_exg


def filter_clusters_by_green_crown(clusters, top_fraction=0.20, min_exg=0.05):
    """
    Remove non-vegetation clusters by checking the greenness of their most-
    green points.

    Applies has_green_crown() to every cluster and returns only those that
    pass. Intended to run after cluster_by_chm_peaks() or
    split_large_clusters() and before feature extraction, so false-positive
    clusters (rock patches, bare ground, sparse shrubs) don't pollute the
    feature dataframes.

    Args:
        clusters (list of o3d.geometry.PointCloud): Cluster list to filter.
        top_fraction (float): Passed to has_green_crown(). Fraction of
            points ranked by ExG to inspect. Default 0.20.
        min_exg (float): Passed to has_green_crown(). Minimum mean ExG of
            the top fraction. Default 0.05.

    Returns:
        list of o3d.geometry.PointCloud: Clusters that passed the filter.
            Order is preserved.

    Requirements:
        numpy, open3d
    """
    before   = len(clusters)
    filtered = [
        c for c in clusters
        if has_green_crown(c, top_fraction=top_fraction, min_exg=min_exg)
    ]
    removed  = before - len(filtered)

    print(f"Green crown filter: {before} → {len(filtered)} clusters "
          f"({removed} removed, top_fraction={top_fraction}, min_exg={min_exg})")

    return filtered