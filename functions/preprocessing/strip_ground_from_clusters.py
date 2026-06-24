import numpy as np
import open3d as o3d


def strip_ground_from_clusters(clusters, ground_percentile=10, min_height_above_ground=0.5):
    """
    Remove ground-level points from each cluster using a per-cluster Z threshold.
 
    Estimates the local ground elevation inside each cluster as the
    ground_percentile-th percentile of its Z values, then discards all points
    below ground + min_height_above_ground. This is done per-cluster — not
    globally — so that trees on sloped terrain (e.g. Sunset Crater's cinder
    cone) each get their own ground reference rather than a single Z cutoff
    that would incorrectly strip canopy points from trees on high ground.
 
    Clusters that have fewer than 5 points remaining after stripping are
    dropped entirely.
 
    Args:
        clusters (list of o3d.geometry.PointCloud): Input clusters from
            cluster_by_chm_peaks() or split_large_clusters().
        ground_percentile (int): Percentile of Z used as the ground proxy.
            Lower values are more conservative (keep more points). Raise to
            20-25 if clusters still contain obvious ground returns.
            Default 10.
        min_height_above_ground (float): Minimum height above the estimated
            ground for a point to be kept (metres). Removes grass, litter,
            and low shrubs without eating into the lower trunk. Default 0.5.
 
    Returns:
        list of o3d.geometry.PointCloud: Ground-stripped clusters. Clusters
            reduced below 5 points are excluded.
 
    Requirements:
        numpy, open3d
    """

    stripped_clusters = []

    for pcd in clusters:
        points = np.asarray(pcd.points)

        if len(points) < 5:
            continue

        # estimate local ground as a low percentile of Z
        ground_z = np.percentile(points[:, 2], ground_percentile)

        # keep only points sufficiently above that ground estimate
        mask = points[:, 2] > ground_z + min_height_above_ground

        if mask.sum() < 5:
            continue

        stripped = pcd.select_by_index(np.where(mask)[0])
        stripped_clusters.append(stripped)

    n_dropped = len(clusters) - len(stripped_clusters)
    print(f"Ground stripping: {len(clusters)} → {len(stripped_clusters)} clusters "
          f"({n_dropped} dropped as too small after stripping)")

    return stripped_clusters