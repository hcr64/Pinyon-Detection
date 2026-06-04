import numpy as np
import open3d as o3d


def strip_ground_from_clusters(clusters, ground_percentile=10, min_height_above_ground=0.5):
    """
    Desc:
        Removes ground-level points from each cluster by estimating the local
        ground elevation as a low percentile of Z values within the cluster,
        then discarding all points below (ground + min_height_above_ground).

        This is done per-cluster rather than globally so that clusters on
        sloped terrain (e.g. the Sunset Crater cinder cone) each get their
        own local ground reference, rather than a single global Z cutoff which
        would incorrectly strip canopy points from clusters on high ground.

    Args:
        clusters, list of o3d.PointCloud:
            Input clusters, typically from cluster_by_chm_peaks() or after
            split_large_clusters().
        ground_percentile, int:
            Percentile of Z values within each cluster used to estimate the
            local ground level. Lower values are more conservative (keep more
            points). Default 10 — the 10th percentile Z is a safe proxy for
            the ground surface under the crown. Raise to 20-25 if clusters
            still look like they have a lot of ground in them.
        min_height_above_ground, float:
            Points must be at least this many metres above the estimated ground
            level to be kept. Default 0.5 m — removes grass, litter, and low
            shrubs under the canopy without eating into the lower trunk.
            Raise to 1.0-1.5 if you want to keep only upper canopy.

    Returns:
        stripped_clusters, list of o3d.PointCloud:
            Clusters with ground points removed. Clusters that lose too many
            points to pass a basic sanity check (< 5 points remaining) are
            dropped entirely.

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