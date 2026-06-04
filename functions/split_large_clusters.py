import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import MeanShift, estimate_bandwidth


def filter_cluster(pcd, min_height=1.0, min_radius=0.3):
    """
    Reject clusters that are too flat or too narrow to be trees.
    Grass patches tend to be wide but very short.
    Isolated shrubs tend to be small in all dimensions.
    """
    points = np.asarray(pcd.points)
    aabb   = pcd.get_axis_aligned_bounding_box()
    bounds = aabb.max_bound - aabb.min_bound

    height = bounds[2]
    radius = max(bounds[:2]) / 2

    if height < min_height:
        return False
    if radius < min_radius:
        return False
    return True


def split_large_clusters(clusters, min_points=10, max_radius=2.0,
                         min_peak_distance=3.0, k=50, min_density_ratio=1.5,
                         save_pre_split_path="/pre_split_clusters/"):
    """
    Desc:
        Splits large clusters that likely contain multiple trees using Mean Shift
        clustering. Mean Shift replaces the previous KMeans approach — it discovers
        the number of sub-trees automatically by finding density modes in the XY
        plane, rather than relying on a hand-tuned density peak detector to count
        them first.

        For clusters that are within max_radius, no split is attempted and they
        are passed through unchanged. Only genuinely oversized clusters reach
        the Mean Shift step.

    How Mean Shift works here:
        - Only the XY coordinates of each point are clustered (horizontal footprint).
        - The bandwidth (kernel radius) controls how far apart two modes need to be
          to be counted as separate trees. Set it close to the minimum expected
          distance between tree trunks — usually 1.5x to 2x the typical crown radius.
        - `bin_seeding=True` makes it much faster on large point clouds by pre-binning
          points onto a grid before running the shift iterations.
        - Unlike KMeans, no `n_clusters` is needed. If the cluster is really just one
          dense blob, Mean Shift will return a single centroid and no split happens.

    Args:
        clusters, list of o3d.PointCloud:
            Input clusters, typically from cluster_by_chm_peaks() or
            cluster_pointcloud().
        min_points, int:
            Minimum points a cluster (or sub-cluster after splitting) must have
            to be kept. Default 10.
        max_radius, float:
            Clusters whose XY radius is <= this value are not split. In metres.
            Default 2.0.
        min_peak_distance, float:
            Passed to Mean Shift as the bandwidth (metres). Two density peaks
            closer than this will be merged into one. Should be roughly the
            minimum expected trunk-to-trunk distance. Default 3.0.
        k, int:
            Unused — kept for API compatibility with the old KMeans version.
        min_density_ratio, float:
            Unused — kept for API compatibility. Mean Shift handles density
            implicitly via the kernel.

    Returns:
        final_clusters, list of o3d.PointCloud:
            Clusters after splitting. Length >= len(clusters).

    Requirements:
        numpy, open3d, scipy.spatial.KDTree,
        sklearn.cluster.MeanShift, sklearn.cluster.estimate_bandwidth
    """

    final_clusters = []
    pre_split_clusters = []  # clusters that are valid candidates for splitting

    for pcd in clusters:
        points = np.asarray(pcd.points)

        # gate 1: ignore tiny clusters entirely
        if len(points) < min_points:
            continue

        # gate 2: only consider splitting if the cluster is actually large
        aabb = pcd.get_axis_aligned_bounding_box()
        width = max((aabb.max_bound - aabb.min_bound)[:2])
        radius = width / 2

        if radius <= max_radius:
            final_clusters.append(pcd)
            continue

        # only reach here if the cluster is genuinely large
        peaks, _ = find_density_peaks(
            points,
            k=k,
            min_peak_distance=min_peak_distance,
            min_density_ratio=min_density_ratio
        )
        n_trees = len(peaks)

        if n_trees < 2:
            final_clusters.append(pcd)
            continue

        # this cluster is a valid split candidate — record it
        pre_split_clusters.append(pcd)

        print(f"Splitting cluster (radius={radius:.2f}m) into {n_trees} trees")

        # seed KMeans with actual peak locations
        peak_coords = points[peaks]