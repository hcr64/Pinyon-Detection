import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import MeanShift, estimate_bandwidth

from functions.save_clusters import save_clusters


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
                         save_pre_split_path="pre_split_clusters/"):
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
    pre_split_clusters = []

    for pcd in clusters:
        points = np.asarray(pcd.points)

        # gate 1: ignore tiny clusters entirely
        if len(points) < min_points:
            continue

        # gate 2: only consider splitting if the cluster is actually large
        aabb   = pcd.get_axis_aligned_bounding_box()
        width  = max((aabb.max_bound - aabb.min_bound)[:2])
        radius = width / 2

        if radius <= max_radius:
            final_clusters.append(pcd)
            continue

        # only reach here if the cluster is genuinely large
        # check for multiple density peaks before attempting a split
        peaks, _ = find_density_peaks(
            points,
            k=min(k, len(points) - 1),
            min_peak_distance=min_peak_distance,
            min_density_ratio=min_density_ratio
        )
        n_peaks = len(peaks)

        if n_peaks < 2:
            # wide cluster but only one density core — still just one tree
            final_clusters.append(pcd)
            continue

        print(f"Found {n_peaks} density peaks in cluster (radius={radius:.2f}m), attempting Mean Shift split...")

        # ── only reach here if the cluster is genuinely large ─────────────────

        # Mean Shift on XY only — we care about horizontal crown separation,
        # not vertical height variation
        xy = points[:, :2]

        # subsample for bandwidth estimation if the cluster is huge
        # (estimate_bandwidth is O(n^2) so cap at 2000 points)
        sample_size = min(len(xy), 2000)
        rng         = np.random.default_rng(42)
        sample_xy   = xy[rng.choice(len(xy), sample_size, replace=False)]

        # bandwidth = min_peak_distance lets you control "how far apart must
        # two crowns be to be counted separately" directly
        # if you'd rather let sklearn estimate it from data density, uncomment:
        # bandwidth = estimate_bandwidth(sample_xy, quantile=0.15)
        bandwidth = min_peak_distance

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=5)
        ms.fit(xy)

        sub_labels  = ms.labels_
        n_trees     = len(np.unique(sub_labels))

        if n_trees < 2:
            # Mean Shift found only one mode — still just one tree
            final_clusters.append(pcd)
            continue

        # this cluster is a valid split candidate — record it
        pre_split_clusters.append(pcd)

        print(f"Mean Shift split cluster (radius={radius:.2f}m) into {n_trees} sub-clusters")

        # ── validate each sub-cluster before accepting the split ───────────────
        valid_subs = []
        for i in np.unique(sub_labels):
            mask    = sub_labels == i
            sub_pcd = pcd.select_by_index(np.where(mask)[0])
            if (len(sub_pcd.points) >= min_points and
                    filter_cluster(sub_pcd, min_height=1.0, min_radius=0.3)):
                valid_subs.append(sub_pcd)

        # keep whatever valid sub-clusters came out, even if only one survives
        if len(valid_subs) >= 1:
            discarded = n_trees - len(valid_subs)
            if discarded > 0:
                print(f"  Discarded {discarded} sub-clusters that failed validation")
            final_clusters.extend(valid_subs)
        else:
            # nothing survived validation at all — keep the original
            print(f"  Split rejected — no sub-clusters passed validation, keeping original")
            final_clusters.append(pcd)

    # save pre-split clusters locally if a path was given
    if save_pre_split_path is not None:
        save_clusters(pre_split_clusters, save_pre_split_path)
        print(f"Saved {len(pre_split_clusters)} pre-split clusters to {save_pre_split_path}")

    print(f"Clusters before splitting: {len(clusters)}")
    print(f"Clusters after splitting:  {len(final_clusters)}")
    return final_clusters



def find_density_peaks(points, k=30, min_peak_distance=3.0, min_density_ratio=2.5):
    """
    Find local density peaks in a point cloud.
    Each peak likely corresponds to a separate tree trunk/canopy core.
    """

    # cap k to avoid index errors on small clusters
    k = min(k, len(points) - 1)

    tree = KDTree(points)
    distances, neighbor_indices = tree.query(points, k=k)
    density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)

    mean_density = density.mean()

    # a point is a peak if it has higher density than all its k neighbors
    # AND is meaningfully denser than the cluster average
    peaks = []
    for i, neighbors in enumerate(neighbor_indices):
        if (density[i] == density[neighbors].max() and
                density[i] > mean_density * min_density_ratio):
            peaks.append(i)

    # filter out peaks that are too close together — likely the same tree
    filtered_peaks = []
    for i in peaks:
        too_close = any(
            np.linalg.norm(points[i] - points[j]) < min_peak_distance
            for j in filtered_peaks
        )
        if not too_close:
            filtered_peaks.append(i)

    return filtered_peaks, density