import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import MeanShift, estimate_bandwidth

from functions.io.save_clusters import save_clusters
from functions.io.save_clusters_descriptive import save_clusters_descriptive

def filter_cluster(pcd, min_height=1.0, min_radius=0.3):
    """
    Return True if a cluster meets minimum geometric criteria to be a tree.
 
    Rejects clusters that are too flat (likely grass patches) or too narrow
    (likely isolated shrubs or noise).
 
    Args:
        pcd (o3d.geometry.PointCloud): Cluster to evaluate.
        min_height (float): Minimum Z range in metres. Default 1.0.
        min_radius (float): Minimum XY half-width in metres. Default 0.3.
 
    Returns:
        bool: True if the cluster passes both thresholds, False otherwise.
 
    Requirements:
        numpy, open3d
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
    Split oversized clusters that likely contain multiple merged tree crowns.
 
    Large clusters (XY radius > max_radius) are inspected for multiple density
    peaks using find_density_peaks(). Only clusters with two or more
    well-separated peaks proceed to splitting — wide single-tree crowns are
    passed through unchanged. Accepted candidates are split with Mean Shift
    clustering on the XY plane, which discovers the number of sub-crowns
    automatically without requiring n_clusters.
 
    Each sub-cluster is validated by filter_cluster() before being accepted.
    Sub-clusters that fail validation are discarded individually; the split is
    only fully reverted if no sub-clusters survive validation at all. Pre-split
    clusters are optionally saved for post-run inspection, named descriptively
    (radius + sub-crown count) rather than by generic index — this folder is
    a Drive export for visual review and is never re-read by load_clusters(),
    so filenames don't need to encode a positional index.
 
    Args:
        clusters (list of o3d.geometry.PointCloud): Input clusters from
            cluster_by_chm_peaks() or cluster_pointcloud().
        min_points (int): Minimum points for any cluster (or sub-cluster after
            splitting) to be kept. Default 10.
        max_radius (float): XY radius threshold in metres below which a cluster
            is not split. Default 2.0.
        min_peak_distance (float): Used both as the minimum separation between
            density peaks in find_density_peaks() and as the Mean Shift
            bandwidth. Roughly the minimum expected trunk-to-trunk distance
            in metres. Default 3.0.
        k (int): Kept for API compatibility; unused since Mean Shift replaced
            KMeans. Previously controlled the KDTree neighbour count.
        min_density_ratio (float): Kept for API compatibility; passed to
            find_density_peaks() to gate splitting. A peak must be this many
            times denser than the cluster mean to qualify. Default 1.5.
        save_pre_split_path (str | None): Directory to save clusters before
            splitting, for inspection. Pass None to skip. Default
            "pre_split_clusters/".
 
    Returns:
        list of o3d.geometry.PointCloud: Clusters after splitting.
            Length >= len(clusters).
 
    Requirements:
        numpy, open3d, scipy.spatial.KDTree,
        sklearn.cluster.MeanShift
    """

    final_clusters = []
    pre_split_clusters = []
    pre_split_names = []

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
        pre_split_names.append(f"radius{radius:.2f}m_split_into_{n_trees}")

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
        save_clusters_descriptive(pre_split_clusters, pre_split_names, save_pre_split_path)
        print(f"Saved {len(pre_split_clusters)} pre-split clusters to {save_pre_split_path}")

    print(f"Clusters before splitting: {len(clusters)}")
    print(f"Clusters after splitting:  {len(final_clusters)}")
    return final_clusters



def find_density_peaks(points, k=30, min_peak_distance=3.0, min_density_ratio=2.5):
    """
    Find local density peaks in a point cloud using KDTree-based density estimation.
 
    A point qualifies as a peak if its local density (inverse mean distance to k
    neighbours) is higher than all its k neighbours *and* exceeds the cluster-wide
    mean density by min_density_ratio. Peaks closer than min_peak_distance to an
    already-accepted peak are suppressed (greedy nearest-first).
 
    Called by split_large_clusters() to gate splitting — only clusters with two
    or more surviving peaks are sent to Mean Shift.
 
    Args:
        points (np.ndarray): (N, 3) XYZ array of cluster points.
        k (int): Number of neighbours for local density estimation. Capped
            internally at len(points) - 1. Default 30.
        min_peak_distance (float): Minimum separation in metres between two
            accepted peaks. Default 3.0.
        min_density_ratio (float): A candidate peak must be at least this many
            times denser than the cluster mean density. Default 2.5.
 
    Returns:
        filtered_peaks (list of int): Indices into points of accepted peaks.
        density (np.ndarray): (N,) per-point density values.
 
    Requirements:
        numpy, scipy.spatial.KDTree
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