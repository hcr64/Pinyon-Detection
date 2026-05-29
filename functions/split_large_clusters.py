import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from scipy.spatial import KDTree


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


def filter_cluster(pcd, min_height=1.0, min_radius=0.3):
    """
    Reject clusters that are too flat or too narrow to be trees.
    Grass patches tend to be wide but very short.
    Isolated shrubs tend to be small in all dimensions.
    """
    points = np.asarray(pcd.points)
    aabb = pcd.get_axis_aligned_bounding_box()
    bounds = aabb.max_bound - aabb.min_bound

    height = bounds[2]
    radius = max(bounds[:2]) / 2

    if height < min_height:
        return False

    if radius < min_radius:
        return False

    return True


def split_large_clusters(clusters, min_points=10, max_radius=2.0,
                         min_peak_distance=3.0, k=50, min_density_ratio=1.5):
    final_clusters = []

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
            # small enough — keep as is, no split needed
            final_clusters.append(pcd)
            continue

        # only reach here if the cluster is genuinely large
        # use density peaks to decide how many trees are inside
        peaks, _ = find_density_peaks(
            points,
            k=k,
            min_peak_distance=min_peak_distance,
            min_density_ratio=min_density_ratio
        )
        n_trees = len(peaks)

        if n_trees < 2:
            # large cluster but only one density core — still just one tree
            final_clusters.append(pcd)
            continue

        print(f"Splitting cluster (radius={radius:.2f}m) into {n_trees} trees")

        # seed KMeans with actual peak locations
        peak_coords = points[peaks]
        kmeans = KMeans(n_clusters=n_trees, init=peak_coords, n_init=1, random_state=42)
        sub_labels = kmeans.fit_predict(points)

        # validate each sub-cluster before accepting the split
        valid_subs = []
        for i in range(n_trees):
            mask = sub_labels == i
            sub_pcd = pcd.select_by_index(np.where(mask)[0])
            if len(sub_pcd.points) > min_points and filter_cluster(sub_pcd, min_height=1.0, min_radius=0.3):
                valid_subs.append(sub_pcd)

        # if the split produced garbage, keep the original instead
        if len(valid_subs) >= 2:
            final_clusters.extend(valid_subs)
        else:
            print(f"Split rejected — sub-clusters failed validation, keeping original")
            final_clusters.append(pcd)

    print(f"Clusters before splitting: {len(clusters)}")
    print(f"Clusters after splitting: {len(final_clusters)}")
    return final_clusters