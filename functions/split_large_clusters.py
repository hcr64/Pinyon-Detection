import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from scipy.spatial import KDTree

def find_density_peaks(points, k=10, min_peak_distance=1.0):
    """
    Find local density peaks in a point cloud.
    Each peak likely corresponds to a separate tree trunk/canopy core.
    """
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k)
    density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)

    # a point is a peak if it has higher density than all its k neighbors
    _, neighbor_indices = tree.query(points, k=k)
    peaks = []
    for i, neighbors in enumerate(neighbor_indices):
        if density[i] == density[neighbors].max():
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


def split_large_clusters(clusters, min_points=10, max_radius=2.0, min_peak_distance=2.0, k=20):
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
        # now use density peaks to decide how many trees are inside
        peaks, _ = find_density_peaks(points, k=k, min_peak_distance=min_peak_distance)
        n_trees = len(peaks)

        if n_trees < 2:
            # large cluster but only one density core — still just one tree
            final_clusters.append(pcd)
            continue

        print(f"Splitting cluster (radius={radius:.2f}m) into {n_trees} trees")
        peak_coords = points[peaks]
        kmeans = KMeans(n_clusters=n_trees, init=peak_coords, n_init=1, random_state=42)
        sub_labels = kmeans.fit_predict(points)

        for i in range(n_trees):
            mask = sub_labels == i
            sub_pcd = pcd.select_by_index(np.where(mask)[0])
            if len(sub_pcd.points) > min_points:
                final_clusters.append(sub_pcd)

    print(f"Clusters before splitting: {len(clusters)}")
    print(f"Clusters after splitting: {len(final_clusters)}")
    return final_clusters