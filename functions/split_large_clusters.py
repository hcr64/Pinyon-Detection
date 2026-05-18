import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d

def split_large_clusters(clusters, max_radius=2.0):
    final_clusters = []
    
    for pcd in clusters:
        points = np.asarray(pcd.points)
        aabb = pcd.get_axis_aligned_bounding_box()
        width = max((aabb.max_bound - aabb.min_bound)[:2])  # x/y width
        radius = width / 2

        if radius > max_radius:
            # estimate how many trees fit in this cluster
            n_splits = max(2, int(radius / max_radius))
            print(f"Splitting cluster into {n_splits} (radius={radius:.2f}m)")

            # split using KMeans
            kmeans = KMeans(n_clusters=n_splits, random_state=42)
            sub_labels = kmeans.fit_predict(points)

            for i in range(n_splits):
                mask = sub_labels == i
                sub_pcd = pcd.select_by_index(np.where(mask)[0])
                if len(sub_pcd.points) > 10:  # ignore tiny fragments
                    final_clusters.append(sub_pcd)
        else:
            final_clusters.append(pcd)

    print(f"Clusters before splitting: {len(clusters)}")
    print(f"Clusters after splitting: {len(final_clusters)}")
    return final_clusters