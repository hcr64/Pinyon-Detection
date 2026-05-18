import open3d as o3d
import numpy as np

def cluster_pointcloud(point_cloud, eps=1.5, min_points=10):
    """ 
    Desc:
        Strips a given pointcloud of all non-green points. Returns stripped pcd as open3D pcd object.

    Args:
        point_cloud, open3D pointcloud: The pointcloud to cluster. Usually the stripped pointcloud of green points.
        eps, double (1.5-3.0): Distance between points for clustering. In meters.
        min_points, int: minimum amount of points per cluster. Cluster parameter.

    Returns:
        clusters: An array of 'clsuters', each a pointcloud object.
        labels: I am unsure, it is unused.

    Requirements:
        numpy, open3d
    """

    # run DBSCAN clustering
    labels = np.array(point_cloud.cluster_dbscan(
        eps=eps,                # max distance between points in same cluster
        min_points=min_points,  # minimum points to form a cluster
        print_progress=True
    ))

    # labels == -1 means noise (no cluster)
    n_clusters = labels.max() + 1
    print(f"Found {n_clusters} clusters")

    # split into individual point clouds
    clusters = []
    for i in range(n_clusters):
        mask = labels == i
        cluster = point_cloud.select_by_index(np.where(mask)[0])
        clusters.append(cluster)

    return clusters, labels