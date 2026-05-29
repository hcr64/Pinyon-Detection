import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def cluster_by_chm_peaks(point_cloud, peak_coords, crown_radius=2.0, min_points=40):
    """
    Desc:
        Clusters a point cloud by assigning each point to its nearest CHM peak,
        as long as the point is within crown_radius metres of that peak (XY only).
        Points outside crown_radius of any peak are discarded entirely, which
        naturally strips out most ground cover and grass that DBSCAN would pick up.

    Args:
        point_cloud, o3d.PointCloud:  The cleaned/green-filtered point cloud.
        peak_coords, np.ndarray:      (N, 2) UTM XY coordinates from find_chm_peaks().
        crown_radius, float:          Maximum XY distance a point can be from its
                                      nearest peak to be included in that cluster.
                                      Roughly the expected maximum crown radius in
                                      metres. Default 2.0 m.
        min_points, int:              Minimum points a cluster must have to be kept.
                                      Default 40.

    Returns:
        clusters, list of o3d.PointCloud: One cluster per surviving peak.

    Requirements:
        numpy, open3d, scipy.spatial.KDTree
    """

    points = np.asarray(point_cloud.points)
    xy     = points[:, :2]                      # only use XY for assignment

    peak_tree              = KDTree(peak_coords)
    distances, peak_labels = peak_tree.query(xy) # nearest peak for every point

    # discard points that are too far from any peak
    valid_mask = distances <= crown_radius
    print(f"Points within crown_radius: {valid_mask.sum()} / {len(points)} "
          f"({100 * valid_mask.mean():.1f}% kept)")

    clusters = []
    for i in range(len(peak_coords)):
        mask    = valid_mask & (peak_labels == i)
        n_pts   = mask.sum()

        if n_pts < min_points:
            continue

        cluster = point_cloud.select_by_index(np.where(mask)[0])
        clusters.append(cluster)

    print(f"Clusters from CHM peaks: {len(clusters)} "
          f"(of {len(peak_coords)} peaks, min_points={min_points})")

    return clusters