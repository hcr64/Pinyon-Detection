import open3d as o3d
import numpy as np
 
def clean_up_pointcloud(point_cloud, green_threshold=0.01):
    """ 
    Desc:
        Strips a given pointcloud of all non-green points. Returns stripped pcd as open3D pcd object.

    Args:
        point_cloud, open3D pointcloud: the pointcloud to be 'stripped' down.
        green_threshold, double: How inclusive the stripping process is. The closer to 1, the less inclusive to pixels.

    Returns:
        The pointcloud parameter minus all voxels that do not meet the trheshold. 
        
    Requirements:
        numpy, open3d
    """
    if not point_cloud.has_colors():
        print("Warning: no color data, skipping green filter.")
        return point_cloud

    colors = np.asarray(point_cloud.colors)  # shape (N, 3), values 0.0–1.0
    points = np.asarray(point_cloud.points)

    # keep points where green channel dominates over red and blue
    mask = (
        (colors[:, 1] > colors[:, 0] + green_threshold) &  # green > red
        (colors[:, 1] > colors[:, 2] + green_threshold)    # green > blue
    )

    # option 2 — ExG index, best for vegetation
    """exg = 2 * colors[:, 1] - colors[:, 0] - colors[:, 2]
    mask = exg > 0.0  # just needs to be net-green at all"""

    cleaned_pc = point_cloud.select_by_index(np.where(mask)[0])

    return cleaned_pc