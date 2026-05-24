import numpy as np
import open3d as o3d

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

    # must be tall enough to be a tree, not just a ground patch
    if height < min_height:
        return False

    # must have some horizontal spread (eliminates single vertical returns)
    if radius < min_radius:
        return False

    return True