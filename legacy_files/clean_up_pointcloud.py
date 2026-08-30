import open3d as o3d
import numpy as np
 
def clean_up_pointcloud(point_cloud, green_threshold=0.01):
    """
    Remove non-vegetation points by filtering on green channel dominance.
 
    Keeps only points where the green channel exceeds both the red and blue
    channels by at least green_threshold. This retains live vegetation while
    discarding bare ground, rock, and sky. Operates in normalised RGB space
    (0.0–1.0).
 
    Args:
        point_cloud (o3d.geometry.PointCloud): Input cloud with colour data.
            If colour is absent a warning is printed and the cloud is returned
            unchanged.
        green_threshold (float): Minimum margin by which green must exceed
            both red and blue. 0.025 was found optimal at Sunset Crater;
            values above ~0.05 begin to exclude valid vegetation. Default 0.01.
 
    Returns:
        o3d.geometry.PointCloud: Filtered cloud containing only points that
            pass the green dominance test.
 
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


""" IMPLEMENTATION in run_clustering:



# ── clean pointcloud ──────────────────────────────────────────────────
    # usually set to false, strips all non-green points, has not been part of the workflow for a while
    # it does make the program a lot faster
    # goes with cluster_pointcloud, not used with CHM workflow
    if STEPS['Clean_Pointcloud']:
        points = np.asarray(point_cloud.points)
        print(f"Point count:  {len(points)}")
        print(f"XYZ min: {points.min(axis=0)}")
        print(f"XYZ max: {points.max(axis=0)}")
        print(f"Any NaN: {np.any(np.isnan(points))}")

        print("Cleaning up point cloud...")
        point_cloud = clean_up_pointcloud(point_cloud, green_threshold=GREEN_THRESHOLD)
        print("Point cloud successfully cleaned.\n")

        print(point_cloud)

        if SAVE:
            print("Saving cleaned pointcloud...")
            if not os.path.exists(os.path.dirname(PATHS['Cleaned_pcd'])):
                os.makedirs(os.path.dirname(PATHS['Cleaned_pcd']))
            o3d.io.write_point_cloud(PATHS['Cleaned_pcd'], point_cloud)
            print("Cleaned pointcloud saved.\n")
        else:
            print("SAVE=False — skipping cleaned pointcloud write.\n")

    else:
        print("Unprocessed pointcloud being used. Pointcloud is not being 'cleaned.'\n") 
        
"""