import numpy as np
import pandas as pd
import laspy
import open3d as o3d
import os


def las_folder_to_pointcloud(folder_path, rm_floor=False, silent=False, downsize_pcd=True, v_size=0.05):
    if not silent:
        print("begin las_folder_to_pointcloud...")

    combined_pcd = o3d.geometry.PointCloud()

    folder_contents = os.listdir(folder_path)

    if not silent:
        print("begin iterating through files...")

    for file in folder_contents:

        print(f"Loading {file}...")

        las = laspy.read(folder_path + file)

        # Extract XYZ coordinates
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # --- Extract RGB color if available ---
        colors = None

        if not silent:
            print("loading colors...")

        try:
            # LAS stores RGB as 16-bit (0–65535), Open3D expects 0.0–1.0
            r = np.array(las.red,   dtype=np.float64) / 65535.0
            g = np.array(las.green, dtype=np.float64) / 65535.0
            b = np.array(las.blue,  dtype=np.float64) / 65535.0
            colors = np.vstack((r, g, b)).transpose()
        except AttributeError:
            print(f"No RGB data found in {file}, skipping color.")

        if not silent:
            print("Building pointcloud...")

        # Build this file's point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if not silent:
            print("Appending pointcloud...")

        # Append onto combined cloud (points + colors together)
        existing_points = np.asarray(combined_pcd.points)
        new_points      = np.asarray(pcd.points)
        combined_pcd.points = o3d.utility.Vector3dVector(
            np.vstack((existing_points, new_points)) if len(existing_points) else new_points
        )
        if not silent:
            print("Checking for colors...")

        if pcd.has_colors():
            existing_colors = np.asarray(combined_pcd.colors)
            new_colors      = np.asarray(pcd.colors)
            combined_pcd.colors = o3d.utility.Vector3dVector(
                np.vstack((existing_colors, new_colors)) if len(existing_colors) else new_colors
            )

        print(f"Successfully loaded {file}.\n")
            
    
    print("Done iterating through files.\n")
    
    # Downsample — voxel_down_sample preserves color automatically
    if downsize_pcd:
        print("Downsizing pointcloud...")
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=v_size)
        print("Done downsizing.")

    if rm_floor:
        plane_model, inliers = combined_pcd.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=1000
        )
        ground         = combined_pcd.select_by_index(inliers)
        forest_no_floor = combined_pcd.select_by_index(inliers, invert=True)
        o3d.io.write_point_cloud("forest_no_floor.ply", forest_no_floor)

    return combined_pcd