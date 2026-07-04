import numpy as np
import pandas as pd
import laspy
import open3d as o3d
import os


def las_folder_to_pointcloud(folder_path, silent=False, downsize_pcd=True, v_size=0.05):
    """
    Load all .las files in a folder and merge them into a single Open3D PointCloud.
 
    Reads XYZ coordinates and RGB colour (if present) from each .las file,
    concatenates them, and optionally voxel-downsamples the combined result.
    RGB values are normalised from 16-bit (0–65535) to float (0.0–1.0) as
    required by Open3D.
 
    Args:
        folder_path (str): Path to a folder containing only .las files.
        silent (bool): Suppress per-file progress messages when True.
            Default False.
        downsize_pcd (bool): Apply voxel downsampling to the merged cloud
            when True. Default True.
        v_size (float): Voxel size in metres used for downsampling.
            Typical range 0.05–0.1. Default 0.05.
 
    Returns:
        o3d.geometry.PointCloud: Merged point cloud in UTM metres (EPSG:26912
            for Sunset Crater). Colour channels are present if any input file
            contained RGB data.
 
    Requirements:
        numpy, laspy, open3d, os
    """

    if not silent:
        print("begin las_folder_to_pointcloud...")

    combined_pcd = o3d.geometry.PointCloud()

    folder_contents = os.listdir(folder_path)

    if not silent:
        print("begin iterating through files...")

    for file in folder_contents:
        if not silent:
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
    

    return combined_pcd