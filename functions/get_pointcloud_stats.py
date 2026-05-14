# import libraries
import numpy as np
import pandas as pd
import open3d as o3d
import os

# a function to put all the pinyon stats into a df
def clusters_to_dataframe(clusters_folder):

    rows = []

    for file in os.listdir(clusters_folder):
        if file.endswith(".ply"):
            path = os.path.join(clusters_folder, file)
            stats = get_pointcloud_stats(path)
            stats["file"] = file  # keep track of which file
            rows.append(stats)

    df = pd.DataFrame(rows)

    # sort by file name
    df = df.sort_values("file").reset_index(drop=True)

    return df

# a function to get stats from a pointcloud, returns array
def get_pointcloud_stats(pcd_path, silent=True):

    # read the pointcloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound  # [x, y, z]
    max_bound = aabb.max_bound

    # height (z axis)
    height = max_bound[2] - min_bound[2]

    # width/radius (x and y axes)
    width_x = max_bound[0] - min_bound[0]
    width_y = max_bound[1] - min_bound[1]
    radius = max(width_x, width_y) / 2

    # center position (x, y)
    center = aabb.get_center()
    x_pos = center[0]
    y_pos = center[1]

    # number of points
    n_points = len(points)

    # point density (points per cubic meter)
    volume = width_x * width_y * height
    density = n_points / volume if volume > 0 else 0

    stats = {
        "n_points":  n_points,
        "height":    round(height, 3),
        "width_x":   round(width_x, 3),
        "width_y":   round(width_y, 3),
        "radius":    round(radius, 3),
        "x_pos":     round(x_pos, 3),
        "y_pos":     round(y_pos, 3),
        "density":   round(density, 3),
    }
    if not silent:
      for k, v in stats.items():
          print(f"{k}: {v}")

    return stats