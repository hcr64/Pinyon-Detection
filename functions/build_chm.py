# functions/build_chm.py
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.interpolate import griddata

def build_chm(point_cloud, resolution=0.5, ground_percentile=5):
    """
    Build a Canopy Height Model from a point cloud.

    Args:
        point_cloud: Open3D PointCloud (raw, before green filtering)
        resolution:  CHM pixel size in meters (0.5 = 50cm grid)
        ground_percentile: percentile of Z values per cell used as ground estimate

    Returns:
        chm:        2D numpy array of vegetation heights
        x_bins, y_bins: grid coordinates for the CHM
    """
    points = np.asarray(point_cloud.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # build grid
    x_bins = np.arange(x.min(), x.max(), resolution)
    y_bins = np.arange(y.min(), y.max(), resolution)

    dsm    = np.full((len(y_bins), len(x_bins)), np.nan)
    dtm    = np.full((len(y_bins), len(x_bins)), np.nan)

    xi = np.digitize(x, x_bins) - 1
    yi = np.digitize(y, y_bins) - 1

    # for each cell, DSM = max Z, DTM = low percentile Z (approx ground)
    for i in range(len(x_bins)):
        for j in range(len(y_bins)):
            mask = (xi == i) & (yj == j)
            if mask.sum() > 0:
                cell_z = z[mask]
                dsm[j, i] = np.max(cell_z)
                dtm[j, i] = np.percentile(cell_z, ground_percentile)

    chm = np.where(np.isnan(dsm) | np.isnan(dtm), np.nan, dsm - dtm)
    chm = np.clip(chm, 0, None)  # no negative heights

    return chm, x_bins, y_bins


def normalize_heights_by_ground(point_cloud, resolution=0.5, ground_percentile=5):
    points = np.asarray(point_cloud.points).copy()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    xi = ((x - x.min()) / resolution).astype(int)
    yi = ((y - y.min()) / resolution).astype(int)

    df = pd.DataFrame({'xi': xi, 'yi': yi, 'z': z, 'idx': np.arange(len(z))})

    # native pandas quantile — fully vectorized, no lambda
    ground_per_cell = (
        df.groupby(['xi', 'yi'])['z']
        .quantile(ground_percentile / 100.0)
        .rename('ground_z')
        .reset_index()
    )

    df = df.merge(ground_per_cell, on=['xi', 'yi'], how='left')
    df = df.sort_values('idx')  # restore original point order

    points[:, 2] = z - df['ground_z'].values

    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(points)
    if point_cloud.has_colors():
        normalized_pcd.colors = point_cloud.colors

    return normalized_pcd