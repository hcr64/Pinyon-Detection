# functions/build_chm.py
import numpy as np
import pandas as pd
import open3d as o3d
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import os


def build_chm(
    point_cloud,
    resolution=0.5,
    ground_percentile=5,
    save_path=None,
    utm_zone=12,
    northern_hemisphere=True,
):
    """
    Build a Canopy Height Model (CHM) from a raw point cloud and optionally
    save it as a GeoTIFF raster via rasterio.

    The CHM is computed as:
        CHM = DSM (max Z per cell) - DTM (low-percentile Z per cell)

    Args:
        point_cloud (o3d.geometry.PointCloud):
            Raw Open3D point cloud *before* green filtering.
            Must have XYZ coordinates in UTM metres (same CRS used
            elsewhere in the pipeline, e.g. EPSG:26912 for UTM zone 12N).
        resolution (float):
            Raster cell size in metres. Default 0.5 m.
        ground_percentile (int):
            Percentile of Z values within a cell used to estimate ground
            elevation for the DTM. Default 5 (5th percentile).
        save_path (str | None):
            If given, the CHM is written to this path as a single-band
            float32 GeoTIFF (e.g. "pointclouds/trial/chm.tif").
            Directories are created automatically.  Pass None to skip
            saving.
        utm_zone (int):
            UTM zone number matching the point cloud's projected CRS.
            Default 12 (Arizona / Sunset Crater area).
        northern_hemisphere (bool):
            True for northern hemisphere EPSG codes (326xx), False for
            southern (327xx). Default True.

    Returns:
        chm (np.ndarray):
            2-D float32 array of vegetation heights (rows = Y, cols = X).
            No-data cells are NaN.
        transform (rasterio.transform.Affine):
            Affine transform mapping pixel indices → UTM coordinates.
            Pass to rasterio.open() or use with rasterio.transform.xy()
            to convert pixel positions back to real-world coordinates.
        crs (rasterio.crs.CRS):
            Coordinate reference system of the raster.

    Requirements:
        numpy, pandas, open3d, rasterio
    """

    points = np.asarray(point_cloud.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # ── build regular grid ────────────────────────────────────────────────────
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # number of cells in each direction
    n_cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    n_rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    # cell index for every point  (0-based, row 0 = northernmost)
    col_idx = np.floor((x - x_min) / resolution).astype(int)
    row_idx = np.floor((y_max - y) / resolution).astype(int)   # flip Y so row 0 is top

    # clamp to grid bounds (safety margin for floating-point edge cases)
    col_idx = np.clip(col_idx, 0, n_cols - 1)
    row_idx = np.clip(row_idx, 0, n_rows - 1)

    # ── per-cell DSM and DTM ──────────────────────────────────────────────────
    dsm = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    dtm = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # group points by cell using a pandas DataFrame (vectorised, avoids nested loops)
    df = pd.DataFrame({"row": row_idx, "col": col_idx, "z": z})
    grouped = df.groupby(["row", "col"])["z"]

    dsm_series = grouped.max()
    dtm_series = grouped.quantile(ground_percentile / 100.0)

    for (r, c), val in dsm_series.items():
        dsm[r, c] = val

    for (r, c), val in dtm_series.items():
        dtm[r, c] = val

    # ── CHM = DSM − DTM, clipped to [0, ∞) ───────────────────────────────────
    chm = np.where(np.isnan(dsm) | np.isnan(dtm), np.nan, dsm - dtm).astype(np.float32)
    chm = np.where(np.isnan(chm), np.nan, np.clip(chm, 0, None))

    # ── rasterio affine transform ─────────────────────────────────────────────
    # from_origin(west, north, xsize, ysize) — origin is the top-left corner
    transform = from_origin(
        west=x_min,
        north=y_max,
        xsize=resolution,
        ysize=resolution,     # rasterio uses positive ysize; it negates internally
    )

    # UTM CRS  (EPSG 326xx = north, 327xx = south)
    epsg_code = 32600 + utm_zone if northern_hemisphere else 32700 + utm_zone
    crs = CRS.from_epsg(epsg_code)

    # ── optional GeoTIFF export ───────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

        with rasterio.open(
            save_path,
            mode="w",
            driver="GTiff",
            height=n_rows,
            width=n_cols,
            count=1,                    # single band
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="lzw",             # lossless; keeps file sizes small
        ) as dst:
            dst.write(chm, 1)           # band 1

        print(f"CHM saved to {save_path}  ({n_cols}×{n_rows} px, {resolution} m/px)")

    return chm, transform, crs


# ── height normalisation (unchanged logic, kept here for locality) ────────────

def normalize_heights_by_ground(point_cloud, resolution=0.5, ground_percentile=5):
    """
    Subtract a per-cell ground estimate from every point's Z value so that
    vegetation heights are relative to local ground rather than sea level.

    Args:
        point_cloud (o3d.geometry.PointCloud): Input cloud with absolute Z.
        resolution (float): Cell size in metres for the ground grid.
        ground_percentile (int): Percentile used as the ground proxy (default 5).

    Returns:
        o3d.geometry.PointCloud: New point cloud with normalised Z values.
    """
    points = np.asarray(point_cloud.points).copy()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    xi = ((x - x.min()) / resolution).astype(int)
    yi = ((y - y.min()) / resolution).astype(int)

    df = pd.DataFrame({"xi": xi, "yi": yi, "z": z, "idx": np.arange(len(z))})

    ground_per_cell = (
        df.groupby(["xi", "yi"])["z"]
        .quantile(ground_percentile / 100.0)
        .rename("ground_z")
        .reset_index()
    )

    df = df.merge(ground_per_cell, on=["xi", "yi"], how="left").sort_values("idx")
    points[:, 2] = z - df["ground_z"].values

    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(points)
    if point_cloud.has_colors():
        normalized_pcd.colors = point_cloud.colors

    return normalized_pcd


# ── convenience: read a saved CHM back in ────────────────────────────────────

def load_chm(chm_path):
    """
    Read a CHM GeoTIFF saved by build_chm() back into numpy.

    Args:
        chm_path (str): Path to the .tif file.

    Returns:
        chm (np.ndarray): 2-D float32 height array.
        transform (Affine): Rasterio affine transform.
        crs (CRS): Coordinate reference system.
    """
    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return chm, transform, crs