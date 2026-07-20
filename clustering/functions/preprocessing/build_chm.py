# functions/build_chm.py
import numpy as np
import pandas as pd
import open3d as o3d
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import os


# ── external DSM ingestion (PixMapper4D or similar photogrammetry output) ────

def build_chm_from_external_dsm(
    dsm_path,
    point_cloud=None,
    dtm_path=None,
    ground_percentile=5,
    ground_grid_resolution=0.5,
    save_path=None,
):
    """
    (docstring — same as before, plus:)

    ground_grid_resolution (float): Resolution in metres used to compute the
        point-cloud-derived DTM (mode 2 only), BEFORE resampling onto the
        DSM's native grid. Must be coarse enough that many points fall in
        each cell — this is what makes the ground_percentile statistically
        meaningful. Should be similar to or coarser than the point cloud's
        voxel_size. Computing the DTM directly on a DSM grid finer than the
        point spacing (e.g. a 3-4cm PixMapper4D DSM vs an 8cm voxel-downsampled
        cloud) gives near-empty cells and single-point "percentiles" that are
        really just raw, possibly-canopy Z values — this parameter exists to
        avoid that failure mode. Default 0.5, matching build_chm()'s default.
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin

    with rasterio.open(dsm_path) as src:
        dsm       = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        n_rows, n_cols = dsm.shape
        nodata    = src.nodata

    if nodata is not None:
        dsm = np.where(dsm == nodata, np.nan, dsm)

    print(f"External DSM loaded: {dsm_path}  ({n_cols}×{n_rows} px, crs={crs})")

    points = None
    if point_cloud is not None:
        points = np.asarray(point_cloud.points)
        px_min, py_min = points[:, 0].min(), points[:, 1].min()
        px_max, py_max = points[:, 0].max(), points[:, 1].max()
        dsm_left,  dsm_top    = transform * (0, 0)
        dsm_right, dsm_bottom = transform * (n_cols, n_rows)

        print(f"Point cloud bounds: x=[{px_min:.1f}, {px_max:.1f}]  y=[{py_min:.1f}, {py_max:.1f}]")
        print(f"DSM bounds:         x=[{dsm_left:.1f}, {dsm_right:.1f}]  y=[{dsm_bottom:.1f}, {dsm_top:.1f}]")

        overlap = (px_min <= dsm_right and px_max >= dsm_left and
                   py_min <= dsm_top   and py_max >= dsm_bottom)
        if not overlap:
            print("⚠  WARNING: point cloud and DSM bounding boxes do not overlap.")

    if dtm_path is not None:
        print(f"Using external DTM: {dtm_path}")
        with rasterio.open(dtm_path) as src:
            dtm_raw       = src.read(1).astype(np.float32)
            dtm_transform = src.transform
            dtm_crs       = src.crs
            dtm_nodata    = src.nodata
        if dtm_nodata is not None:
            dtm_raw = np.where(dtm_raw == dtm_nodata, np.nan, dtm_raw)

        if (dtm_transform == transform and dtm_crs == crs
                and dtm_raw.shape == dsm.shape):
            dtm = dtm_raw
        else:
            print("DTM grid differs from DSM grid — reprojecting DTM onto DSM grid...")
            dtm = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
            reproject(
                source=dtm_raw, destination=dtm,
                src_transform=dtm_transform, src_crs=dtm_crs,
                dst_transform=transform, dst_crs=crs,
                resampling=Resampling.bilinear,
            )

    else:
        if point_cloud is None:
            raise ValueError(
                "build_chm_from_external_dsm: no dtm_path given, so a "
                "point_cloud is required to compute the DTM (mode 2)."
            )

        # ── compute DTM on a COARSE grid first (many points/cell), ───────────
        # then resample onto the fine DSM grid. Computing percentiles
        # directly on the DSM's native (often sub-voxel-size) resolution
        # gives near-empty cells and unreliable single-point "percentiles."
        print(f"No external DTM provided — computing DTM at "
              f"{ground_grid_resolution}m resolution ({ground_percentile}th "
              f"percentile), then resampling onto the DSM's {abs(transform.a):.4f}m grid...")

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        coarse_x_min, coarse_y_max = x.min(), y.max()
        coarse_n_cols = int(np.ceil((x.max() - x.min()) / ground_grid_resolution)) + 1
        coarse_n_rows = int(np.ceil((y.max() - y.min()) / ground_grid_resolution)) + 1

        coarse_col = np.floor((x - coarse_x_min) / ground_grid_resolution).astype(int)
        coarse_row = np.floor((coarse_y_max - y) / ground_grid_resolution).astype(int)
        coarse_col = np.clip(coarse_col, 0, coarse_n_cols - 1)
        coarse_row = np.clip(coarse_row, 0, coarse_n_rows - 1)

        df = pd.DataFrame({"row": coarse_row, "col": coarse_col, "z": z})
        dtm_series = df.groupby(["row", "col"])["z"].quantile(ground_percentile / 100.0)

        coarse_dtm = np.full((coarse_n_rows, coarse_n_cols), np.nan, dtype=np.float32)
        for (r, c), val in dtm_series.items():
            coarse_dtm[r, c] = val

        pts_per_cell = len(points) / max(1, dtm_series.notna().sum())
        print(f"  Coarse DTM grid: {coarse_n_cols}×{coarse_n_rows} px, "
              f"~{pts_per_cell:.1f} points/occupied cell")

        coarse_transform = from_origin(
            west=coarse_x_min, north=coarse_y_max,
            xsize=ground_grid_resolution, ysize=ground_grid_resolution,
        )

        # resample the coarse, statistically-sound DTM onto the fine DSM grid
        dtm = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        reproject(
            source=coarse_dtm, destination=dtm,
            src_transform=coarse_transform, src_crs=crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.bilinear,
        )

    chm = np.where(np.isnan(dsm) | np.isnan(dtm), np.nan, dsm - dtm).astype(np.float32)
    chm = np.where(np.isnan(chm), np.nan, np.clip(chm, 0, None))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        if os.path.exists(save_path):
            os.remove(save_path)   # avoid GDAL's internal delete-check choking on a
                                    # truncated/corrupted leftover from a prior
                                    # interrupted or concurrent write
        with rasterio.open(
            save_path, mode="w", driver="GTiff",
            height=n_rows, width=n_cols, count=1,
            dtype=rasterio.float32, crs=crs, transform=transform,
            nodata=np.nan, compress="lzw",
        ) as dst:
            dst.write(chm, 1)
        print(f"CHM (from external DSM) saved to {save_path}")

    return chm, transform, crs

def build_chm(
    point_cloud,
    resolution=0.5,
    ground_percentile=5,
    save_path=None,
    utm_zone=12,
    northern_hemisphere=True,
):
    """
    Build a Canopy Height Model (CHM) raster from a raw point cloud.
 
    Rasterises the cloud at the given resolution, then computes:
        DSM = max Z per cell  (top of canopy or bare ground)
        DTM = low-percentile Z per cell  (ground surface proxy)
        CHM = DSM − DTM, clipped to [0, ∞)
 
    Use the raw (unfiltered) point cloud as input — the CHM needs ground
    returns to build the DTM. Filtering vegetation out first will break the
    ground estimate.
 
    The CHM is saved as a single-band float32 GeoTIFF (LZW-compressed) when
    save_path is provided, and the rasterio Affine transform is returned so
    pixel indices can be converted back to UTM coordinates anywhere downstream.
 
    Args:
        point_cloud (o3d.geometry.PointCloud): Raw point cloud with XYZ in
            UTM metres (e.g. EPSG:26912 for zone 12N).
        resolution (float): Raster cell size in metres. Default 0.5.
        ground_percentile (int): Z percentile used as the ground proxy for the
            DTM. Default 5 (5th percentile).
        save_path (str | None): File path for the output GeoTIFF. Parent
            directories are created automatically. Pass None to skip saving.
        utm_zone (int): UTM zone number matching the point cloud CRS.
            Default 12 (Arizona / Sunset Crater).
        northern_hemisphere (bool): True for northern hemisphere (EPSG 326xx),
            False for southern (EPSG 327xx). Default True.
 
    Returns:
        chm (np.ndarray): 2-D float32 array of vegetation heights
            (rows = Y descending, cols = X ascending). No-data cells are NaN.
        transform (rasterio.transform.Affine): Affine transform mapping pixel
            indices to UTM coordinates.
        crs (rasterio.crs.CRS): Coordinate reference system of the raster.
 
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
    Normalise point Z values relative to the local ground surface.
 
    Estimates the ground elevation within each raster cell as a low percentile
    of Z, then subtracts it from every point in that cell so heights become
    relative to local terrain rather than sea level.
 
    Note: this was found to *hurt* matching scores at Sunset Crater
    (NORMALIZE_HEIGHTS=False outperforms True), likely because the cinder cone
    terrain introduces artefacts when the ground estimate is coarse. The
    function is retained for experimental use.
 
    Args:
        point_cloud (o3d.geometry.PointCloud): Input cloud with absolute Z
            in UTM metres.
        resolution (float): Cell size in metres for the ground grid.
            Default 0.5.
        ground_percentile (int): Percentile used as the ground proxy.
            Default 5.
 
    Returns:
        o3d.geometry.PointCloud: New point cloud with normalised Z values.
            Colours are preserved from the input.
 
    Requirements:
        numpy, pandas, open3d
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
        chm (np.ndarray): 2-D float32 CHM height array.
        transform (rasterio.transform.Affine): Rasterio affine transform.
        crs (rasterio.crs.CRS): Coordinate reference system.
 
    Requirements:
        rasterio, numpy
    """
    
    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return chm, transform, crs