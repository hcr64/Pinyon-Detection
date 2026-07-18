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
    save_path=None,
):
    """
    Build a CHM using a DSM produced externally by photogrammetry software
    (e.g. PixMapper4D's 1_dsm/*.tif) instead of build_chm()'s own per-cell
    max-Z DSM computed from the raw point cloud.

    Why: PixMapper4D's DSM comes from the full densified point cloud/mesh
    with proper interpolation and gap-filling, and is generally more
    accurate than a coarse groupby-max on a voxel-downsampled cloud. This
    keeps CHM = DSM - DTM but lets you plug in the externally-produced DSM.

    Two DTM modes:
        1. dtm_path given — reads a second GeoTIFF (e.g. from PixMapper4D's
           optional DTM export) and reprojects it onto the DSM's grid if
           resolution/extent differ.
        2. dtm_path is None — computes a DTM the same way build_chm() does
           (low Z-percentile per cell) but rasterized onto the *external
           DSM's* grid. Requires `point_cloud`.

    A bounds-overlap sanity check runs whenever point_cloud is provided,
    since PixMapper4D and your .las exports could in principle disagree on
    CRS/UTM zone — this would silently produce a CHM that doesn't line up
    with your point cloud, so it's flagged rather than assumed away.

    Args:
        dsm_path (str): Path to the external DSM GeoTIFF, e.g.
            ".../1_dsm/dsm.tif" from a PixMapper4D export.
        point_cloud (o3d.geometry.PointCloud | None): Raw point cloud.
            Required if dtm_path is None (mode 2); also used for the
            bounds sanity check if provided in mode 1. Must be in the same
            CRS/units as the DSM (UTM metres — EPSG:26912 for Sunset Crater).
        dtm_path (str | None): Path to an external DTM GeoTIFF, if
            available. Default None.
        ground_percentile (int): Ground Z percentile used in mode 2 only.
            Default 5 — matches build_chm()'s default.
        save_path (str | None): Where to save the resulting CHM GeoTIFF.
            Default None (skip saving).

    Returns:
        chm (np.ndarray): 2-D float32 CHM array, same shape as the DSM.
        transform (rasterio.transform.Affine): DSM's affine transform.
        crs (rasterio.crs.CRS): DSM's CRS.

    Requirements:
        numpy, pandas, rasterio, rasterio.warp, open3d (mode 2 / bounds check)
    """
    from rasterio.warp import reproject, Resampling

    # ── read the external DSM ──────────────────────────────────────────────
    with rasterio.open(dsm_path) as src:
        dsm       = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        n_rows, n_cols = dsm.shape
        nodata    = src.nodata

    if nodata is not None:
        dsm = np.where(dsm == nodata, np.nan, dsm)

    print(f"External DSM loaded: {dsm_path}  ({n_cols}×{n_rows} px, crs={crs})")

    # ── bounds sanity check (catches CRS/UTM-zone mismatches early) ──────────
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
            print("⚠  WARNING: point cloud and DSM bounding boxes do not overlap. "
                  "Check that both are in the same CRS (expected UTM zone 12N / "
                  "EPSG:26912) — PixMapper4D may have exported in a different "
                  "CRS than your .las files.")

    # ── get the DTM ────────────────────────────────────────────────────────
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
                source=dtm_raw,
                destination=dtm,
                src_transform=dtm_transform,
                src_crs=dtm_crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )

    else:
        if point_cloud is None:
            raise ValueError(
                "build_chm_from_external_dsm: no dtm_path given, so a "
                "point_cloud is required to compute the DTM (mode 2)."
            )

        print("No external DTM provided — computing DTM from point cloud "
              f"on the DSM's grid ({ground_percentile}th percentile)...")

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        rows, cols = rasterio.transform.rowcol(transform, x, y)
        rows = np.array(rows)
        cols = np.array(cols)

        in_bounds = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        rows, cols, z_in = rows[in_bounds], cols[in_bounds], z[in_bounds]

        if in_bounds.sum() == 0:
            raise ValueError(
                "No point cloud points fall inside the DSM raster bounds — "
                "see the bounds warning above, this is almost certainly a "
                "CRS mismatch."
            )

        df = pd.DataFrame({"row": rows, "col": cols, "z": z_in})
        dtm_series = df.groupby(["row", "col"])["z"].quantile(ground_percentile / 100.0)

        dtm = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        for (r, c), val in dtm_series.items():
            dtm[r, c] = val

    # ── CHM = DSM - DTM, clipped to [0, inf) ─────────────────────────────────
    chm = np.where(np.isnan(dsm) | np.isnan(dtm), np.nan, dsm - dtm).astype(np.float32)
    chm = np.where(np.isnan(chm), np.nan, np.clip(chm, 0, None))

    # ── optional save ────────────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
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