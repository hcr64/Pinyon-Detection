import numpy as np


def build_occupancy_grid(points_xy, cell=0.5):
    """
    Rasterise a point cloud's XY footprint into a boolean occupancy grid.

    Args:
        points_xy (np.ndarray): (N, 2) UTM XY coordinates.
        cell (float): Grid cell size in metres. Keep well above the cloud's
            voxel size (0.08 m). Default 0.5.

    Returns:
        tuple: (occ, x0, y0, cell) where occ is a (ncx, ncy) bool array and
            (x0, y0) is the grid origin.

    Requirements:
        numpy
    """
    x0, y0 = points_xy[:, 0].min(), points_xy[:, 1].min()
    ncx = int((points_xy[:, 0].max() - x0) / cell) + 1
    ncy = int((points_xy[:, 1].max() - y0) / cell) + 1
    occ = np.zeros((ncx, ncy), dtype=bool)
    ix = ((points_xy[:, 0] - x0) / cell).astype(int)
    iy = ((points_xy[:, 1] - y0) / cell).astype(int)
    occ[ix, iy] = True
    return occ, x0, y0, cell


def gps_coverage(coords, grid, radius):
    """
    Fraction of the disc of `radius` around each coordinate that has cloud
    points under it. Cells outside the grid count as uncovered.

    Args:
        coords (np.ndarray): (M, 2) UTM XY GPS coordinates.
        grid (tuple): Output of build_occupancy_grid().
        radius (float): Disc radius in metres.

    Returns:
        np.ndarray: (M,) coverage fractions in [0, 1].

    Requirements:
        numpy
    """
    occ, x0, y0, cell = grid
    ncx, ncy = occ.shape

    offs = np.arange(-radius, radius + cell, cell)
    ox, oy = np.meshgrid(offs, offs)
    in_disc = np.hypot(ox, oy) <= radius
    ox, oy = ox[in_disc], oy[in_disc]

    out = np.empty(len(coords))
    for k, (gx, gy) in enumerate(coords):
        ix = np.floor((gx + ox - x0) / cell).astype(int)
        iy = np.floor((gy + oy - y0) / cell).astype(int)
        ok = (ix >= 0) & (ix < ncx) & (iy >= 0) & (iy < ncy)
        hit = np.zeros(len(ix), dtype=bool)
        hit[ok] = occ[ix[ok], iy[ok]]
        out[k] = hit.mean()
    return out


def filter_gps_by_coverage(csv_coords, df_labels, points_xy, radius=6.0,
                           min_coverage=0.8, cell=0.5, verbose=True):
    """
    Drop GPS labels sitting on the fringe of the point cloud.

    A label is kept when at least `min_coverage` of the disc of `radius`
    metres around it has cloud points beneath it. Fringe labels (and labels
    over holes in the reconstruction) fall below the threshold.

    Args:
        csv_coords (np.ndarray): (M, 2) UTM GPS coords, aligned with df_labels.
        df_labels (pd.DataFrame): Label frame, same row order as csv_coords.
        points_xy (np.ndarray): (N, 2) XY of the point cloud.
        radius (float): Disc radius in metres. Default 6.0.
        min_coverage (float): Minimum covered fraction to keep a label.
            Default 0.8.
        cell (float): Occupancy grid cell size in metres. Default 0.5.
        verbose (bool): Print how many labels were dropped. Default True.

    Returns:
        csv_coords (np.ndarray): Filtered coordinates.
        df_labels (pd.DataFrame): Filtered labels, index reset.
        coverage (np.ndarray): (M,) coverage of every ORIGINAL label.

    Requirements:
        numpy, pandas
    """
    grid = build_occupancy_grid(points_xy, cell=cell)
    coverage = gps_coverage(csv_coords, grid, radius)
    keep = coverage >= min_coverage

    if verbose:
        print(f"GPS coverage filter (radius={radius} m, min={min_coverage:.0%}): "
              f"dropped {(~keep).sum()} of {len(keep)} labels")

    return csv_coords[keep], df_labels[keep].reset_index(drop=True), coverage