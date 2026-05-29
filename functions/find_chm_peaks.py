import numpy as np
from scipy.ndimage import maximum_filter, label
import rasterio.transform


def find_chm_peaks(chm, transform, min_height=1.5, search_radius_m=1.5, resolution=0.5):
    """
    Desc:
        Find local maxima in a Canopy Height Model raster. Each peak above
        min_height is treated as a candidate tree top. Returns UTM coordinates
        so the peaks can be used as KMeans seeds or DBSCAN pre-filters.

    Args:
        chm, np.ndarray:        2D float32 CHM array from build_chm().
        transform, Affine:      Rasterio affine transform from build_chm().
        min_height, float:      Minimum canopy height (metres) to count as a
                                tree. Filters out ground clutter. Default 1.5.
        search_radius_m, float: Radius (metres) of the local maximum filter
                                window. Should be roughly the minimum expected
                                crown radius. Default 1.5 m.
        resolution, float:      CHM cell size in metres — must match the value
                                used in build_chm(). Default 0.5 m.

    Returns:
        peak_coords, np.ndarray: (N, 2) array of (easting, northing) UTM
                                 coordinates for each detected tree top.
        peak_heights, np.ndarray: (N,) array of CHM heights at each peak.

    Requirements:
        numpy, scipy.ndimage, rasterio.transform
    """

    # ── local maximum filter ──────────────────────────────────────────────────
    # window size in pixels that corresponds to search_radius_m
    window_px = max(3, int(np.ceil(search_radius_m / resolution) * 2 + 1))  # must be odd

    local_max = maximum_filter(chm, size=window_px)

    # a pixel is a peak if it equals the local maximum AND clears min_height
    # also ignore NaN cells (no-data)
    peak_mask = (
        (chm == local_max) &
        (chm >= min_height) &
        (~np.isnan(chm))
    )

    # ── suppress duplicate peaks in flat-top regions ─────────────────────────
    # label connected regions of equal-valued local maxima, keep only centroid
    labeled, n_regions = label(peak_mask)
    rows, cols = [], []
    for region_id in range(1, n_regions + 1):
        region_pixels = np.argwhere(labeled == region_id)
        centroid = region_pixels.mean(axis=0)          # float row, col
        rows.append(centroid[0])
        cols.append(centroid[1])

    rows = np.array(rows)
    cols = np.array(cols)

    if len(rows) == 0:
        print("No CHM peaks found — try lowering min_height or search_radius_m.")
        return np.empty((0, 2)), np.empty((0,))

    # ── convert pixel indices → UTM coordinates ───────────────────────────────
    # rasterio.transform.xy returns (xs, ys) for arrays of rows/cols
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    peak_coords = np.column_stack((xs, ys))

    # height at each peak (nearest-pixel lookup on integer indices)
    peak_heights = chm[rows.astype(int), cols.astype(int)]

    print(f"Found {len(peak_coords)} CHM peaks "
          f"(min_height={min_height} m, window={window_px} px)")

    return peak_coords, peak_heights


def filter_clusters_by_chm_peaks(clusters, peak_coords, max_distance=3.0):
    """
    Desc:
        Keep only clusters that have at least one CHM peak nearby.
        Acts as a fast pre-filter before DBSCAN or KMeans, removing
        ground patches and shrubs that have no corresponding canopy peak.

    Args:
        clusters, list of o3d.PointCloud: Clusters from cluster_pointcloud().
        peak_coords, np.ndarray:          (N, 2) UTM peak coords from find_chm_peaks().
        max_distance, float:              A cluster must have its XY centroid within
                                          this many metres of a peak to be kept.
                                          Default 3.0 m.

    Returns:
        filtered, list of o3d.PointCloud: Clusters that overlap a CHM peak.

    Requirements:
        numpy, scipy.spatial.KDTree
    """
    from scipy.spatial import KDTree
    import numpy as np

    if len(peak_coords) == 0:
        print("Warning: no peaks provided, returning all clusters unfiltered.")
        return clusters

    peak_tree = KDTree(peak_coords)

    filtered = []
    for pcd in clusters:
        points   = np.asarray(pcd.points)
        centroid = points[:, :2].mean(axis=0)          # XY only
        dist, _  = peak_tree.query(centroid)
        if dist <= max_distance:
            filtered.append(pcd)

    print(f"CHM filter: {len(clusters)} → {len(filtered)} clusters "
          f"(max_distance={max_distance} m)")
    return filtered


def chm_peaks_as_kmeans_seeds(peak_coords, cluster_points):
    """
    Desc:
        Given a set of CHM peak coordinates that fall inside a single large
        cluster, return them as initial KMeans centroids. Replaces the
        find_density_peaks() call in split_large_clusters.py.

    Args:
        peak_coords, np.ndarray:   (N, 2) UTM XY coords of peaks inside
                                   this cluster (filter to the cluster bbox
                                   before calling).
        cluster_points, np.ndarray: (M, 3) XYZ point array of the cluster.

    Returns:
        seeds, np.ndarray: (N, 3) seed centroids with Z set to the mean Z
                           of the cluster, ready for KMeans init.

    Requirements:
        numpy
    """
    mean_z = cluster_points[:, 2].mean()
    seeds  = np.column_stack((peak_coords, np.full(len(peak_coords), mean_z)))
    return seeds