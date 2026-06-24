import numpy as np
from scipy.ndimage import maximum_filter, label
import rasterio.transform


def find_chm_peaks(chm, transform, min_height=1.5, search_radius_m=1.5, resolution=0.5):
    """
    Detect local maxima in a Canopy Height Model raster as candidate tree tops.
 
    Applies scipy.ndimage.maximum_filter with a window sized to search_radius_m
    and marks pixels that equal the local maximum and exceed min_height as peaks.
    Flat-topped regions (multiple equal-valued adjacent pixels) are collapsed to
    their centroid. Pixel indices are converted to UTM coordinates via the
    rasterio transform.
 
    Returned peak coordinates are used as watershed markers in
    cluster_by_chm_peaks() and as Mean Shift seeds in split_large_clusters().
 
    Args:
        chm (np.ndarray): 2-D float32 CHM array from build_chm().
        transform (rasterio.transform.Affine): Affine transform from build_chm().
        min_height (float): Minimum canopy height in metres to qualify as a
            tree. Filters out low ground clutter. 1.0 m outperformed 1.5 m
            at Sunset Crater. Default 1.5.
        search_radius_m (float): Radius in metres of the local maximum filter
            window. Roughly the minimum expected crown radius — two peaks
            closer than this will be merged. 3.0 m outperformed smaller values
            at Sunset Crater. Default 1.5.
        resolution (float): CHM cell size in metres. Must match the value
            used in build_chm(). Default 0.5.
 
    Returns:
        peak_coords (np.ndarray): (N, 2) array of (easting, northing) UTM
            coordinates for each detected tree top.
        peak_heights (np.ndarray): (N,) array of CHM heights at each peak.
 
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
    Discard clusters that have no CHM peak nearby.
 
    Compares each cluster's XY centroid against the peak coordinate set and
    removes clusters further than max_distance from any peak. Acts as a fast
    pre-filter to eliminate ground patches and shrubs before further processing.
 
    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to filter.
        peak_coords (np.ndarray): (N, 2) UTM peak coordinates from
            find_chm_peaks().
        max_distance (float): Maximum distance in metres from a cluster
            centroid to the nearest peak. Default 3.0.
 
    Returns:
        list of o3d.geometry.PointCloud: Clusters whose centroid is within
            max_distance of at least one peak.
 
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