import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from skimage.segmentation import watershed
from skimage.morphology import dilation, disk
import rasterio.transform


def cluster_by_chm_peaks(point_cloud, peak_coords, chm, transform, crown_radius=2.0, min_points=40):
    """
    Segment a point cloud into individual tree clusters using CHM watershed.
 
    Uses marker-controlled watershed segmentation on the CHM to define each
    tree's crown footprint, then assigns 3D points from the point cloud to
    their corresponding watershed region. Watershed respects the natural
    ridgeline between adjacent crowns and handles non-circular or concave
    canopies more accurately than radius-based nearest-peak assignment.
 
    Each CHM peak from find_chm_peaks() becomes one watershed marker (seed).
    The CHM is inverted before flooding so that high-canopy cells become deep
    basins. A hard crown_radius cap is applied after watershed to prevent
    runaway flooding into open ground.
 
    Steps:
        1. Convert peak UTM coords to raster pixel indices
        2. Build marker image (one integer label per peak)
        3. Dilate markers with a 3×3 footprint for robust seeding
        4. Run watershed on the inverted CHM with a vegetation-height mask
        5. Map each 3D point's XY to its raster label
        6. Zero out labels for points further than crown_radius from any peak
        7. Build one PointCloud per label; discard clusters below min_points
 
    Args:
        point_cloud (o3d.geometry.PointCloud): Green-filtered point cloud
            from clean_up_pointcloud().
        peak_coords (np.ndarray): (N, 2) UTM XY peak coordinates from
            find_chm_peaks(). Each peak becomes one watershed seed.
        chm (np.ndarray): 2-D float32 CHM array from build_chm().
        transform (rasterio.transform.Affine): Rasterio affine transform
            from build_chm().
        crown_radius (float): Hard maximum distance in metres from a point
            to its nearest peak. Points beyond this are unassigned regardless
            of watershed label. Default 2.0.
        min_points (int): Minimum points for a cluster to be kept.
            Default 40.
 
    Returns:
        list of o3d.geometry.PointCloud: One cluster per surviving watershed
            region, in peak-index order.
 
    Requirements:
        numpy, open3d, scipy.spatial.KDTree, skimage.segmentation,
        skimage.morphology, rasterio.transform
    """

    points = np.asarray(point_cloud.points)
    xy     = points[:, :2]

    n_rows, n_cols = chm.shape

    # ── step 1: convert peak UTM coords → pixel indices ──────────────────────
    # rasterio.transform.rowcol is the inverse of rasterio.transform.xy
    peak_rows, peak_cols = rasterio.transform.rowcol(transform, peak_coords[:, 0], peak_coords[:, 1])
    peak_rows = np.array(peak_rows)
    peak_cols = np.array(peak_cols)

    # clamp to raster bounds (peaks very near the edge can land 1px outside)
    peak_rows = np.clip(peak_rows, 0, n_rows - 1)
    peak_cols = np.clip(peak_cols, 0, n_cols - 1)

    # ── step 2: build marker image ────────────────────────────────────────────
    # each peak gets a unique integer label starting at 1 (0 = background)
    markers = np.zeros((n_rows, n_cols), dtype=np.int32)
    for i, (r, c) in enumerate(zip(peak_rows, peak_cols), start=1):
        markers[r, c] = i

    # slightly dilate markers so watershed has an easier time seeding
    # (single-pixel seeds sometimes get swallowed by noise in the CHM)
    from skimage.morphology import dilation, footprint_rectangle
    markers = dilation(markers, footprint_rectangle((3, 3)))
    
    # ── step 3: run watershed on the inverted CHM ─────────────────────────────
    # watershed floods *uphill* from markers, so we invert the CHM
    # (high canopy → deep basin, ground → flat plateau)
    chm_filled = np.where(np.isnan(chm), 0.0, chm).astype(np.float32)
    surface    = -chm_filled

    # mask: only flood cells with meaningful vegetation height
    # this stops watershed from bleeding far into bare ground
    mask = chm_filled > 0.5   # metres — tune if needed

    labels_raster = watershed(surface, markers=markers, mask=mask)
    # labels_raster[r, c] == i  means pixel (r,c) belongs to peak i
    # labels_raster[r, c] == 0  means no assignment (outside mask)

    print(f"Watershed produced {labels_raster.max()} labelled regions "
          f"from {len(peak_coords)} peaks")

    # ── step 4: convert each 3D point's XY → raster label ────────────────────
    # rasterio.transform.rowcol vectorised over all points
    pt_rows, pt_cols = rasterio.transform.rowcol(transform, xy[:, 0], xy[:, 1])
    pt_rows = np.array(pt_rows)
    pt_cols = np.array(pt_cols)

    # clamp so out-of-bounds points get label 0 (discarded) rather than crashing
    in_bounds = (
        (pt_rows >= 0) & (pt_rows < n_rows) &
        (pt_cols >= 0) & (pt_cols < n_cols)
    )
    pt_labels = np.zeros(len(points), dtype=np.int32)
    pt_labels[in_bounds] = labels_raster[pt_rows[in_bounds], pt_cols[in_bounds]]

    # ── step 5: optional crown_radius hard cap ────────────────────────────────
    # watershed can still bleed into sparse ground; cap by distance to nearest peak
    peak_tree           = KDTree(peak_coords)
    distances, nn_peak  = peak_tree.query(xy)
    # if a point is too far from any peak, zero out its label
    too_far             = distances > crown_radius
    pt_labels[too_far]  = 0

    kept = (pt_labels > 0).sum()
    print(f"Points assigned by watershed + radius cap: {kept} / {len(points)} "
          f"({100 * kept / len(points):.1f}% kept)")

    # ── step 6: build one PointCloud per label ────────────────────────────────
    clusters = []
    for i in range(1, len(peak_coords) + 1):
        mask_i  = pt_labels == i
        n_pts   = mask_i.sum()

        if n_pts < min_points:
            continue

        cluster = point_cloud.select_by_index(np.where(mask_i)[0])
        clusters.append(cluster)

    print(f"Clusters from watershed: {len(clusters)} "
          f"(of {len(peak_coords)} peaks, min_points={min_points})")

    return clusters