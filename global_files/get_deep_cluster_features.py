import numpy as np
import pandas as pd
import open3d as o3d
import os

from scipy.spatial import ConvexHull, KDTree
from scipy.stats import skew, kurtosis
from matplotlib.colors import rgb_to_hsv

def make_deep_dataframe(clusters):
    """
    Build a DataFrame of shape, PCA, colour, and derived features from a
    cluster list.

    Each row corresponds to one cluster. The "file" column stores the integer
    index so rows can be joined to df_clusters after label matching. This
    DataFrame is used as the feature matrix for train_tree_classifier().

    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to featurise.

    Returns:
        pd.DataFrame: One row per cluster with columns produced by
            get_deep_cluster_features() plus a "file" index column.

    Requirements:
        numpy, pandas, open3d
    """
    rows = []

    for i, cluster in enumerate(clusters):
        features = get_deep_cluster_features(cluster)
        features["file"] = i
        rows.append(features)

    df_deep = pd.DataFrame(rows)
    return df_deep

def _count_density_peaks(points, k=15, min_peak_distance=1.0, min_density_ratio=1.5):
    """
    Lightweight local density peak counter for a single cluster.

    A scaled-down version of find_density_peaks() in split_large_clusters.py,
    reimplemented here rather than imported to avoid pulling clustering's
    sys.path-hacked io/sklearn dependency chain into global_files/ for a
    single feature. Same core logic: a point is a peak if it's denser than
    all its k neighbours and clears min_density_ratio * mean density;
    peaks closer than min_peak_distance to an already-accepted peak are
    suppressed.

    Intended to run on an already-split cluster (post split_large_clusters())
    as a QC signal — a residual 2+ peak count after splitting may indicate
    multi-stem growth (juniper) that Mean Shift didn't separate, rather than
    a single tree with a wide but unimodal crown.

    Args:
        points (np.ndarray): (N, 3) XYZ array.
        k (int): Neighbour count for local density. Capped at len(points)-1.
            Default 15 (smaller than split_large_clusters' default 30/50
            since this runs on already-small, already-split clusters).
        min_peak_distance (float): Minimum separation in metres between
            accepted peaks. Default 1.0 (tighter than the pre-split 3.0,
            since we're now looking for sub-crown structure within an
            already-reasonably-sized cluster).
        min_density_ratio (float): A candidate peak must be at least this
            many times denser than the cluster mean. Default 1.5.

    Returns:
        int: Number of surviving density peaks.

    Requirements:
        numpy, scipy.spatial.KDTree
    """
    n = len(points)
    if n < 5:
        return 1

    k = min(k, n - 1)
    tree = KDTree(points)
    distances, neighbor_indices = tree.query(points, k=k)
    density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
    mean_density = density.mean()

    peaks = []
    for i, neighbors in enumerate(neighbor_indices):
        if (density[i] == density[neighbors].max() and
                density[i] > mean_density * min_density_ratio):
            peaks.append(i)

    filtered_peaks = []
    for i in peaks:
        too_close = any(
            np.linalg.norm(points[i] - points[j]) < min_peak_distance
            for j in filtered_peaks
        )
        if not too_close:
            filtered_peaks.append(i)

    return max(1, len(filtered_peaks))


def _trunk_gap_score(z, n_bins=20):
    """
    Measure whether a cluster's Z-histogram shows a "trunk gap" — a
    low-density band between a lower trunk mass and an upper crown mass.

    Ponderosa's high, sparse-trunk canopy should show a pronounced dip
    somewhere in the lower-middle of its Z-histogram; pinyon and especially
    juniper (foliage reaching near-ground) should show a much flatter,
    gap-free profile.

    Bins Z into n_bins equal-width bins, finds the bin with minimum count
    within the middle 60% of the height range (excludes the very top/bottom
    bins, which are trivially sparse for any cluster), and reports that
    bin's count as a fraction of the mean bin count elsewhere. A LOW score
    means a pronounced gap (strong trunk signal); a score near 1.0 means no
    gap (foliage fills the profile top to bottom).

    Args:
        z (np.ndarray): (N,) array of point Z values for one cluster.
        n_bins (int): Number of histogram bins. Default 20.

    Returns:
        float: Ratio of the minimum-density middle bin to the mean of all
            other bins. Range roughly [0, 1+]; lower = stronger trunk gap.

    Requirements:
        numpy
    """
    if len(z) < n_bins:
        return 1.0

    counts, _ = np.histogram(z, bins=n_bins)

    # only consider the middle 60% of bins — a dip right at the very top
    # or bottom is just the crown tapering out / ground clearance, not a
    # real trunk gap
    lo = int(n_bins * 0.2)
    hi = int(n_bins * 0.8)
    middle = counts[lo:hi]

    if len(middle) == 0 or middle.sum() == 0:
        return 1.0

    min_bin = middle.min()
    other_bins = np.concatenate([counts[:lo], counts[hi:], middle])
    other_mean = other_bins[other_bins > 0].mean() if (other_bins > 0).any() else 1.0

    return float(min_bin / (other_mean + 1e-6))

def get_deep_cluster_features(pcd):
    """
    Extract shape, PCA, colour, and structural features from a single cluster.

    Used internally by make_deep_dataframe(). Features are chosen to
    discriminate between pinyon, juniper, and ponderosa pine based on crown
    geometry, point distribution, and colour.

    Shape features:
        height       — Z range of the axis-aligned bounding box
        radius       — max XY half-width of the AABB
        n_points     — total point count
        obb_extent_* — oriented bounding box dimensions (x, y, z)

    PCA features (eigenvalues of the 3x3 covariance matrix):
        eigenvalue_1/2/3 — raw eigenvalues (descending order)
        linearity        — (λ1 - λ2) / λ1
        planarity        — (λ2 - λ3) / λ1
        sphericity       — λ3 / λ1

    Colour features (normalised RGB, 0.0-1.0):
        mean_r/g/b  — average colour per channel
        std_r/g/b   — colour variation per channel

    Structural features (target species-specific crown architecture):
        verticality      — height / max horizontal OBB extent; ponderosa is
                           tall and columnar, pinyon is wide and squat
        flatness_ratio   — planarity / height; juniper sprawls flat relative
                           to its height, ponderosa does not
        crown_base_ratio — point density in upper third of crown vs lower
                           third; ponderosa has a high canopy with a sparse
                           lower trunk, pinyon fills more evenly

    Footprint / irregularity features (target juniper's lobed, non-radial
    crown shape vs. pinyon/ponderosa's more circular footprint):
        hull_area_ratio    — XY convex hull area / area of a circle with
                             the cluster's radius; low = irregular footprint
        hull_volume_ratio  — 3D convex hull volume / AABB volume; how
                             "filled" the bounding box actually is
        obb_aspect_ratio   — longer / shorter horizontal OBB extent;
                             1.0 = circular, higher = elongated/asymmetric
        n_density_peaks    — count of residual density peaks post-split;
                             2+ may indicate multi-stem growth Mean Shift
                             didn't separate

    Vertical/radial profile features (richer than crown_base_ratio):
        z_skewness         — skewness of the Z distribution; ponderosa's
                             sparse-trunk/dense-canopy shape should skew
                             strongly negative
        z_kurtosis         — kurtosis of the Z distribution
        trunk_gap_score     — see _trunk_gap_score(); low = pronounced
                             trunk gap, high = foliage fills top-to-bottom
        radial_density_gradient — point density in inner half-radius vs.
                             outer half-radius (XY); low = mass hugs the
                             trunk, high = sprawls to the crown edge

    Colour features, reframed (raw RGB/ExG contributes little per feature
    importances; HSV and bark exposure may separate species along axes
    chromaticity doesn't):
        hue_mean, hue_std   — mean/std of per-point hue (HSV); illumination-
                             invariant along a different axis than chroma
        saturation_mean     — mean per-point saturation
        bark_fraction       — fraction of points in the bottom Z-quartile
                             that are non-green (ExG < 0); proxy for visible
                             trunk/bark exposure near ground

    Chromaticity features (shadow-robust — per-point RGB normalised by
    per-point brightness before averaging, so a shaded/dark point and a
    sunlit point of the same true colour contribute similarly instead of
    the shaded one dragging mean_r/g/b down):
        chroma_r/g/b   — mean per-point chromaticity (r/(r+g+b) etc.)
        chroma_g_std   — std of per-point green chromaticity; a tight
                        distribution means colour is stable across the
                        crown regardless of shadow, a wide one flags a
                        cluster straddling sun/shade
        exg_chroma     — ExG computed on chromaticity-normalised colour
                        (2*chroma_g - chroma_r - chroma_b) rather than raw
                        RGB; same greenness signal, shadow-corrected

    Colour extremes (top-fraction ranked — extends the ExG-style top-
    fraction approach from has_green_crown() to other channels; targets
    juniper's distinctive yellow/orange crown tones and exposed bark):
        top_red_exr    — mean Excess Red (2r-g-b) of the reddest 20% of points
        top_blue_exb   — mean Excess Blue (2b-r-g) of the bluest 20% of points
        top_yellow_exy — mean Excess Yellow (r+g-2b) of the yellowest 20%
        top_grey       — mean achromaticity (-(max channel - min channel))
                        of the greyest 20% of points; proxy for exposed
                        bark/deadwood surfaces

    Args:
        pcd (o3d.geometry.PointCloud): A single cluster with colour data.

    Returns:
        dict: Feature name -> scalar value for all features listed above.

    Requirements:
        numpy, open3d, scipy.spatial, scipy.stats, matplotlib.colors
    """

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # ── bounding boxes ────────────────────────────────────────────────────────
    aabb = pcd.get_axis_aligned_bounding_box()
    obb  = pcd.get_oriented_bounding_box()

    bounds = aabb.max_bound - aabb.min_bound
    height = bounds[2]
    radius = max(bounds[:2]) / 2

    # ── PCA ───────────────────────────────────────────────────────────────────
    cov     = np.cov(points.T)
    eigvals = np.linalg.eigvals(cov)
    eigvals = np.sort(np.abs(eigvals))[::-1]

    λ1, λ2, λ3 = eigvals[0], eigvals[1], eigvals[2]
    linearity  = (λ1 - λ2) / (λ1 + 1e-6)
    planarity  = (λ2 - λ3) / (λ1 + 1e-6)
    sphericity = λ3 / (λ1 + 1e-6)

    # ── colour ────────────────────────────────────────────────────────────────
    mean_color = colors.mean(axis=0)
    std_color  = colors.std(axis=0)

    # ── structural features ───────────────────────────────────────────────────
    max_horiz_extent = max(obb.extent[0], obb.extent[1])
    verticality = height / (max_horiz_extent + 1e-6)
    flatness_ratio = planarity / (height + 1e-6)

    z         = points[:, 2]
    z_min     = z.min()
    z_max     = z.max()
    z_range   = z_max - z_min + 1e-6

    lower_mask = z < (z_min + z_range / 3)
    upper_mask = z > (z_max - z_range / 3)

    n_lower = lower_mask.sum()
    n_upper = upper_mask.sum()

    lower_density = n_lower / (z_range / 3 + 1e-6)
    upper_density = n_upper / (z_range / 3 + 1e-6)
    crown_base_ratio = upper_density / (lower_density + 1e-6)

    # ── footprint / irregularity features ─────────────────────────────────────
    xy = points[:, :2]
    try:
        hull_2d = ConvexHull(xy)
        hull_area = hull_2d.volume  # scipy quirk: 2D ConvexHull .volume == area
    except Exception:
        hull_area = 0.0
    circle_area = np.pi * (radius ** 2) + 1e-6
    hull_area_ratio = hull_area / circle_area

    try:
        hull_3d = ConvexHull(points)
        hull_volume = hull_3d.volume
    except Exception:
        hull_volume = 0.0
    aabb_volume = bounds[0] * bounds[1] * bounds[2] + 1e-6
    hull_volume_ratio = hull_volume / aabb_volume

    horiz_extents = sorted([obb.extent[0], obb.extent[1]])
    obb_aspect_ratio = horiz_extents[1] / (horiz_extents[0] + 1e-6)

    n_density_peaks = _count_density_peaks(points)

    # ── vertical/radial profile features ──────────────────────────────────────
    z_skewness = float(skew(z)) if len(z) >= 8 else 0.0
    z_kurtosis = float(kurtosis(z)) if len(z) >= 8 else 0.0
    trunk_gap_score = _trunk_gap_score(z)

    centroid_xy = xy.mean(axis=0)
    dist_from_center = np.linalg.norm(xy - centroid_xy, axis=1)
    half_radius = radius / 2 + 1e-6
    inner_mask = dist_from_center <= half_radius
    outer_mask = ~inner_mask

    inner_area = np.pi * half_radius ** 2 + 1e-6
    outer_area = circle_area - inner_area + 1e-6
    inner_density = inner_mask.sum() / inner_area
    outer_density = outer_mask.sum() / outer_area
    radial_density_gradient = inner_density / (outer_density + 1e-6)

    # ── colour, reframed ───────────────────────────────────────────────────────
    hsv = rgb_to_hsv(np.clip(colors, 0.0, 1.0))
    hue_mean        = float(hsv[:, 0].mean())
    hue_std         = float(hsv[:, 0].std())
    saturation_mean = float(hsv[:, 1].mean())

    z_q1 = np.percentile(z, 25)
    bottom_mask = z <= z_q1
    if bottom_mask.sum() > 0:
        bottom_colors = colors[bottom_mask]
        exg_bottom = 2 * bottom_colors[:, 1] - bottom_colors[:, 0] - bottom_colors[:, 2]
        bark_fraction = float((exg_bottom < 0.0).mean())
    else:
        bark_fraction = 0.0

    # ── chromaticity features (shadow-robust) ─────────────────────────────────
    # normalise each point's RGB by its own brightness BEFORE averaging, so a
    # shaded point of true colour (r,g,b) and a sunlit point of the same true
    # colour but higher magnitude contribute equally to the mean — mean_r/g/b
    # above do not have this property, since they average raw (shadow-scaled)
    # values directly.
    r_ch, g_ch, b_ch = colors[:, 0], colors[:, 1], colors[:, 2]
    brightness   = r_ch + g_ch + b_ch + 1e-6
    chroma_r_pp  = r_ch / brightness
    chroma_g_pp  = g_ch / brightness
    chroma_b_pp  = b_ch / brightness
    exg_chroma_pp = 2 * chroma_g_pp - chroma_r_pp - chroma_b_pp

    chroma_r     = float(chroma_r_pp.mean())
    chroma_g     = float(chroma_g_pp.mean())
    chroma_b     = float(chroma_b_pp.mean())
    chroma_g_std = float(chroma_g_pp.std())
    exg_chroma   = float(exg_chroma_pp.mean())

    # ── colour extremes (top-fraction ranked) ─────────────────────────────────
    exr  = 2 * r_ch - g_ch - b_ch          # excess red
    exb  = 2 * b_ch - r_ch - g_ch          # excess blue
    exy  = r_ch + g_ch - 2 * b_ch          # excess yellow
    grey = -(colors.max(axis=1) - colors.min(axis=1))  # higher = more achromatic

    top_red_exr    = _top_fraction_mean(exr,  top_fraction=0.20)
    top_blue_exb   = _top_fraction_mean(exb,  top_fraction=0.20)
    top_yellow_exy = _top_fraction_mean(exy,  top_fraction=0.20)
    top_grey       = _top_fraction_mean(grey, top_fraction=0.20)

    # ── assemble feature dict ─────────────────────────────────────────────────
    features = {
        # shape
        "height":           height,
        "radius":           radius,
        "n_points":         len(points),
        "obb_extent_x":     obb.extent[0],
        "obb_extent_y":     obb.extent[1],
        "obb_extent_z":     obb.extent[2],

        # PCA
        "eigenvalue_1":     λ1,
        "eigenvalue_2":     λ2,
        "eigenvalue_3":     λ3,
        "linearity":        linearity,
        "planarity":        planarity,
        "sphericity":       sphericity,

        # colour
        "mean_r":           mean_color[0],
        "mean_g":           mean_color[1],
        "mean_b":           mean_color[2],
        "std_r":            std_color[0],
        "std_g":            std_color[1],
        "std_b":            std_color[2],

        # structural
        "verticality":      verticality,
        "flatness_ratio":   flatness_ratio,
        "crown_base_ratio": crown_base_ratio,

        # footprint / irregularity
        "hull_area_ratio":   hull_area_ratio,
        "hull_volume_ratio": hull_volume_ratio,
        "obb_aspect_ratio":  obb_aspect_ratio,
        "n_density_peaks":   n_density_peaks,

        # vertical/radial profile
        "z_skewness":              z_skewness,
        "z_kurtosis":              z_kurtosis,
        "trunk_gap_score":         trunk_gap_score,
        "radial_density_gradient": radial_density_gradient,

        # colour, reframed
        "hue_mean":        hue_mean,
        "hue_std":          hue_std,
        "saturation_mean":  saturation_mean,
        "bark_fraction":    bark_fraction,

        # chromaticity (shadow-robust)
        "chroma_r":      chroma_r,
        "chroma_g":      chroma_g,
        "chroma_b":      chroma_b,
        "chroma_g_std":  chroma_g_std,
        "exg_chroma":    exg_chroma,

        # colour extremes (top-fraction ranked)
        "top_red_exr":    top_red_exr,
        "top_blue_exb":   top_blue_exb,
        "top_yellow_exy": top_yellow_exy,
        "top_grey":       top_grey,
    }

    return features


def engineer_features(df):
    """
    Add derived ratio features to the deep cluster DataFrame.

    These are computed from existing columns rather than raw point cloud
    geometry, so they are cheaper to recompute and easier to iterate on than
    the structural features in get_deep_cluster_features(). Call this after
    make_deep_dataframe() and before train_tree_classifier().

    New columns added:
        height_to_radius — ponderosa tends tall+narrow (high), pinyon
                           short+wide (low)
        green_dominance  — mean green minus average of red and blue; juniper
                           tends lower than pinyon
        crown_volume     — radius² × height; proxy for total canopy mass
        color_saturation — sum of per-channel std; ponderosa bark pulls
                           std_r higher

    Args:
        df (pd.DataFrame): Output of make_deep_dataframe().

    Returns:
        pd.DataFrame: Same DataFrame with four additional columns appended.

    Requirements:
        pandas
    """
    df = df.copy()

    df["height_to_radius"] = df["height"] / (df["radius"] + 1e-6)
    df["green_dominance"]  = df["mean_g"] - (df["mean_r"] + df["mean_b"]) / 2
    df["crown_volume"]     = df["radius"] ** 2 * df["height"]
    df["color_saturation"] = df["std_r"] + df["std_g"] + df["std_b"]

    return df


def get_color_extremes_features(pcd, top_fraction=0.20):
    """
    Rank points by red/blue/yellow/grey dominance indices and return the
    mean of the top top_fraction for each — same logic as has_green_crown()'s
    ExG check, but for classification features rather than a pass/fail filter.

    Whole-cluster mean_r/g/b (already in get_deep_cluster_features) washes
    out a signal that only covers part of a crown. Ranking + taking the top
    fraction preserves it — e.g. a juniper with 15% yellow-orange senescent
    foliage still shows a strong top_yellow_exy even though its mean color
    looks green overall.

    Indices (mirrors ExG = 2g - r - b):
        ExR   = 2r - g - b                 — excess red
        ExB   = 2b - r - g                 — excess blue
        ExY   = (r + g) - 2b               — excess yellow/orange, motivated
                                              by juniper's yellow/orange tinge
        Grey  = 1 - (max(r,g,b)-min(r,g,b)) — achromaticity; high for bark,
                                              dead/dry material, rock, shadow.
                                              Not a species-color signal on
                                              its own — a QC / crown-exposure
                                              proxy, potentially useful
                                              alongside crown_base_ratio for
                                              flagging sparse-crown species.

    Args:
        pcd (o3d.geometry.PointCloud): A single cluster with colour data,
            normalised RGB (0.0-1.0).
        top_fraction (float): Fraction of points to inspect, ranked by each
            index descending. Default 0.20 (same default as has_green_crown).

    Returns:
        dict: top_red_exr, top_blue_exb, top_yellow_exy, top_grey — mean of
            the index over the top top_fraction of points ranked by that
            index.

    Requirements:
        numpy, open3d
    """
    empty = {"top_red_exr": 0.0, "top_blue_exb": 0.0,
             "top_yellow_exy": 0.0, "top_grey": 0.0}

    if not pcd.has_colors():
        return empty

    colors = np.asarray(pcd.colors)
    if len(colors) < 5:
        return empty

    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    exr  = 2 * r - g - b
    exb  = 2 * b - r - g
    exy  = (r + g) - 2 * b
    grey = 1 - (np.max(colors, axis=1) - np.min(colors, axis=1))

    n_top = max(1, int(len(colors) * top_fraction))

    def top_mean(arr):
        return float(np.partition(arr, -n_top)[-n_top:].mean())

    return {
        "top_red_exr":    top_mean(exr),
        "top_blue_exb":   top_mean(exb),
        "top_yellow_exy": top_mean(exy),
        "top_grey":       top_mean(grey),
    }


def _top_fraction_mean(values, top_fraction=0.20):
    """
    Mean of the top `top_fraction` largest values in a 1D array.

    Extends the ExG-style greenness-ranking approach from
    filter_green_crown.has_green_crown() to the other colour-extreme
    indices below. Ranking by the top slice rather than the cluster-wide
    mean matters here specifically because the signal (e.g. juniper's
    yellow-orange crown tips, exposed bark near the base) is often
    concentrated in a minority of points — averaging over the whole
    cluster would dilute it into noise.

    Args:
        values (np.ndarray): (N,) per-point index values.
        top_fraction (float): Fraction of points to average, ranked
            descending. Default 0.20.

    Returns:
        float: Mean of the top `top_fraction` values. 0.0 if values is empty.

    Requirements:
        numpy
    """
    n = len(values)
    if n == 0:
        return 0.0
    n_top = max(1, int(n * top_fraction))
    top_vals = np.partition(values, -n_top)[-n_top:]
    return float(top_vals.mean())