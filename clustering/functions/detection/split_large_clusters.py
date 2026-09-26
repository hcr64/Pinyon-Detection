"""
split_large_clusters.py  (colour-augmented)
────────────────────────────────────────────────────────────────────────────────
Drop-in replacement for clustering/functions/detection/split_large_clusters.py.

Backward compatible: every function keeps its original signature plus new
optional keyword args, all defaulting to the ORIGINAL XY-only behaviour
(use_color=False). Existing callers (run_clustering.py's
split_large_clusters(clusters, min_points=..., max_radius=..., ...) call)
need no changes to keep working exactly as before.

Why this exists
────────────────
find_density_peaks() and the Mean Shift split downstream of it only ever
look at (x, y). That correctly separates two trees with a visible gap
between trunks, but is blind to two crowns that occupy the *same* XY
footprint at different heights or interleaved in a mixed canopy — the
pattern diagnose_cluster_bimodality.py's "color_bimodal_spatial_unimodal"
flag exists to catch. This file adds an optional path where BOTH the
density-peak gate and the Mean Shift split itself operate on an augmented
feature space (XY + weighted per-point chromaticity) instead of XY alone,
so two spatially-overlapping, differently-coloured crowns can be pulled
apart even when their XY density looks unimodal.

How the augmentation works
────────────────────────────
Per-point shadow-robust chromaticity (same normalisation used throughout
get_deep_cluster_features.py — divide each point's RGB by its own
brightness before anything else, so a shaded half of a crown doesn't
masquerade as a colour difference) is computed for each requested channel,
multiplied by `color_weight`, and concatenated onto the raw XY columns:

    split_features = [x, y, color_weight * chroma_g, color_weight * chroma_r, ...]

color_weight is in "metres-equivalent" units — it controls how much a given
amount of chromaticity difference (which lives in [0, 1]) counts as
equivalent to how much XY separation (in metres) when Mean Shift measures
distance between points. There's no principled default; SWEEP THIS. A
starting point of 10.0 means a full chroma_g swing from 0 to 1 counts the
same as 10 metres of XY separation — given typical inter-species chroma_g
differences are more like 0.05-0.15, that puts a "real" colour difference
somewhere around 0.5-1.5 "metres" of pull, comparable to genuine spatial
separation. Tune against known merged-tree clusters (e.g. ones flagged by
diagnose_cluster_bimodality.py) rather than guessing blind.

min_peak_distance keeps its original meaning (max Mean Shift bandwidth /
minimum peak separation) but now operates in this augmented space when
use_color=True — so its effective "metres" interpretation shifts slightly
once colour is contributing distance too. This is intentional: it lets one
bandwidth control both spatial and colour-driven splitting jointly, same
as the original single-purpose spatial bandwidth did for XY alone.

Requirements
────────────
    numpy, open3d, scipy.spatial.KDTree, scikit-learn (MeanShift)
"""

import numpy as np
import open3d as o3d
import os
import re
from scipy.spatial import KDTree
from sklearn.cluster import MeanShift, estimate_bandwidth


# ── save-descriptive helper (deliberately inlined, not imported) ──────────────
# Same logic as clustering/functions/io/save_clusters_descriptive.py.
# Reimplemented here rather than imported so this file has no dependency on
# `functions.io...` resolving as a bare top-level package — that only
# happens automatically when this file runs as part of run_clustering.py's
# own execution (Python auto-adds a script's own directory to sys.path).
# Importing this module standalone (e.g. in a notebook) has no equivalent,
# so the original absolute import broke outside that one specific context.
# This mirrors the project's existing tolerance for this kind of small
# duplication (see "FEATURES is duplicated, not shared" in
# modelling/README.md, and _per_point_chroma in diagnose_cluster_bimodality.py).

def _save_clusters_descriptive(clusters, filenames, save_path):
    """
    Save a list of point cloud clusters to disk with caller-supplied,
    human-readable filenames instead of generic cluster0.ply, cluster1.ply
    naming. Clears the destination folder before writing.

    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to save.
        filenames (list of str): One filename per cluster, same length and
            order as `clusters`. ".ply" appended if not already present.
            Sanitised but NOT deduplicated — pass unique names.
        save_path (str): Directory to write .ply files into. Created
            automatically if it does not exist.

    Returns:
        list of str: Full paths written, in input order.

    Requirements:
        open3d, os, re
    """
    if len(clusters) != len(filenames):
        raise ValueError(
            f"_save_clusters_descriptive: clusters ({len(clusters)}) and "
            f"filenames ({len(filenames)}) must be the same length"
        )

    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))
    else:
        os.makedirs(save_path)

    illegal = re.compile(r'[<>:"/\\|?*]')

    written = []
    seen = set()
    for cluster, name in zip(clusters, filenames):
        clean = illegal.sub("_", name)
        if not clean.endswith(".ply"):
            clean += ".ply"

        if clean in seen:
            print(f"Warning: duplicate filename '{clean}' — this write will "
                  f"overwrite a previous cluster with the same name.")
        seen.add(clean)

        out_path = os.path.join(save_path, clean)
        o3d.io.write_point_cloud(out_path, cluster)
        written.append(out_path)

    print(f"All {len(clusters)} clusters saved to {save_path} with descriptive names")
    return written


# ── colour helper (deliberately duplicated, not imported) ─────────────────────
# Same shadow-robust chromaticity normalisation as
# global_files/get_deep_cluster_features.py. Reimplemented here rather than
# imported to avoid pulling modelling/'s import chain into clustering/ for a
# three-line function — this project already tolerates this kind of small
# duplication elsewhere (see "FEATURES is duplicated, not shared" in
# modelling/README.md); if that duplication ever gets centralised, fold this
# in too.

def _per_point_chroma(colors, channels=("chroma_g", "chroma_r", "chroma_b")):
    """
    Shadow-robust per-point chromaticity for the requested channels.

    Args:
        colors (np.ndarray): (N, 3) normalised RGB, 0.0-1.0.
        channels (tuple[str]): Which of "chroma_r", "chroma_g", "chroma_b"
            to compute and return, in the given order. Default all three.

    Returns:
        np.ndarray: (N, len(channels)) array, columns in the order given.

    Requirements:
        numpy
    """
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    brightness = r + g + b + 1e-6
    lookup = {
        "chroma_r": r / brightness,
        "chroma_g": g / brightness,
        "chroma_b": b / brightness,
    }
    return np.column_stack([lookup[c] for c in channels])


# ── validation (unchanged) ─────────────────────────────────────────────────────

def filter_cluster(pcd, min_height=1.0, min_radius=0.3):
    """
    Return True if a cluster meets minimum geometric criteria to be a tree.

    Rejects clusters that are too flat (likely grass patches) or too narrow
    (likely isolated shrubs or noise).

    Args:
        pcd (o3d.geometry.PointCloud): Cluster to evaluate.
        min_height (float): Minimum Z range in metres. Default 1.0.
        min_radius (float): Minimum XY half-width in metres. Default 0.3.

    Returns:
        bool: True if the cluster passes both thresholds, False otherwise.

    Requirements:
        numpy, open3d
    """
    points = np.asarray(pcd.points)
    aabb   = pcd.get_axis_aligned_bounding_box()
    bounds = aabb.max_bound - aabb.min_bound

    height = bounds[2]
    radius = max(bounds[:2]) / 2

    if height < min_height:
        return False
    if radius < min_radius:
        return False
    return True


# ── main entry point ────────────────────────────────────────────────────────

def split_large_clusters(clusters, min_points=10, max_radius=2.0,
                         min_peak_distance=3.0, k=50, min_density_ratio=1.5,
                         save_pre_split_path=None,
                         use_color=False, color_weight=10.0,
                         color_channels=("chroma_g", "chroma_r", "chroma_b"),
                         force_check_indices=None):
    """
    Split oversized clusters that likely contain multiple merged tree crowns.

    Large clusters (XY radius > max_radius) are inspected for multiple
    density peaks using find_density_peaks(). Only clusters with two or
    more well-separated peaks proceed to splitting — wide single-tree
    crowns are passed through unchanged. Accepted candidates are split with
    Mean Shift clustering, which discovers the number of sub-crowns
    automatically without requiring n_clusters.

    With use_color=False (default), this is byte-for-byte the original
    XY-only behaviour. With use_color=True, both find_density_peaks() and
    the Mean Shift split itself operate on XY + weighted chromaticity
    instead of XY alone — see the module docstring for why and how to tune
    color_weight. Clusters with no colour data fall back to XY-only
    splitting automatically (with a warning) regardless of use_color, since
    there's nothing to augment with.

    Each sub-cluster is validated by filter_cluster() before being
    accepted. Sub-clusters that fail validation are discarded individually;
    the split is only fully reverted if no sub-clusters survive validation
    at all. Pre-split clusters are optionally saved for post-run
    inspection, named descriptively (radius + sub-crown count).

    Args:
        clusters (list of o3d.geometry.PointCloud): Input clusters from
            cluster_by_chm_peaks() or cluster_pointcloud().
        min_points (int): Minimum points for any cluster (or sub-cluster
            after splitting) to be kept. Default 10.
        max_radius (float): XY radius threshold in metres below which a
            cluster is not split — UNLESS its index is in
            force_check_indices (see below). Default 2.0.
        min_peak_distance (float): Used both as the minimum separation
            between density peaks in find_density_peaks() and as the Mean
            Shift bandwidth. Roughly the minimum expected trunk-to-trunk
            distance in metres when use_color=False; when use_color=True,
            operates in the combined XY+colour space (see module
            docstring). Default 3.0.
        k (int): Kept for API compatibility; unused since Mean Shift
            replaced KMeans.
        min_density_ratio (float): Passed to find_density_peaks() to gate
            splitting. A peak must be this many times denser than the
            cluster mean to qualify. Default 1.5.
        save_pre_split_path (str | None): Directory to save clusters before
            splitting, for inspection. Pass None to skip. Default None.
        use_color (bool): Augment both the density-peak gate and the Mean
            Shift split with weighted chromaticity, not just XY. Default
            False (identical to the original function).
        color_weight (float): "Metres-equivalent" scale factor applied to
            chromaticity before concatenating onto XY. SWEEP THIS — see
            module docstring for a starting-point rationale. Only used
            when use_color=True. Default 10.0.
        color_channels (tuple[str]): Which chromaticity channels to
            include when use_color=True. Default ("chroma_g", "chroma_r",
            "chroma_b") — all three; drop channels that don't separate
            species well in your feature_separability plots if you want a
            narrower, less noisy augmentation.
        force_check_indices (set[int] | list[int] | None): Positions in
            `clusters` (matching whatever index the caller used to
            identify them — e.g. df_clusters["file"]) that should proceed
            to the density-peak / Mean Shift check regardless of whether
            their radius clears max_radius. Exists because two heavily
            overlapping crowns can produce a COMPACT combined footprint
            rather than a wide one — max_radius alone will never flag them
            for splitting, no matter how obviously merged they are by
            colour or point density. Build this list with
            diagnose_cluster_bimodality.get_color_spatial_override_candidates()
            rather than passing an arbitrary list — that function already
            validates bimodal-AND-spatially-separated colour evidence, so
            the override stays targeted at genuine spatial merges instead
            of forcing every cluster through the expensive check. Default
            None (identical to the original function — nothing bypasses
            max_radius).

    Returns:
        list of o3d.geometry.PointCloud: Clusters after splitting.
            Length >= len(clusters).

    Requirements:
        numpy, open3d, scipy.spatial.KDTree, scikit-learn.cluster.MeanShift
    """

    final_clusters = []
    pre_split_clusters = []
    pre_split_names = []

    n_color_fallback = 0
    force_check_indices = set(force_check_indices) if force_check_indices else set()
    n_force_checked = 0

    for cluster_idx, pcd in enumerate(clusters):
        points = np.asarray(pcd.points)

        # gate 1: ignore tiny clusters entirely
        if len(points) < min_points:
            continue

        # gate 2: only consider splitting if the cluster is actually large
        # — OR its index was explicitly forced through by the caller
        # (see force_check_indices docstring above: a compact combined
        # footprint from two overlapping crowns can never clear a radius
        # threshold, however merged the cluster actually is).
        aabb   = pcd.get_axis_aligned_bounding_box()
        width  = max((aabb.max_bound - aabb.min_bound)[:2])
        radius = width / 2

        is_forced = cluster_idx in force_check_indices
        if radius <= max_radius and not is_forced:
            final_clusters.append(pcd)
            continue
        if is_forced and radius <= max_radius:
            n_force_checked += 1
            print(f"Cluster {cluster_idx} (radius={radius:.2f}m, below "
                  f"max_radius={max_radius}) force-checked via "
                  f"force_check_indices override")

        # ── determine whether colour augmentation is actually usable ─────────
        cluster_use_color = use_color and pcd.has_colors()
        if use_color and not pcd.has_colors():
            n_color_fallback += 1

        colors = np.asarray(pcd.colors) if cluster_use_color else None

        # only reach here if the cluster is genuinely large
        # check for multiple density peaks before attempting a split
        peaks, _ = find_density_peaks(
            points,
            colors=colors,
            k=min(k, len(points) - 1),
            min_peak_distance=min_peak_distance,
            min_density_ratio=min_density_ratio,
            use_color=cluster_use_color,
            color_weight=color_weight,
            color_channels=color_channels,
        )
        n_peaks = len(peaks)

        if n_peaks < 2:
            # wide cluster but only one density core — still just one tree
            final_clusters.append(pcd)
            continue

        mode_str = "XY+colour" if cluster_use_color else "XY-only"
        print(f"Found {n_peaks} density peaks ({mode_str}) in cluster "
              f"(radius={radius:.2f}m), attempting Mean Shift split...")

        # ── build the feature space Mean Shift actually splits on ─────────────
        split_features = points[:, :2]  # xy, always the base
        if cluster_use_color:
            chroma = _per_point_chroma(colors, channels=color_channels)
            split_features = np.column_stack([split_features, chroma * color_weight])

        # subsample for bandwidth estimation if the cluster is huge
        # (estimate_bandwidth is O(n^2) so cap at 2000 points)
        sample_size = min(len(split_features), 2000)
        rng         = np.random.default_rng(42)
        sample_feat = split_features[rng.choice(len(split_features), sample_size, replace=False)]

        # bandwidth = min_peak_distance lets you control "how far apart must
        # two crowns be to be counted separately" directly. To let sklearn
        # estimate it from data density instead, uncomment:
        # bandwidth = estimate_bandwidth(sample_feat, quantile=0.15)
        bandwidth = min_peak_distance

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=5)
        ms.fit(split_features)

        sub_labels  = ms.labels_
        n_trees     = len(np.unique(sub_labels))

        if n_trees < 2:
            # Mean Shift found only one mode — still just one tree
            final_clusters.append(pcd)
            continue

        # this cluster is a valid split candidate — record it
        pre_split_clusters.append(pcd)
        pre_split_names.append(
            f"radius{radius:.2f}m_split_into_{n_trees}"
            f"{'_color' if cluster_use_color else ''}"
        )

        print(f"Mean Shift split cluster (radius={radius:.2f}m, {mode_str}) "
              f"into {n_trees} sub-clusters")

        # ── validate each sub-cluster before accepting the split ───────────────
        valid_subs = []
        for i in np.unique(sub_labels):
            mask    = sub_labels == i
            sub_pcd = pcd.select_by_index(np.where(mask)[0])
            if (len(sub_pcd.points) >= min_points and
                    filter_cluster(sub_pcd, min_height=1.0, min_radius=0.3)):
                valid_subs.append(sub_pcd)

        # keep whatever valid sub-clusters came out, even if only one survives
        if len(valid_subs) >= 1:
            discarded = n_trees - len(valid_subs)
            if discarded > 0:
                print(f"  Discarded {discarded} sub-clusters that failed validation")
            final_clusters.extend(valid_subs)
        else:
            # nothing survived validation at all — keep the original
            print(f"  Split rejected — no sub-clusters passed validation, keeping original")
            final_clusters.append(pcd)

    # save pre-split clusters locally if a path was given
    if save_pre_split_path is not None:
        _save_clusters_descriptive(pre_split_clusters, pre_split_names, save_pre_split_path)
        print(f"Saved {len(pre_split_clusters)} pre-split clusters to {save_pre_split_path}")

    if use_color and n_color_fallback > 0:
        print(f"⚠  {n_color_fallback} large cluster(s) had no colour data — "
              f"fell back to XY-only splitting for those despite use_color=True")

    if force_check_indices:
        print(f"force_check_indices: {n_force_checked}/{len(force_check_indices)} "
              f"forced cluster(s) were actually below max_radius and used the override "
              f"(the rest were already above max_radius and would have been checked anyway)")

    print(f"Clusters before splitting: {len(clusters)}")
    print(f"Clusters after splitting:  {len(final_clusters)}")
    return final_clusters


def find_density_peaks(points, colors=None, k=30, min_peak_distance=3.0,
                       min_density_ratio=2.5, use_color=False,
                       color_weight=10.0,
                       color_channels=("chroma_g", "chroma_r", "chroma_b")):
    """
    Find local density peaks in a point cloud using KDTree-based density
    estimation, optionally in a colour-augmented feature space.

    A point qualifies as a peak if its local density (inverse mean distance
    to k neighbours) is higher than all its k neighbours *and* exceeds the
    cluster-wide mean density by min_density_ratio. Peaks closer than
    min_peak_distance to an already-accepted peak are suppressed
    (greedy nearest-first).

    With use_color=False (default), density is computed purely on XYZ —
    identical to the original function. With use_color=True, density is
    computed on XY + weighted chromaticity instead, so two spatially
    overlapping but differently-coloured crowns can register as separate
    density peaks even when their XY density alone looks like one blob.
    Peak-separation suppression (min_peak_distance) still uses the SAME
    augmented distance the density was computed on, for consistency with
    what "peak" meant when it was found — if you need suppression judged
    on pure XY distance regardless of colour, do that filtering as a
    separate post-processing step on the returned peak indices.

    Called by split_large_clusters() to gate splitting — only clusters
    with two or more surviving peaks are sent to Mean Shift.

    Args:
        points (np.ndarray): (N, 3) XYZ array of cluster points.
        colors (np.ndarray | None): (N, 3) normalised RGB, required when
            use_color=True. Ignored when use_color=False. Default None.
        k (int): Number of neighbours for local density estimation. Capped
            internally at len(points) - 1. Default 30.
        min_peak_distance (float): Minimum separation between two accepted
            peaks, in the same units/space density was computed in
            (metres if use_color=False; augmented XY+colour distance if
            use_color=True). Default 3.0.
        min_density_ratio (float): A candidate peak must be at least this
            many times denser than the cluster mean density. Default 2.5.
        use_color (bool): Compute density on XY + weighted chromaticity
            instead of raw XYZ. Requires `colors`. Default False.
        color_weight (float): "Metres-equivalent" scale factor for
            chromaticity — see module docstring. Only used when
            use_color=True. Default 10.0.
        color_channels (tuple[str]): Which chromaticity channels to
            include when use_color=True. Default all three.

    Returns:
        filtered_peaks (list of int): Indices into points of accepted peaks.
        density (np.ndarray): (N,) per-point density values, in whichever
            space (XYZ or augmented) was actually used.

    Requirements:
        numpy, scipy.spatial.KDTree
    """

    # cap k to avoid index errors on small clusters
    k = min(k, len(points) - 1)

    if use_color:
        if colors is None:
            raise ValueError(
                "find_density_peaks: use_color=True but no colors were "
                "given. Pass colors=np.asarray(pcd.colors), or fall back "
                "to use_color=False for this cluster."
            )
        chroma = _per_point_chroma(colors, channels=color_channels)
        feature_space = np.column_stack([points[:, :2], chroma * color_weight])
    else:
        feature_space = points

    tree = KDTree(feature_space)
    distances, neighbor_indices = tree.query(feature_space, k=k)
    density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)

    mean_density = density.mean()

    # a point is a peak if it has higher density than all its k neighbors
    # AND is meaningfully denser than the cluster average
    peaks = []
    for i, neighbors in enumerate(neighbor_indices):
        if (density[i] == density[neighbors].max() and
                density[i] > mean_density * min_density_ratio):
            peaks.append(i)

    # filter out peaks that are too close together — likely the same tree.
    # distance here is computed in the SAME feature_space density was
    # computed on (see docstring note above).
    filtered_peaks = []
    for i in peaks:
        too_close = any(
            np.linalg.norm(feature_space[i] - feature_space[j]) < min_peak_distance
            for j in filtered_peaks
        )
        if not too_close:
            filtered_peaks.append(i)

    return filtered_peaks, density