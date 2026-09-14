"""
diagnose_cluster_bimodality.py
────────────────────────────────────────────────────────────────────────────────
Cheap pre-check for whether misclassified / flagged clusters are actually two
overlapping trees whose crowns interleave in XY (so Mean Shift on position
alone can't see them) but separate cleanly in colour.

Why this exists
────────────────
split_large_clusters.py's find_density_peaks() + MeanShift only ever look at
(x, y). That correctly splits two trees with a visible gap between trunks,
but is blind to two crowns that occupy the *same* XY footprint at different
heights or in a mixed canopy (e.g. juniper growing into a pinyon's crown) —
exactly the failure mode misclassification analysis has been pointing at:
cluster 1370's abnormal n_density_peaks (~24) and the three stable
low-crown_base_ratio errors both look like "this is probably two trees,"
without necessarily showing two separated XY density peaks.

Before rebuilding split_large_clusters.py to cluster in a combined
(XY + colour) feature space, this script answers a cheaper question first:
for the specific clusters already flagged as suspicious, is there evidence
of real bimodality in colour (or height) that XY-only splitting would have
missed? If most flagged clusters turn out to be spatially bimodal too (i.e.
Mean Shift should already be catching them, and something else is going
wrong), colour-augmented splitting isn't the fix to reach for.

What it does, per cluster
──────────────────────────
1. Fits a 1-component and a 2-component GaussianMixture to each requested
   per-point feature (colour chromaticity channels by default, height
   optionally) and compares BIC. A large BIC improvement for 2 components
   is evidence of real bimodality, not just noise — unlike a plain
   two-hump-looking histogram, BIC penalises the extra parameters, so a
   marginal improvement is discounted.
2. For color features, also reports the mean separation between the two
   fitted components in pooled standard-deviation units — a large delta-BIC
   with near-identical component means isn't meaningfully separable, so
   this catches that case where a t-test on the two GMM components alone
   would not.
3. For color features that ARE statistically bimodal, checks whether the
   two colour-assigned point groups also occupy different XY sub-regions
   of the cluster's footprint (see _color_groups_spatially_separated()).
   This is the critical gate: a single healthy tree very commonly has
   genuinely bimodal chromaticity from illumination alone — sunlit crown
   top vs. shaded interior, exterior vs. interior foliage — and that
   variation is spatially interspersed across the whole footprint, not
   segregated into two areas. Two merged trees, by contrast, occupy
   different ground positions. Without this check, ordinary within-crown
   colour variation dominates the flagged list at full-dataset scale.
4. Computes a plain spatial (XY) density peak count using the same
   density-peak logic as split_large_clusters.find_density_peaks(), so you
   can directly compare "does this look like 2 trees in color-space" against
   "does this look like 2 trees in XY-density-space" for the same cluster.
5. Flags clusters where colour says "two trees" (bimodal AND spatially
   separated) but XY density says "one tree" — the specific pattern that
   would justify a colour-augmented splitting rebuild.

Usage
─────
    from diagnose_cluster_bimodality import diagnose_cluster_bimodality

    # cluster_indices: whatever you already have flagged — e.g. from
    # inspect_misclassified_clusters()'s "file" column, or clusters with
    # unusually high n_density_peaks in df_deep_clusters
    df_diag = diagnose_cluster_bimodality(
        clusters,                      # full list from load_clusters()
        cluster_indices=[1370, 204, 88, 141],
        save_path=PATHS['Images'] + 'bimodality_diagnostics/',
    )
    print(df_diag)

Or pull candidate indices automatically from what you already have on disk:

    from diagnose_cluster_bimodality import get_flagged_indices_from_sources

    indices = get_flagged_indices_from_sources(
        df_deep_clusters,
        misclassification_csv=PATHS['Dataframes'] + 'pinyon_misclassification_report.csv',
        n_density_peaks_top_n=10,
    )

Requirements
────────────
    numpy, pandas, matplotlib, scikit-learn (GaussianMixture, already in venv)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.spatial import KDTree


# ── per-point feature helpers ─────────────────────────────────────────────────

def _per_point_chroma(colors):
    """
    Shadow-robust per-point chromaticity — same normalisation used in
    get_deep_cluster_features.py (divide each point's RGB by its own
    brightness before doing anything else), so a shaded half of a crown
    doesn't masquerade as a colour difference between two trees.

    Args:
        colors (np.ndarray): (N, 3) normalised RGB, 0.0-1.0.

    Returns:
        dict: "chroma_r", "chroma_g", "chroma_b" -> (N,) arrays.

    Requirements:
        numpy
    """
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    brightness = r + g + b + 1e-6
    return {
        "chroma_r": r / brightness,
        "chroma_g": g / brightness,
        "chroma_b": b / brightness,
    }


def _fit_gmm_bimodality(values, random_state=42, max_fit_points=800):
    """
    Fit 1- and 2-component 1D Gaussian mixtures to a per-point feature and
    compare BIC, plus report how separated the two components are.

    max_fit_points caps how many points the GMM actually fits on — 800
    random points is plenty to detect real bimodality (a mixture's shape is
    determined by its parameters, not by feeding it every last point), and
    this is what makes running the diagnostic over every cluster in a trial
    (~3,600) tractable instead of fitting on clusters with several thousand
    points each.

    BIC (not raw log-likelihood) is used for the 1-vs-2 comparison because
    it penalises the extra parameters a 2-component fit always has more
    freedom to exploit — a 2-component fit will always fit training data at
    least as well as 1-component, so raw likelihood alone would call every
    cluster "bimodal." A meaningfully lower BIC for 2 components is a much
    stronger signal.

    separation_in_std is reported alongside delta_bic because a 2-component
    fit can have a large BIC improvement while the two components still sit
    almost on top of each other (e.g. one wide vs one narrow component
    modelling the same underlying single population) — that's a
    "meh, statistically detectable but not practically separable" case,
    which delta_bic alone won't tell you.

    Args:
        values (np.ndarray): (N,) per-point feature values. NaNs dropped.
        random_state (int): GMM init seed. Default 42.

    Returns:
        dict: {
            "bic_1": float, "bic_2": float, "delta_bic": float
                (bic_1 - bic_2; positive means 2-component fit is better),
            "weights_2": (2,) array, mixing weights of the 2-component fit,
            "means_2": (2,) array, sorted ascending,
            "stds_2": (2,) array, matched order to means_2,
            "separation_in_std": float, |mean_a - mean_b| / pooled_std,
        }

    Requirements:
        numpy, scikit-learn
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) < 20:
        # too few points for a stable 2-component fit — report as
        # "insufficient data" rather than a misleadingly confident number
        return {
            "bic_1": np.nan, "bic_2": np.nan, "delta_bic": np.nan,
            "weights_2": (np.nan, np.nan), "means_2": (np.nan, np.nan),
            "stds_2": (np.nan, np.nan), "separation_in_std": np.nan,
            "gmm2": None, "component_order": None,
        }

    if len(values) > max_fit_points:
        rng = np.random.default_rng(random_state)
        values = rng.choice(values, size=max_fit_points, replace=False)

    X = values.reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(X)
    gmm2 = GaussianMixture(n_components=2, random_state=random_state).fit(X)

    bic1 = gmm1.bic(X)
    bic2 = gmm2.bic(X)

    # sort the 2-component fit's parameters by mean so output is stable
    # across runs and comparable across clusters
    order = np.argsort(gmm2.means_.ravel())
    means_2 = gmm2.means_.ravel()[order]
    stds_2  = np.sqrt(gmm2.covariances_.ravel())[order]
    weights_2 = gmm2.weights_[order]

    pooled_std = np.sqrt((stds_2[0] ** 2 + stds_2[1] ** 2) / 2) + 1e-6
    separation = float(abs(means_2[1] - means_2[0]) / pooled_std)

    return {
        "bic_1": float(bic1),
        "bic_2": float(bic2),
        "delta_bic": float(bic1 - bic2),
        "weights_2": tuple(weights_2),
        "means_2": tuple(means_2),
        "stds_2": tuple(stds_2),
        "separation_in_std": separation,
        "gmm2": gmm2,
        "component_order": order,   # to remap gmm2.predict()'s raw labels to sorted-by-mean order
    }


def _spatial_density_peak_count(xy, k=15, min_peak_distance=1.0, min_density_ratio=1.5):
    """
    Self-contained copy of the density-peak logic from
    split_large_clusters.find_density_peaks() / get_deep_cluster_features's
    _count_density_peaks(), reimplemented here (rather than imported) to
    avoid pulling in the clustering package's sys.path hack and dependency
    chain for a single diagnostic function. Same core algorithm: a point is
    a peak if its local density beats all its k neighbours and clears
    min_density_ratio × the cluster's mean density; peaks closer than
    min_peak_distance to an already-accepted peak are suppressed.

    Args:
        xy (np.ndarray): (N, 2) XY point array.
        k (int): Neighbour count for local density. Capped at len(xy)-1.
            Default 15.
        min_peak_distance (float): Minimum separation (metres) between
            accepted peaks. Default 1.0.
        min_density_ratio (float): Candidate peak must be this many times
            denser than the cluster mean. Default 1.5.

    Returns:
        int: Number of surviving spatial density peaks. >= 1 skew towards
            "single tree"; 2+ means Mean Shift's own gate would already
            have flagged this cluster as a split candidate.

    Requirements:
        numpy, scipy.spatial.KDTree
    """
    n = len(xy)
    if n < 5:
        return 1

    k = min(k, n - 1)
    tree = KDTree(xy)
    distances, neighbor_indices = tree.query(xy, k=k)
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
            np.linalg.norm(xy[i] - xy[j]) < min_peak_distance
            for j in filtered_peaks
        )
        if not too_close:
            filtered_peaks.append(i)

    return max(1, len(filtered_peaks))


def _color_groups_spatially_separated(xy, gmm2, component_order, feature_values,
                                       min_ratio=0.75, min_absolute_m=0.3,
                                       min_group_size=15):
    """
    Check whether a statistically-bimodal color feature also corresponds to
    two spatially distinct regions of the cluster's footprint, as opposed
    to two color populations that are scattered throughout the same
    footprint (the signature of ordinary within-crown variation — sunlit
    vs. shaded foliage, exterior vs. interior branches — rather than two
    overlapping trees).

    This is the gate that was missing from the original bimodality-only
    check and is the main source of false positives at full-dataset scale:
    a single healthy tree very commonly has genuinely bimodal chromaticity
    from illumination alone (chroma normalisation corrects per-point
    brightness scaling, but not illumination-colour shifts between sunlit
    and shadowed portions of one canopy). Requiring the two color-assigned
    point groups to also occupy different XY sub-regions screens that case
    out, because sun/shade variation is spatially interspersed across the
    same footprint, not segregated into two sub-areas — while two merged
    trees genuinely do occupy different ground positions.

    Every point in the cluster (not just the subsample the GMM was fit on)
    is hard-assigned to whichever component has higher posterior
    probability, via gmm2.predict() on the full feature_values array. The
    XY centroids of the two resulting point groups are compared against
    the pooled within-group spatial scatter — analogous to a spatial
    Cohen's d — so a large intercentroid distance relative to how tightly
    each group's own points cluster in XY counts as separated, while two
    groups whose points are both scattered evenly across the same area do
    not, even if their centroids differ slightly by chance.

    Args:
        xy (np.ndarray): (N, 2) XY coordinates for every point in the
            cluster, same row order as feature_values.
        gmm2 (sklearn.mixture.GaussianMixture): The fitted 2-component
            model from _fit_gmm_bimodality()["gmm2"].
        component_order (np.ndarray): Index array from
            _fit_gmm_bimodality()["component_order"] — maps gmm2's raw
            component indices to sorted-by-mean order, so labels are
            consistent with means_2/stds_2/weights_2 from the same fit.
        feature_values (np.ndarray): (N,) the SAME per-point feature
            values gmm2 was fit on (e.g. full-cluster chroma_g, not the
            subsample) — used to assign every point to a component.
        min_ratio (float): Minimum ratio of intercentroid distance to
            pooled within-group spatial scatter to count as separated.
            1.0 means the groups are, on average, as far apart as they are
            internally spread out. 0.75 is a slightly more permissive
            starting point — tune against known merged-tree clusters (e.g.
            1370) vs. known single-tree false positives. Default 0.75.
        min_absolute_m (float): Minimum intercentroid distance in metres,
            regardless of ratio — guards against small/tight clusters
            where a high ratio could come from a trivially small absolute
            separation. Default 0.3.
        min_group_size (int): Minimum points required in EACH colour-
            assigned group for the spatial check to run at all. Below this,
            the intercentroid-distance-to-scatter ratio is unstable — a
            handful of points can look "spatially separated" from pure
            chance arrangement, and few points also mechanically shrink
            the apparent within-group scatter, inflating the ratio without
            a real spatial pattern behind it. This is the main suspected
            driver of over-flagging on small/sparse clusters near
            min_points — raise this (e.g. to 25-30) if small clusters are
            still being flagged after upgrading from the default 5.
            Default 15.

    Returns:
        dict: {
            "spatially_separated": bool,
            "intercentroid_distance_m": float,
            "separation_ratio": float,
            "group_sizes": (int, int),
        }

    Requirements:
        numpy, scikit-learn
    """
    if gmm2 is None:
        return {"spatially_separated": False, "intercentroid_distance_m": np.nan,
                "separation_ratio": np.nan, "group_sizes": (0, 0)}

    valid = ~np.isnan(feature_values)
    if valid.sum() < 20:
        return {"spatially_separated": False, "intercentroid_distance_m": np.nan,
                "separation_ratio": np.nan, "group_sizes": (0, 0)}

    raw_labels = gmm2.predict(feature_values[valid].reshape(-1, 1))
    # remap raw component index -> sorted-by-mean order used everywhere else
    remap = {orig: new for new, orig in enumerate(component_order)}
    labels = np.array([remap[l] for l in raw_labels])

    xy_valid = xy[valid]
    group_a = xy_valid[labels == 0]
    group_b = xy_valid[labels == 1]

    if len(group_a) < min_group_size or len(group_b) < min_group_size:
        # one component doesn't have enough points for the spatial ratio
        # to be statistically stable — not a meaningful spatial split
        return {"spatially_separated": False, "intercentroid_distance_m": np.nan,
                "separation_ratio": np.nan,
                "group_sizes": (len(group_a), len(group_b))}

    centroid_a = group_a.mean(axis=0)
    centroid_b = group_b.mean(axis=0)
    intercentroid_distance = float(np.linalg.norm(centroid_a - centroid_b))

    # pooled within-group spatial scatter — mean distance from each point
    # to its own group's centroid, averaged across both groups
    scatter_a = np.linalg.norm(group_a - centroid_a, axis=1).mean()
    scatter_b = np.linalg.norm(group_b - centroid_b, axis=1).mean()
    pooled_scatter = (scatter_a + scatter_b) / 2 + 1e-6

    ratio = intercentroid_distance / pooled_scatter

    spatially_separated = (
        ratio >= min_ratio and intercentroid_distance >= min_absolute_m
    )

    return {
        "spatially_separated": spatially_separated,
        "intercentroid_distance_m": intercentroid_distance,
        "separation_ratio": float(ratio),
        "group_sizes": (len(group_a), len(group_b)),
    }


# ── main entry point ────────────────────────────────────────────────────────

def diagnose_cluster_bimodality(clusters, cluster_indices,
                                 color_features=("chroma_g", "chroma_r", "chroma_b"),
                                 check_height=True,
                                 delta_bic_threshold=10.0,
                                 separation_threshold=1.0,
                                 spatial_separation_ratio=0.75,
                                 min_intercentroid_m=0.3,
                                 min_spatial_group_size=15,
                                 max_fit_points=800,
                                 save_path=None,
                                 plot_only_flagged=True,
                                 verbose=True,
                                 progress_every=200):
    """
    Run the color/height bimodality check + spatial peak count on a list of
    clusters, and flag the "color says two trees, XY says one tree" pattern
    that would justify color-augmented splitting.

    Scales to the full cluster list (~3,600 at Sunset Crater), not just a
    hand-picked subset — pass cluster_indices=range(len(clusters)) to run
    over everything. At that scale expect the bulk of runtime to be
    load_clusters() reading ~3,600 .ply files off disk, not the GMM fitting
    itself (max_fit_points keeps each fit cheap regardless of cluster size).
    Plotting is automatically restricted to flagged clusters only
    (see plot_only_flagged) so a full run doesn't dump thousands of PNGs.

    Args:
        clusters (list of o3d.geometry.PointCloud): Full cluster list, same
            list/order as load_clusters(PATHS['Clusters']) — indices in
            cluster_indices are positions into this list.
        cluster_indices (list[int] | range): Which clusters to diagnose.
            Pass a small targeted list, or range(len(clusters)) to scan
            every cluster in the trial for merged-tree candidates.
        color_features (tuple[str]): Which chromaticity channels to test
            for bimodality. Default ("chroma_g", "chroma_r", "chroma_b").
        check_height (bool): Also test per-point Z for bimodality — catches
            the "one tree growing into/under another" case that colour
            alone might miss if both species have similar foliage colour
            but occupy different height bands. Default True.
        delta_bic_threshold (float): Minimum BIC improvement (1-component
            BIC minus 2-component BIC) to call a feature "bimodal." 10 is a
            commonly used "very strong evidence" cutoff for BIC differences
            (Kass & Raftery-style rule of thumb) — deliberately conservative
            since a false "yes, split this" recommendation is more costly
            here than a false "no." Default 10.0.
        separation_threshold (float): Minimum separation_in_std for a
            bimodal fit to also be called practically separable, not just
            statistically detectable. Default 1.0 (components at least 1
            pooled std apart).
        spatial_separation_ratio (float): Minimum ratio of intercentroid
            XY distance to pooled within-group spatial scatter for a
            statistically-bimodal color feature to also count as spatially
            coherent — see _color_groups_spatially_separated(). This is
            the main lever against false positives from ordinary
            within-crown color variation (sun/shade, exterior/interior
            foliage): those show up as color-bimodal but spatially
            scattered across the same footprint, and won't clear this
            ratio. Raise toward 1.0-1.5 if still over-flagging; lower
            toward 0.5 if it's missing known merged-tree clusters (e.g.
            1370). Default 0.75.
        min_intercentroid_m (float): Minimum XY centroid separation in
            metres, regardless of ratio — guards against small/tight
            clusters producing a high ratio from a trivial absolute
            distance. Default 0.3.
        min_spatial_group_size (int): Minimum points required in each
            colour-assigned group before the spatial-separation check is
            trusted — see _color_groups_spatially_separated(). Raise this
            if small/sparse clusters keep getting flagged. Default 15.
        max_fit_points (int): Cap on points passed to each GMM fit — see
            _fit_gmm_bimodality(). Lower this (e.g. 400) if a full-dataset
            run is too slow; raise it if separation_in_std results look
            unstable between runs. Default 800.
        save_path (str | None): Directory to save diagnostic plots into.
            Pass None to skip plotting entirely. Default None.
        plot_only_flagged (bool): When save_path is set, only save plots
            for clusters where color_bimodal_spatial_unimodal is True,
            rather than one PNG per cluster checked. Ignored if save_path
            is None. Set False to plot everything (only reasonable for
            small, targeted runs). Default True.
        verbose (bool): Print progress and results. At more than 20
            clusters, per-cluster detail is suppressed in favour of a
            progress counter + immediate print whenever a cluster is
            flagged, so a full-dataset run doesn't scroll past thousands of
            lines of output. Default True.
        progress_every (int): Print a progress line every N clusters when
            running over a large list. Default 200.

    Returns:
        pd.DataFrame: One row per (cluster, feature) tested, with columns:
            "file", "feature", "delta_bic", "separation_in_std",
            "bimodal" (bool, both thresholds cleared),
            "n_spatial_peaks" (same value repeated per cluster, for
            convenience), "color_bimodal_spatial_unimodal" (bool) — the
            specific pattern this diagnostic exists to catch. Filter this
            DataFrame down to the flagged rows for the actual candidate
            list — see the summary block printed at the end for a
            ready-made list.

    Requirements:
        numpy, pandas, matplotlib, scikit-learn, scipy.spatial.KDTree
    """
    cluster_indices = list(cluster_indices)
    n_total = len(cluster_indices)
    detailed_print = verbose and n_total <= 20

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    features_to_test = list(color_features)
    if check_height:
        features_to_test.append("height_z")

    rows = []

    for progress_i, idx in enumerate(cluster_indices, start=1):
        if verbose and not detailed_print and progress_i % progress_every == 0:
            n_flagged_so_far = len({r["file"] for r in rows
                                     if r.get("color_bimodal_spatial_unimodal")})
            print(f"  ...{progress_i}/{n_total} clusters checked "
                  f"({n_flagged_so_far} flagged so far)")
        if idx >= len(clusters):
            print(f"  ⚠  cluster index {idx} out of range ({len(clusters)} "
                  f"clusters total) — skipping")
            continue

        pcd = clusters[idx]
        points = np.asarray(pcd.points)
        xy = points[:, :2]
        z = points[:, 2]

        n_spatial_peaks = _spatial_density_peak_count(xy)

        if not pcd.has_colors():
            print(f"  ⚠  cluster {idx} has no colour data — skipping colour "
                  f"features")
            per_point = {}
        else:
            colors = np.asarray(pcd.colors)
            per_point = _per_point_chroma(colors)
        per_point["height_z"] = z

        cluster_results = {}
        for feat in features_to_test:
            if feat not in per_point:
                continue
            fit = _fit_gmm_bimodality(per_point[feat], max_fit_points=max_fit_points)
            bimodal = (
                not np.isnan(fit["delta_bic"]) and
                fit["delta_bic"] >= delta_bic_threshold and
                fit["separation_in_std"] >= separation_threshold
            )

            # spatial-correlation check — only meaningful (and only computed)
            # for color features that are already statistically bimodal;
            # this is the gate that screens out ordinary within-crown colour
            # variation (sun/shade, exterior/interior) from genuine merged
            # trees. See _color_groups_spatially_separated() docstring.
            spatial_check = {
                "spatially_separated": False,
                "intercentroid_distance_m": np.nan,
                "separation_ratio": np.nan,
            }
            if bimodal and feat in color_features:
                spatial_check = _color_groups_spatially_separated(
                    xy, fit["gmm2"], fit["component_order"], per_point[feat],
                    min_ratio=spatial_separation_ratio,
                    min_absolute_m=min_intercentroid_m,
                    min_group_size=min_spatial_group_size,
                )

            cluster_results[feat] = {**fit, "bimodal": bimodal, **spatial_check}

            rows.append({
                "file": idx,
                "feature": feat,
                "n_points": len(points),
                "delta_bic": fit["delta_bic"],
                "separation_in_std": fit["separation_in_std"],
                "bimodal": bimodal,
                "spatially_separated": spatial_check["spatially_separated"],
                "intercentroid_distance_m": spatial_check["intercentroid_distance_m"],
                "separation_ratio": spatial_check["separation_ratio"],
                "n_spatial_peaks": n_spatial_peaks,
            })

        # a feature only counts toward "this cluster is probably two trees"
        # if it's BOTH statistically bimodal AND the two colour populations
        # occupy different parts of the footprint — bimodal alone is very
        # commonly just sun/shade structure within one crown (see module
        # docstring). height_z is excluded here since spatial correlation
        # wasn't computed for it (a vertical light gradient inside one
        # crown would trivially "spatially separate" by height without
        # meaning anything about XY position).
        color_bimodal = any(
            cluster_results.get(f, {}).get("bimodal", False) and
            cluster_results.get(f, {}).get("spatially_separated", False)
            for f in color_features
        )
        spatially_unimodal = n_spatial_peaks < 2
        color_bimodal_spatial_unimodal = color_bimodal and spatially_unimodal

        if detailed_print:
            print(f"\n  cluster {idx}  (n_points={len(points)}, "
                  f"n_spatial_peaks={n_spatial_peaks})")
            for feat, res in cluster_results.items():
                flag = " ← BIMODAL" if res["bimodal"] else ""
                sep_note = ""
                if feat in color_features and res["bimodal"]:
                    sep_note = (
                        f"  [xy_sep={res['intercentroid_distance_m']:.2f}m, "
                        f"ratio={res['separation_ratio']:.2f}"
                        f"{', SPATIALLY SEPARATED' if res['spatially_separated'] else ', same footprint — likely sun/shade'}]"
                    )
                print(f"    {feat:<12} delta_bic={res['delta_bic']:>8.1f}  "
                      f"separation={res['separation_in_std']:.2f}σ{flag}{sep_note}")
            if color_bimodal_spatial_unimodal:
                print(f"    ⚠  COLOUR-BIMODAL / SPATIALLY-SEPARATED / "
                      f"XY-DENSITY-UNIMODAL — this is the pattern "
                      f"colour-augmented splitting would fix")
        elif verbose and color_bimodal_spatial_unimodal:
            # in batch mode, still surface flagged clusters immediately
            # rather than only in the final summary — useful for spot-
            # checking .ply files while a long run is still going
            best_feat = max(
                (f for f in color_features if f in cluster_results),
                key=lambda f: cluster_results[f]["delta_bic"],
                default=None,
            )
            print(f"  ⚠  cluster {idx} flagged (n_points={len(points)}, "
                  f"n_spatial_peaks={n_spatial_peaks}"
                  + (f", strongest feature={best_feat}" if best_feat else "")
                  + ")")

        # tag every row for this cluster with the combined flag
        for r in rows[-len(cluster_results):]:
            r["color_bimodal_spatial_unimodal"] = color_bimodal_spatial_unimodal

        if save_path is not None and (color_bimodal_spatial_unimodal or not plot_only_flagged):
            _plot_cluster_diagnostics(
                idx, per_point, cluster_results, n_spatial_peaks, save_path
            )

    df = pd.DataFrame(rows)

    if verbose and len(df) > 0:
        flagged = df.loc[df["color_bimodal_spatial_unimodal"], "file"].unique()
        print(f"\n{'═' * 60}")
        print(f"  SUMMARY  ({len(cluster_indices)} clusters checked)")
        print(f"{'═' * 60}")
        print(f"  Colour-bimodal / spatially-unimodal (candidates for "
              f"colour-augmented splitting): {len(flagged)}")
        if len(flagged):
            print(f"    {sorted(flagged.tolist())}")
        spatial_and_color = df.loc[
            df["bimodal"] & (df["n_spatial_peaks"] >= 2), "file"
        ].unique()
        print(f"  Both colour- and spatially-bimodal (Mean Shift should "
              f"already be catching these — investigate why it isn't, "
              f"e.g. min_peak_distance/min_density_ratio too strict): "
              f"{len(spatial_and_color)}")
        if len(spatial_and_color):
            print(f"    {sorted(spatial_and_color.tolist())}")
        print()

    return df


def _plot_cluster_diagnostics(idx, per_point, cluster_results,
                               n_spatial_peaks, save_path):
    """
    Save one PNG per cluster: histogram + 1- and 2-component GMM curves
    overlaid, one subplot per tested feature.
    """
    n_feats = len(cluster_results)
    if n_feats == 0:
        return

    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4))
    axes = np.atleast_1d(axes)

    for ax, (feat, res) in zip(axes, cluster_results.items()):
        values = np.asarray(per_point[feat])
        values = values[~np.isnan(values)]
        ax.hist(values, bins=30, density=True, alpha=0.5, color="#888888")

        if not np.isnan(res["separation_in_std"]):
            x_range = np.linspace(values.min(), values.max(), 300)
            for w, m, s in zip(res["weights_2"], res["means_2"], res["stds_2"]):
                y = w * (1.0 / (s * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((x_range - m) / s) ** 2)
                ax.plot(x_range, y, linewidth=2)

        flag = " [BIMODAL]" if res["bimodal"] else ""
        sep_note = ""
        if "spatially_separated" in res and res["bimodal"] and \
                not np.isnan(res.get("intercentroid_distance_m", np.nan)):
            sep_note = (
                f"\nxy_sep={res['intercentroid_distance_m']:.2f}m "
                f"ratio={res['separation_ratio']:.2f}"
                f"{' [SPATIAL]' if res['spatially_separated'] else ' [same footprint]'}"
            )
        ax.set_title(f"{feat}{flag}\nΔBIC={res['delta_bic']:.1f}  "
                     f"sep={res['separation_in_std']:.2f}σ{sep_note}", fontsize=9)
        ax.set_xlabel(feat, fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"Cluster {idx} — bimodality diagnostics "
                 f"(n_spatial_peaks={n_spatial_peaks})", fontsize=11)
    plt.tight_layout()

    out = os.path.join(save_path, f"bimodality_cluster{idx}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def diagnose_all_clusters(clusters, save_path=None, **kwargs):
    """
    Thin convenience wrapper around diagnose_cluster_bimodality() that runs
    over every cluster in the list — for scanning a whole trial (~3,600
    clusters at Sunset Crater) for merged-tree candidates, not just
    clusters already flagged by misclassification or n_density_peaks.

    Since this checks clusters with no prior reason to be suspicious, false
    positives are expected to be more common than in a targeted run — treat
    the flagged list as candidates for visual (.ply) spot-checking, not as
    confirmed merged-tree clusters. Consider raising delta_bic_threshold
    (e.g. to 20-30) for a full-dataset run if the flagged count comes back
    implausibly high relative to the ~4 currently-known misclassifications.

    Args:
        clusters (list of o3d.geometry.PointCloud): Full cluster list from
            load_clusters(PATHS['Clusters']).
        save_path (str | None): Passed through to diagnose_cluster_bimodality();
            plots are still restricted to flagged clusters by default.
        **kwargs: Any other diagnose_cluster_bimodality() argument
            (delta_bic_threshold, separation_threshold, max_fit_points,
            progress_every, etc.).

    Returns:
        pd.DataFrame: Same shape as diagnose_cluster_bimodality().

    Requirements:
        numpy, pandas, matplotlib, scikit-learn, scipy.spatial.KDTree
    """
    print(f"Running bimodality diagnostic on all {len(clusters)} clusters — "
          f"this checks every cluster in the trial, expect more candidates "
          f"than a targeted run and treat flags as leads for visual "
          f"inspection, not confirmed merges.\n")
    return diagnose_cluster_bimodality(
        clusters, range(len(clusters)), save_path=save_path, **kwargs
    )


# ── debug: why did a specific known cluster not get flagged? ─────────────────

def explain_why_not_flagged(clusters, cluster_indices,
                             color_features=("chroma_g", "chroma_r", "chroma_b"),
                             delta_bic_threshold=10.0,
                             separation_threshold=1.0,
                             spatial_separation_ratio=0.75,
                             min_intercentroid_m=0.3,
                             min_spatial_group_size=15,
                             max_fit_points=800,
                             max_radius=None):
    """
    Walk through every gate the combined "color_bimodal_spatial_unimodal"
    flag depends on, for a specific list of clusters, and print exactly
    which gate blocked each one — instead of just the final pass/fail.

    Built for the situation where a known-problem cluster list (e.g. from
    inspect_misclassified_clusters()) still comes back with 0 flags after
    recalibrating thresholds. Continuing to nudge delta_bic_threshold /
    spatial_separation_ratio blind at that point conflates three genuinely
    different failure modes that need different responses:

        1. No colour data on the cluster at all — the color path can never
           fire, regardless of thresholds. Check the .ply actually has RGB.
        2. Statistically not bimodal — no GMM feature clears delta_bic /
           separation_in_std. Either this cluster's colour signal really is
           weak (check plot_feature_separability.py — if chroma features
           don't separate species well in general at Sunset Crater, a
           per-point mixture test won't magically find it either), or a
           merge is real but one species contributes so few points that
           even 800-point subsampling can't surface it statistically.
        3. Bimodal but fails the spatial-separation gate — split further
           into "groups too small to trust" (min_spatial_group_size cutting
           off a minority-species component, e.g. a small juniper sprig
           under a much larger pinyon crown) vs. "genuinely not spatially
           separated" (this cluster's colour bimodality is probably
           ordinary within-crown variation, not two trees).
        4. Bimodal AND spatially separated, but n_spatial_peaks >= 2 — this
           diagnostic's combined flag specifically targets "colour says
           two, XY density says one." If XY density ALSO already finds 2+
           peaks, colour-augmented splitting isn't the missing piece —
           something in the existing spatial split (Mean Shift not
           converging to 2 modes at your min_peak_distance, or a found
           split getting rejected by filter_cluster validation) is.

    Args:
        clusters (list of o3d.geometry.PointCloud): Full cluster list.
        cluster_indices (list[int]): The specific clusters to explain —
            normally your known-problem list.
        color_features, delta_bic_threshold, separation_threshold,
        spatial_separation_ratio, min_intercentroid_m,
        min_spatial_group_size, max_fit_points: Same meaning and defaults
            as diagnose_cluster_bimodality() — pass the SAME values you're
            actually using so this reflects your real run.
        max_radius (float | None): If given, also reports whether each
            cluster's radius exceeds this — informational only, but tells
            you whether the PRODUCTION split_large_clusters() would even
            have attempted a spatial split on this cluster in the first
            place (its own gate 1 skips clusters at or below max_radius
            entirely, unlike this diagnostic which checks every cluster
            regardless of size). Pass the same max_radius your pinyons.sh
            uses (4.25 as of the last known-good sweep). Default None
            (skip this check).

    Returns:
        None — prints a per-cluster breakdown. Nothing to compare
        programmatically here; this is meant to be read, not consumed.

    Requirements:
        numpy, scikit-learn, scipy.spatial.KDTree
    """
    for idx in cluster_indices:
        if idx >= len(clusters):
            print(f"\n  cluster {idx}: out of range, skipping")
            continue

        pcd = clusters[idx]
        points = np.asarray(pcd.points)
        xy = points[:, :2]

        aabb = pcd.get_axis_aligned_bounding_box()
        bounds = aabb.max_bound - aabb.min_bound
        radius = max(bounds[:2]) / 2

        n_spatial_peaks = _spatial_density_peak_count(xy)

        print(f"\n{'─' * 65}")
        print(f"  cluster {idx}  (n_points={len(points)}, radius={radius:.2f}m, "
              f"n_spatial_peaks={n_spatial_peaks})")
        if max_radius is not None:
            would_split_attempt = radius > max_radius
            print(f"  radius vs max_radius={max_radius}: "
                  f"{'WOULD be size-gated into split_large_clusters()' if would_split_attempt else '⚠ at or below max_radius — production split_large_clusters() would SKIP this cluster entirely (gate 1)'}")
        print(f"{'─' * 65}")

        if not pcd.has_colors():
            print(f"  ❌ no colour data on this cluster — the color path "
                  f"can never fire regardless of thresholds. Check that "
                  f"the .ply was saved with RGB and that colors survived "
                  f"whatever selection/splitting produced this cluster.")
            continue

        colors = np.asarray(pcd.colors)
        per_point = _per_point_chroma(colors)

        any_bimodal = False
        any_spatially_separated = False

        for feat in color_features:
            fit = _fit_gmm_bimodality(per_point[feat], max_fit_points=max_fit_points)

            if np.isnan(fit["delta_bic"]):
                print(f"  {feat:<12} insufficient points for a stable GMM fit")
                continue

            bimodal = (
                fit["delta_bic"] >= delta_bic_threshold and
                fit["separation_in_std"] >= separation_threshold
            )

            if not bimodal:
                print(f"  {feat:<12} ❌ not bimodal  "
                      f"(delta_bic={fit['delta_bic']:.1f}, need >= {delta_bic_threshold};  "
                      f"separation={fit['separation_in_std']:.2f}σ, need >= {separation_threshold})")
                continue

            any_bimodal = True
            spatial = _color_groups_spatially_separated(
                xy, fit["gmm2"], fit["component_order"], per_point[feat],
                min_ratio=spatial_separation_ratio,
                min_absolute_m=min_intercentroid_m,
                min_group_size=min_spatial_group_size,
            )

            group_a, group_b = spatial["group_sizes"]
            if np.isnan(spatial["separation_ratio"]) and not np.isnan(group_a):
                smaller = min(group_a, group_b)
                print(f"  {feat:<12} ⚠  bimodal (delta_bic={fit['delta_bic']:.1f}) but "
                      f"one group too small to trust the spatial check "
                      f"(sizes={group_a}/{group_b}, need >= {min_spatial_group_size} "
                      f"each — smaller group has {smaller} points)")
                continue

            if spatial["spatially_separated"]:
                any_spatially_separated = True
                print(f"  {feat:<12} ✅ bimodal AND spatially separated  "
                      f"(delta_bic={fit['delta_bic']:.1f}, "
                      f"xy_sep={spatial['intercentroid_distance_m']:.2f}m, "
                      f"ratio={spatial['separation_ratio']:.2f}, "
                      f"need ratio>={spatial_separation_ratio} and "
                      f"dist>={min_intercentroid_m}m)")
            else:
                print(f"  {feat:<12} ⚠  bimodal (delta_bic={fit['delta_bic']:.1f}) but "
                      f"groups occupy the SAME footprint — looks like "
                      f"within-crown variation, not two trees  "
                      f"(xy_sep={spatial['intercentroid_distance_m']:.2f}m, "
                      f"ratio={spatial['separation_ratio']:.2f}, "
                      f"need ratio>={spatial_separation_ratio})")

        print()
        if any_spatially_separated and n_spatial_peaks < 2:
            print(f"  ✅ WOULD BE FLAGGED — colour says two trees, XY "
                  f"density says one")
        elif any_spatially_separated and n_spatial_peaks >= 2:
            print(f"  ⚠  colour signal found and spatially separated, but "
                  f"n_spatial_peaks={n_spatial_peaks} already >= 2 — this "
                  f"diagnostic's flag specifically targets cases XY density "
                  f"MISSES. XY already suspects two trees here, so the real "
                  f"question is why the production split_large_clusters() "
                  f"didn't already separate this cluster (Mean Shift not "
                  f"converging to 2 modes at your min_peak_distance, or a "
                  f"found split getting rejected by filter_cluster "
                  f"validation) — not a colour-augmentation problem.")
        elif any_bimodal and not any_spatially_separated:
            print(f"  ❌ not flagged — colour bimodality found but never "
                  f"spatially coherent. Either this cluster's merge (if "
                  f"real) is being masked by too strict a spatial gate, or "
                  f"this cluster isn't actually a merge and the colour "
                  f"bimodality is genuine within-crown variation.")
        else:
            print(f"  ❌ not flagged — no colour feature was even "
                  f"statistically bimodal. If you're confident this is a "
                  f"real merge, the colour signal between whatever two "
                  f"species are involved may just not be strong enough in "
                  f"chroma_r/g/b at Sunset Crater — worth checking "
                  f"feature_separability_colour.png for how well these "
                  f"channels separate species in general before trusting "
                  f"a per-point mixture test to find it.")




def get_color_spatial_override_candidates(clusters, cluster_indices,
                                           color_features=("chroma_g", "chroma_r", "chroma_b"),
                                           delta_bic_threshold=10.0,
                                           separation_threshold=1.0,
                                           spatial_separation_ratio=0.75,
                                           min_intercentroid_m=0.3,
                                           min_spatial_group_size=15,
                                           max_fit_points=800,
                                           verbose=True):
    """
    Identify clusters with at least one color feature that is both
    statistically bimodal AND spatially separated — deliberately WITHOUT
    the n_spatial_peaks / "XY density says one tree" requirement that
    diagnose_cluster_bimodality()'s combined flag uses.

    This exists for a specific downstream use: feeding
    split_large_clusters()'s new force_check_indices parameter, so clusters
    with a validated colour-driven merge signature can bypass the radius
    pre-gate (max_radius) even when their combined footprint isn't unusually
    wide — the case where two crowns overlap heavily rather than sitting
    side by side, producing a compact rather than large radius. Explaining
    a known-problem cluster list with explain_why_not_flagged() may well
    turn up exactly this pattern: strong bimodal+spatially-separated color
    evidence on a cluster that never even reached the density-peak check
    because its radius sat at or under max_radius.

    Deliberately reuses the SAME bimodal+spatial-separation logic as
    diagnose_cluster_bimodality() (not a looser delta_bic-only check) —
    a raw delta_bic threshold alone would reintroduce the sun/shade
    false-positive problem that motivated adding the spatial-separation
    gate in the first place. The spatial requirement is what keeps this
    override targeted at genuine spatial merges rather than ordinary
    within-crown colour variation.

    Args:
        clusters (list of o3d.geometry.PointCloud): Full cluster list.
        cluster_indices (list[int]): Which clusters to check — typically a
            known-problem list, NOT the full dataset (this is meant to
            build a small, deliberate override list, not a blanket rule).
        color_features, delta_bic_threshold, separation_threshold,
        spatial_separation_ratio, min_intercentroid_m,
        min_spatial_group_size, max_fit_points: Same meaning as
            diagnose_cluster_bimodality() — pass matching values.
        verbose (bool): Print which clusters qualified and why.
            Default True.

    Returns:
        list[int]: Cluster indices with at least one bimodal AND spatially
            separated color feature, sorted ascending. Feed this directly
            to split_large_clusters(force_check_indices=...).

    Requirements:
        numpy, scikit-learn, scipy.spatial.KDTree
    """
    candidates = []

    for idx in cluster_indices:
        if idx >= len(clusters):
            continue
        pcd = clusters[idx]
        if not pcd.has_colors():
            continue

        points = np.asarray(pcd.points)
        xy = points[:, :2]
        colors = np.asarray(pcd.colors)
        per_point = _per_point_chroma(colors)

        best_feat, best_ratio = None, -np.inf
        for feat in color_features:
            fit = _fit_gmm_bimodality(per_point[feat], max_fit_points=max_fit_points)
            if np.isnan(fit["delta_bic"]):
                continue
            bimodal = (
                fit["delta_bic"] >= delta_bic_threshold and
                fit["separation_in_std"] >= separation_threshold
            )
            if not bimodal:
                continue
            spatial = _color_groups_spatially_separated(
                xy, fit["gmm2"], fit["component_order"], per_point[feat],
                min_ratio=spatial_separation_ratio,
                min_absolute_m=min_intercentroid_m,
                min_group_size=min_spatial_group_size,
            )
            if spatial["spatially_separated"] and spatial["separation_ratio"] > best_ratio:
                best_feat, best_ratio = feat, spatial["separation_ratio"]

        if best_feat is not None:
            candidates.append(idx)
            if verbose:
                print(f"  cluster {idx}: override candidate "
                      f"(strongest feature={best_feat}, ratio={best_ratio:.2f})")

    if verbose:
        print(f"\n  {len(candidates)}/{len(cluster_indices)} clusters qualify "
              f"as radius-gate override candidates")

    return sorted(candidates)




def calibrate_thresholds(clusters, cluster_indices,
                          color_features=("chroma_g", "chroma_r", "chroma_b"),
                          check_height=True,
                          max_fit_points=800,
                          min_spatial_group_size=15,
                          verbose=True):
    """
    Compute raw delta_bic / separation_in_std / spatial-separation stats for
    a handful of clusters WITHOUT applying any pass/fail threshold — for
    picking real cutoffs by comparing known cases side by side, instead of
    guessing a threshold, running the full diagnostic, and re-guessing when
    the flagged count comes back at 0 or "basically every cluster."

    Spatial-separation stats are computed for every color feature
    regardless of whether it clears delta_bic_threshold (unlike the gated
    path in diagnose_cluster_bimodality(), which only bothers with the
    spatial check once a feature is already statistically bimodal) — that
    matters here because you want to see the FULL joint distribution of
    (delta_bic, separation_ratio) across known cases to pick both cutoffs
    together, not just the cases that already passed a first guess at
    delta_bic_threshold.

    Recommended workflow: include at least one cluster you have independent
    reason to believe is a real merge (e.g. 1370 — abnormal
    n_density_peaks, low crown_base_ratio) alongside a handful of ordinary
    clusters (random sample, or known-clean labeled ones). Real merges
    should show up as high delta_bic AND high separation_ratio together;
    ordinary clusters with incidental color bimodality (sun/shade) should
    show high delta_bic but LOW separation_ratio. Set delta_bic_threshold /
    spatial_separation_ratio in diagnose_cluster_bimodality() just above
    wherever the ordinary clusters top out, once you can see the actual
    numbers instead of guessing.

    Args:
        clusters (list of o3d.geometry.PointCloud): Full cluster list.
        cluster_indices (list[int]): Clusters to compute raw stats for —
            keep this small and deliberately chosen (known merge +
            comparison cases), not a full-dataset scan.
        color_features, check_height, max_fit_points, min_spatial_group_size:
            Same meaning as in diagnose_cluster_bimodality().
        verbose (bool): Print a comparison table. Default True.

    Returns:
        pd.DataFrame: One row per (cluster, feature), columns: "file",
            "feature", "n_points", "delta_bic", "separation_in_std",
            "intercentroid_distance_m", "separation_ratio", "group_sizes".
            No "bimodal" or flag column — this is raw numbers for you to
            eyeball, not a verdict.

    Requirements:
        numpy, pandas, scikit-learn, scipy.spatial.KDTree
    """
    features_to_test = list(color_features)
    if check_height:
        features_to_test.append("height_z")

    rows = []
    for idx in cluster_indices:
        if idx >= len(clusters):
            print(f"  ⚠  cluster index {idx} out of range — skipping")
            continue

        pcd = clusters[idx]
        points = np.asarray(pcd.points)
        xy = points[:, :2]
        z = points[:, 2]

        if not pcd.has_colors():
            print(f"  ⚠  cluster {idx} has no colour data — skipping")
            continue

        colors = np.asarray(pcd.colors)
        per_point = _per_point_chroma(colors)
        per_point["height_z"] = z

        for feat in features_to_test:
            fit = _fit_gmm_bimodality(per_point[feat], max_fit_points=max_fit_points)

            spatial = {"intercentroid_distance_m": np.nan,
                       "separation_ratio": np.nan, "group_sizes": (np.nan, np.nan)}
            if feat in color_features and fit["gmm2"] is not None:
                spatial = _color_groups_spatially_separated(
                    xy, fit["gmm2"], fit["component_order"], per_point[feat],
                    min_ratio=-np.inf,     # no gating here — always compute
                    min_absolute_m=-np.inf,
                    min_group_size=min_spatial_group_size,
                )

            rows.append({
                "file": idx,
                "feature": feat,
                "n_points": len(points),
                "delta_bic": fit["delta_bic"],
                "separation_in_std": fit["separation_in_std"],
                "intercentroid_distance_m": spatial["intercentroid_distance_m"],
                "separation_ratio": spatial["separation_ratio"],
                "group_sizes": spatial.get("group_sizes", (np.nan, np.nan)),
            })

    df = pd.DataFrame(rows)

    if verbose and len(df) > 0:
        print(f"\n{'═' * 70}")
        print("  RAW CALIBRATION STATS (no thresholds applied)")
        print(f"{'═' * 70}")
        color_only = df[df["feature"].isin(color_features)].sort_values(
            ["file", "delta_bic"], ascending=[True, False]
        )
        print(color_only[["file", "feature", "n_points", "delta_bic",
                          "separation_in_std", "intercentroid_distance_m",
                          "separation_ratio", "group_sizes"]].to_string(index=False))
        print(f"\n  Look for: known-merge clusters (e.g. 1370) should sit "
              f"high on BOTH delta_bic and separation_ratio. Ordinary "
              f"clusters with incidental color bimodality should show high "
              f"delta_bic but noticeably LOWER separation_ratio. Set "
              f"delta_bic_threshold / spatial_separation_ratio in "
              f"diagnose_cluster_bimodality() just above wherever the "
              f"ordinary clusters top out.\n")

    return df


# ── convenience: resolve flagged rows to actual .ply paths ────────────────────

def get_flagged_cluster_paths(df_diag, clusters_dir,
                               flag_column="color_bimodal_spatial_unimodal",
                               filename_template="cluster{idx}.ply",
                               verbose=True):
    """
    Resolve flagged cluster indices to actual file paths on disk, using the
    same "cluster{N}.ply" naming convention as save_clusters()/
    load_clusters() in global_files/save_clusters.py — so this only works
    correctly if clusters_dir is PATHS['Clusters'] for the SAME run that
    produced the `clusters` list passed to diagnose_cluster_bimodality()
    (indices are positions in that list, not any kind of stable ID).

    Args:
        df_diag (pd.DataFrame): Output of diagnose_cluster_bimodality() or
            diagnose_all_clusters().
        clusters_dir (str): Directory containing the .ply files — normally
            PATHS['Clusters'] from get_paths(trial_name).
        flag_column (str): Which boolean column in df_diag to filter on.
            Default "color_bimodal_spatial_unimodal".
        filename_template (str): Format string for the filename, with
            "{idx}" as the placeholder. Default "cluster{idx}.ply",
            matching save_clusters()'s naming exactly. Change this if
            you're pointing at a differently-named export folder (e.g.
            labeled_clusters/, which uses "<species>_<n>.ply" instead —
            that naming isn't recoverable from file index alone, so this
            function won't work there).
        verbose (bool): Print how many paths were found vs. missing.
            Default True.

    Returns:
        list[str]: Full paths to flagged clusters that actually exist on
            disk, sorted by file index. Missing files are skipped with a
            warning rather than included as broken paths.

    Requirements:
        pandas, os
    """
    if flag_column not in df_diag.columns:
        raise ValueError(
            f"get_flagged_cluster_paths: '{flag_column}' not in df_diag "
            f"columns ({list(df_diag.columns)}). Did you pass the output "
            f"of diagnose_cluster_bimodality() / diagnose_all_clusters()?"
        )

    flagged_indices = sorted(df_diag.loc[df_diag[flag_column], "file"].unique().tolist())

    paths = []
    missing = []
    for idx in flagged_indices:
        path = os.path.join(clusters_dir, filename_template.format(idx=int(idx)))
        if os.path.exists(path):
            paths.append(path)
        else:
            missing.append(path)

    if verbose:
        print(f"  {len(paths)} flagged cluster path(s) found "
              f"({len(flagged_indices)} flagged indices total)")
        if missing:
            print(f"  ⚠  {len(missing)} flagged index(es) had no matching "
                  f"file — clusters_dir may not match the run that produced "
                  f"df_diag, or clusters have been regenerated/reindexed "
                  f"since:")
            for m in missing[:10]:
                print(f"    {m}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

    return paths


def get_flagged_indices_from_sources(df_deep_clusters,
                                      misclassification_csv=None,
                                      n_density_peaks_top_n=10,
                                      n_density_peaks_min=3):
    """
    Build a candidate index list from sources already produced elsewhere in
    the pipeline, so you don't have to hand-type cluster indices.

    Combines:
        1. The top n_density_peaks_top_n clusters by n_density_peaks (with
           at least n_density_peaks_min peaks) — the same signal that
           flagged cluster 1370.
        2. Every "file" in the misclassification report CSV from
           inspect_misclassified_clusters(), if a path is given.

    Args:
        df_deep_clusters (pd.DataFrame): Must have "file" and
            "n_density_peaks" columns.
        misclassification_csv (str | None): Path to a CSV saved by
            inspect_misclassified_clusters() (e.g.
            PATHS['Dataframes'] + 'pinyon_misclassification_report.csv').
            Pass None to skip this source.
        n_density_peaks_top_n (int): How many top-n_density_peaks clusters
            to include. Default 10.
        n_density_peaks_min (int): Minimum n_density_peaks to be eligible
            at all — filters out clusters that are merely top-10 among a
            field that's mostly 1s and 2s. Default 3.

    Returns:
        list[int]: Sorted, deduplicated cluster "file" indices.

    Requirements:
        pandas
    """
    indices = set()

    if "n_density_peaks" in df_deep_clusters.columns:
        candidates = df_deep_clusters[
            df_deep_clusters["n_density_peaks"] >= n_density_peaks_min
        ].sort_values("n_density_peaks", ascending=False)
        top = candidates.head(n_density_peaks_top_n)
        indices.update(top["file"].astype(int).tolist())
        print(f"  {len(top)} clusters from n_density_peaks "
              f"(>= {n_density_peaks_min}, top {n_density_peaks_top_n})")

    if misclassification_csv is not None and os.path.exists(misclassification_csv):
        df_mis = pd.read_csv(misclassification_csv)
        if "file" in df_mis.columns:
            indices.update(df_mis["file"].astype(int).tolist())
            print(f"  {len(df_mis)} clusters from misclassification report")

    result = sorted(indices)
    print(f"  Combined: {len(result)} unique candidate clusters")
    return result