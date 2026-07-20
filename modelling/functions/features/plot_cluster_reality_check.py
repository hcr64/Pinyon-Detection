"""
plot_cluster_reality_check.py
────────────────────────────────────────────────────────────────────────────────
Sanity-check histograms for labeled clusters: do the detected cluster
geometries actually look like the trees they're labeled as?

Motivation
──────────
matching_score and classifier F1 tell you whether labels got attached to
*some* cluster and whether the model can tell species apart — neither tells
you whether the clusters themselves are geometrically sane. A pinyon label
sitting on a cluster with height=15m and n_points=12 means something broke
upstream (watershed over-segmentation, a bad split, ground points that
weren't stripped), even if match_labels_to_clusters() scored it as a
"perfect" 1:1 match.

This module plots per-species histograms for the features most diagnostic of
real tree shape, plus a printed stats table, so you can eyeball whether e.g.
ponderosa clusters are actually taller/more columnar than pinyon (expected),
or whether the three species distributions are suspiciously identical
(suggesting geometry isn't actually varying — a red flag).

Usage
─────
    from plot_cluster_reality_check import plot_cluster_reality_check

    plot_cluster_reality_check(
        df_deep_clusters,
        df_clusters,
        save_path=PATHS['Images']
    )

Call after match_labels_to_clusters() (needs "Name") and after
engineer_features() (needs height_to_radius, crown_volume, etc.) — same
prerequisites as plot_feature_separability().
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


SPECIES_STYLE = {
    "pinyon":    {"color": "#e05c5c"},
    "juniper":   {"color": "#5b8dd9"},
    "ponderosa": {"color": "#f5a623"},
}

# feature -> (display label, expected direction note printed alongside stats)
# "expected" notes are just docstring-level field knowledge, not enforced —
# they're there so the printed table is readable without cross-referencing
# get_deep_cluster_features.py every time.
REALITY_CHECK_FEATURES = {
    "height":            ("Height (m)",
                           "ponderosa generally tallest"),
    "radius":             ("Crown radius (m)",
                           "pinyon/juniper wider relative to height"),
    "height_to_radius":  ("Height / radius ratio",
                           "ponderosa highest (tall+narrow), pinyon lowest"),
    "verticality":       ("Verticality (height / max horiz OBB extent)",
                           "ponderosa highest — columnar crown"),
    "flatness_ratio":    ("Flatness ratio (planarity / height)",
                           "juniper highest — sprawls laterally"),
    "crown_base_ratio":  ("Crown base ratio (upper/lower point density)",
                           "ponderosa highest — sparse lower trunk, dense high canopy"),
    "n_points":          ("Points per cluster",
                           "sanity check — near-zero or huge values indicate bad splits"),
    "crown_volume":      ("Crown volume (radius^2 * height)",
                           "rough proxy for total canopy mass"),
}


def plot_cluster_reality_check(df_deep, df_labels_matched, save_path,
                                features=None, bins=25):
    """
    Plot per-species histograms of geometric features and print summary
    stats, as a sanity check that labeled clusters resemble real trees.

    Two things to look for in the output:
        1. Species should visibly separate on at least some features
           (e.g. height_to_radius, verticality) — if all three species
           produce near-identical distributions, the clusters likely aren't
           capturing real shape differences (or the label matching is
           wrong).
        2. Outliers/tails that don't make ecological sense — e.g. a
           "pinyon" cluster with height > 15m, or any cluster with
           n_points < ~20 (likely a split fragment, not a whole tree) —
           flag specific clusters worth opening in Colab/Plotly for a look.

    Saves one PNG with a grid of histograms (one subplot per feature) to
    `save_path/cluster_reality_check_histograms.png`, and prints a
    per-species mean/std/min/max table to stdout for each feature.

    Args:
        df_deep (pd.DataFrame):
            Output of make_deep_dataframe() + engineer_features(). Must
            contain a "file" column and the feature columns being plotted.
        df_labels_matched (pd.DataFrame):
            Output of match_labels_to_clusters(). Must have "file" and
            "Name".
        save_path (str):
            Directory to write the PNG into. Created if it doesn't exist.
        features (dict | None):
            Override REALITY_CHECK_FEATURES. Keys are column names in
            df_deep, values are (display_label, note) tuples. Pass None to
            use the default set. Columns not present in df_deep are skipped
            with a warning rather than raising.
        bins (int):
            Histogram bin count, shared across all species/features for
            visual comparability. Default 25.

    Returns:
        pd.DataFrame: Per-species summary stats (mean, std, min, max, n)
            for every plotted feature — same table that's printed, returned
            for further inspection/logging.

    Requirements:
        numpy, pandas, matplotlib
    """
    os.makedirs(save_path, exist_ok=True)
    features = features or REALITY_CHECK_FEATURES

    # ── merge features + labels, keep only real species ──────────────────
    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file")
    df = df[df["Name"].isin(["pinyon", "juniper", "ponderosa"])]

    if len(df) == 0:
        print("plot_cluster_reality_check: no labeled clusters found, skipping.")
        return pd.DataFrame()

    counts = df["Name"].value_counts()
    print(f"\n{'═' * 60}")
    print("  CLUSTER REALITY CHECK — labeled cluster geometry")
    print(f"{'═' * 60}")
    print(f"  Labeled clusters: {len(df)}")
    print(counts.to_string())
    print()

    available = {f: v for f, v in features.items() if f in df.columns}
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠  Skipping missing columns: {missing}\n")

    # ── build grid ─────────────────────────────────────────────────────────
    n_feats = len(available)
    n_cols  = 3
    n_rows  = int(np.ceil(n_feats / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flat

    stats_rows = []

    for ax, (feat, (label, note)) in zip(axes, available.items()):
        # shared bin edges across species so overlapping histograms are
        # visually comparable rather than each auto-scaling independently
        all_vals = df[feat].dropna().values
        if len(all_vals) == 0:
            ax.set_visible(False)
            continue
        edges = np.histogram_bin_edges(all_vals, bins=bins)

        for species, style in SPECIES_STYLE.items():
            vals = df.loc[df["Name"] == species, feat].dropna().values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=edges, alpha=0.5, color=style["color"],
                    label=f"{species} (n={len(vals)})", density=True)

            stats_rows.append({
                "feature": feat,
                "species": species,
                "n":       len(vals),
                "mean":    round(float(np.mean(vals)), 3),
                "std":     round(float(np.std(vals)), 3),
                "min":     round(float(np.min(vals)), 3),
                "max":     round(float(np.max(vals)), 3),
            })

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.set_title(note, fontsize=8, style="italic")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    for ax in list(axes)[n_feats:]:
        ax.set_visible(False)

    fig.suptitle("Cluster Reality Check — feature distributions by species",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out = os.path.join(save_path, "cluster_reality_check_histograms.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}\n")

    # ── printed stats table ──────────────────────────────────────────────
    df_stats = pd.DataFrame(stats_rows)
    for feat in available:
        sub = df_stats[df_stats["feature"] == feat]
        if len(sub) == 0:
            continue
        print(f"  {feat}  ({available[feat][1]})")
        print(sub[["species", "n", "mean", "std", "min", "max"]]
              .to_string(index=False))
        print()

    # ── flag likely-bad clusters ─────────────────────────────────────────
    _flag_suspicious_clusters(df)

    return df_stats


def _flag_suspicious_clusters(df):
    """
    Print clusters whose geometry is implausible for any tree, regardless
    of species — very low point count, near-zero height, or extreme
    height/radius ratio. These are candidates for the same kind of visual
    QC as multi_match_clusters/ — pull the .ply and look at it.
    """
    flags = []

    if "n_points" in df.columns:
        thin = df[df["n_points"] < 20]
        for _, row in thin.iterrows():
            flags.append(f"file={int(row['file'])} ({row['Name']}): "
                         f"only {int(row['n_points'])} points — likely a split fragment")

    if "height" in df.columns:
        flat = df[df["height"] < 0.5]
        for _, row in flat.iterrows():
            flags.append(f"file={int(row['file'])} ({row['Name']}): "
                         f"height={row['height']:.2f}m — likely ground/shrub, not a tree")

    if "height_to_radius" in df.columns:
        extreme = df[df["height_to_radius"] > df["height_to_radius"].quantile(0.99)]
        for _, row in extreme.iterrows():
            flags.append(f"file={int(row['file'])} ({row['Name']}): "
                         f"height/radius={row['height_to_radius']:.2f} — top 1%, check for a bad split")

    if flags:
        print(f"  ⚠  {len(flags)} clusters flagged for visual QC:")
        for f in flags[:20]:
            print(f"    {f}")
        if len(flags) > 20:
            print(f"    ... and {len(flags) - 20} more")
        print()
    else:
        print("  No suspicious clusters flagged.\n")