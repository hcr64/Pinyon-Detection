"""
plot_feature_separability.py
────────────────────────────────────────────────────────────────────────────────
Scatter-plot grid showing how well each feature pair separates pinyon, juniper,
and ponderosa. Run this after make_deep_dataframe() + engineer_features() and
after match_labels_to_clusters() so the "Name" column is present.

Usage (add to main.py inside the Train_Model block, or run standalone):

    from plot_feature_separability import plot_feature_separability

    plot_feature_separability(
        df_deep_clusters,
        df_clusters,
        save_path=PATHS['Images']
    )
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ── colour / marker scheme ────────────────────────────────────────────────────
SPECIES_STYLE = {
    "pinyon":    {"color": "#e05c5c", "marker": "o"},
    "juniper":   {"color": "#5b8dd9", "marker": "s"},
    "ponderosa": {"color": "#f5a623", "marker": "^"},
}


def plot_feature_separability(df_deep, df_labels_matched, save_path,
                               feature_pairs=None):
    """
    Save a grid of scatter plots showing class separation in feature space.

    Each subplot shows one pair of features with points coloured by species.
    Well-separated clusters in a plot mean those two features are jointly
    discriminative. Heavy overlap means the model cannot use that pair to
    distinguish species — either the feature is redundant or more labeled
    data is needed.

    Two plots are saved:
        feature_separability_geometry.png — shape and structural features
        feature_separability_colour.png   — colour and PCA features

    Args:
        df_deep (pd.DataFrame):
            Output of make_deep_dataframe() + engineer_features(). Must
            contain a "file" column and all plotted feature columns.
        df_labels_matched (pd.DataFrame):
            Output of match_labels_to_clusters(). Must have "file" and "Name".
        save_path (str):
            Directory to write PNGs into. Created if it does not exist.
        feature_pairs (list of (str, str) | None):
            Override the default feature pairs to plot. Each entry is a tuple
            of two column names. Pass None to use the defaults below.

    Returns:
        None — saves PNGs to save_path.

    Requirements:
        numpy, pandas, matplotlib
    """

    os.makedirs(save_path, exist_ok=True)

    # ── merge so we have labels alongside features ────────────────────────────
    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file")
    df = df[df["Name"].isin(["pinyon", "juniper", "ponderosa"])]

    if len(df) == 0:
        print("plot_feature_separability: no labeled clusters found, skipping.")
        return

    n_per_species = df["Name"].value_counts()
    print(f"Plotting separability for {len(df)} labeled clusters:")
    print(n_per_species.to_string())
    print()

    # ── geometry / structural plot ────────────────────────────────────────────
    geometry_pairs = [
        ("height",           "radius"),
        ("height_to_radius", "crown_volume"),
        ("verticality",      "height"),
        ("flatness_ratio",   "planarity"),
        ("crown_base_ratio", "height_to_radius"),
        ("crown_base_ratio", "verticality"),
    ]

    _make_scatter_grid(
        df, geometry_pairs,
        title="Feature Separability — Geometry & Structure",
        filename=os.path.join(save_path, "feature_separability_geometry.png"),
    )

    # ── colour / PCA plot ─────────────────────────────────────────────────────
    colour_pairs = [
        ("mean_g",          "mean_r"),
        ("green_dominance", "color_saturation"),
        ("mean_g",          "mean_b"),
        ("std_r",           "std_g"),
        ("linearity",       "planarity"),
        ("sphericity",      "eigenvalue_1"),
    ]

    _make_scatter_grid(
        df, colour_pairs,
        title="Feature Separability — Colour & PCA",
        filename=os.path.join(save_path, "feature_separability_colour.png"),
    )

    # ── per-feature box plots ─────────────────────────────────────────────────
    # shows distribution of each feature per species — easier to see if a
    # single feature separates classes without needing to pick pairs
    _make_boxplot_grid(
        df,
        filename=os.path.join(save_path, "feature_separability_boxplots.png"),
    )


def _make_scatter_grid(df, pairs, title, filename):
    """Draw a 2×3 scatter grid for the given feature pairs and save it."""
    n_cols  = 3
    n_rows  = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flat

    for ax, (fx, fy) in zip(axes, pairs):
        # check both features exist
        if fx not in df.columns or fy not in df.columns:
            ax.set_visible(False)
            continue

        for species, style in SPECIES_STYLE.items():
            mask = df["Name"] == species
            if mask.sum() == 0:
                continue
            ax.scatter(
                df.loc[mask, fx],
                df.loc[mask, fy],
                c=style["color"],
                marker=style["marker"],
                label=f"{species} (n={mask.sum()})",
                alpha=0.65,
                s=30,
                edgecolors="none",
            )

        ax.set_xlabel(fx, fontsize=9)
        ax.set_ylabel(fy, fontsize=9)
        ax.tick_params(labelsize=8)

    # hide unused subplots
    for ax in list(axes)[len(pairs):]:
        ax.set_visible(False)

    # single legend on the first axis
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9,
               bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")


def _make_boxplot_grid(df, filename):
    """
    Draw one box-per-species for every feature, arranged in a grid.

    A feature where the three boxes barely overlap is a strong discriminator.
    A feature where all three boxes stack on top of each other is useless.
    """
    # all numeric columns except bookkeeping ones
    skip    = {"file", "Name", "predicted_label"}
    feats   = [c for c in df.select_dtypes(include=np.number).columns
               if c not in skip and not c.startswith("prob_")]
    species = ["pinyon", "juniper", "ponderosa"]
    colors  = [SPECIES_STYLE[s]["color"] for s in species]

    n_cols  = 4
    n_rows  = int(np.ceil(len(feats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flat

    for ax, feat in zip(axes, feats):
        data = [df.loc[df["Name"] == sp, feat].dropna().values
                for sp in species]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 1.5})

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(feat, fontsize=8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([s[:3] for s in species], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for ax in list(axes)[len(feats):]:
        ax.set_visible(False)

    # legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=SPECIES_STYLE[s]["color"],
                            label=s, alpha=0.7) for s in species]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.suptitle("Per-feature Distribution by Species", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")