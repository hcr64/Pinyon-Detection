"""
plot_prediction_map.py
────────────────────────────────────────────────────────────────────────────────
2D map of every detected cluster at its UTM position, coloured by predicted
species, with pinyons split by predicted drought tolerance (if available).
The CHM is drawn underneath as a basemap when the GeoTIFF exists.

Requirements: numpy, pandas, matplotlib, rasterio (already in the venv)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "pinyon_tolerant":    ("#2e9e4f", "Pinyon – drought tolerant"),
    "pinyon_susceptible": ("#d9342b", "Pinyon – drought susceptible"),
    "pinyon_unknown":     ("#9467bd", "Pinyon – drought n/a"),
    "pinyon":             ("#e05c5c", "Pinyon"),
    "juniper":            ("#5b8dd9", "Juniper"),
    "ponderosa":          ("#f5a623", "Ponderosa"),
}


def plot_prediction_map(df_deep, df_clusters, save_path, chm_path=None,
                        drought_col="predicted_drought_class",
                        min_confidence=None, show_ground_truth=True,
                        scale_by_crown=True, filename="prediction_map.png",
                        title=None):
    """
    Save a top-down map of predicted tree labels.

    Args:
        df_deep (pd.DataFrame): Must have "file" and "predicted_label"
            (added by train_tree_classifier / run_advanced_classifiers).
            Optional: `drought_col`, "radius", "prob_<species>" columns.
        df_clusters (pd.DataFrame): Must have "file", "x_pos", "y_pos".
            Optional "Name" (GPS label) for the ground-truth overlay.
        save_path (str): Directory for the PNG.
        chm_path (str | None): CHM GeoTIFF used as a greyscale basemap.
        drought_col (str): Column holding "tolerant"/"susceptible" for
            predicted pinyons. If missing, pinyons are one colour.
        min_confidence (float | None): Clusters whose top species
            probability is below this are drawn as faint grey dots.
        show_ground_truth (bool): Ring the GPS-labeled clusters in black.
        scale_by_crown (bool): Marker area scales with crown radius.

    Returns:
        str: Path of the saved PNG.
    """
    os.makedirs(save_path, exist_ok=True)

    if "predicted_label" not in df_deep.columns:
        raise ValueError("df_deep has no 'predicted_label' — run a classifier first.")

    pos_cols = [c for c in ("file", "x_pos", "y_pos", "Name") if c in df_clusters.columns]
    df = df_deep.merge(df_clusters[pos_cols], on="file", how="inner")

    # ── categories ────────────────────────────────────────────────────────
    cat = df["predicted_label"].astype(str).copy()
    is_pin = cat == "pinyon"
    if drought_col in df.columns:
        d = df[drought_col]
        cat[is_pin & (d == "tolerant")]    = "pinyon_tolerant"
        cat[is_pin & (d == "susceptible")] = "pinyon_susceptible"
        cat[is_pin & ~d.isin(["tolerant", "susceptible"])] = "pinyon_unknown"
    df["_cat"] = cat

    # ── low-confidence mask ───────────────────────────────────────────────
    low_conf = pd.Series(False, index=df.index)
    if min_confidence is not None:
        prob_cols = [c for c in df.columns
                     if c.startswith("prob_") and not c.startswith("prob_drought_")]
        if prob_cols:
            low_conf = df[prob_cols].max(axis=1) < min_confidence

    sizes = (8 + 6 * df["radius"].clip(upper=6) ** 2) if (
        scale_by_crown and "radius" in df.columns) else pd.Series(25, index=df.index)

    fig, ax = plt.subplots(figsize=(14, 14))

    # ── CHM basemap ───────────────────────────────────────────────────────
    if chm_path and os.path.exists(chm_path):
        try:
            import rasterio
            with rasterio.open(chm_path) as src:
                step = max(1, max(src.width, src.height) // 4000)
                arr = src.read(1, out_shape=(src.height // step, src.width // step))
                b = src.bounds
            arr = np.ma.masked_invalid(arr)
            ax.imshow(arr, extent=(b.left, b.right, b.bottom, b.top),
                      cmap="Greys", vmin=0,
                      vmax=float(np.nanpercentile(arr.filled(np.nan), 99)),
                      alpha=0.55, zorder=0)
        except Exception as e:
            print(f"  ⚠  Could not draw CHM basemap ({e}) — continuing without it.")

    # ── clusters ──────────────────────────────────────────────────────────
    if low_conf.any():
        ax.scatter(df.loc[low_conf, "x_pos"], df.loc[low_conf, "y_pos"],
                   s=6, c="#bbbbbb", alpha=0.5, linewidths=0, zorder=1,
                   label=f"Low confidence (<{min_confidence}) (n={int(low_conf.sum())})")

    for key, (color, label) in STYLE.items():
        m = (df["_cat"] == key) & ~low_conf
        if not m.any():
            continue
        ax.scatter(df.loc[m, "x_pos"], df.loc[m, "y_pos"], s=sizes[m],
                   c=color, alpha=0.8, edgecolors="white", linewidths=0.3,
                   label=f"{label} (n={int(m.sum())})", zorder=2)

    # ── GPS ground truth ring ─────────────────────────────────────────────
    if show_ground_truth and "Name" in df.columns:
        g = df["Name"].isin(["pinyon", "juniper", "ponderosa"])
        if g.any():
            ax.scatter(df.loc[g, "x_pos"], df.loc[g, "y_pos"], s=70,
                       facecolors="none", edgecolors="black", linewidths=0.8,
                       label=f"GPS-labeled (n={int(g.sum())})", zorder=3)

    pad = 10
    ax.set_xlim(df["x_pos"].min() - pad, df["x_pos"].max() + pad)
    ax.set_ylim(df["y_pos"].min() - pad, df["y_pos"].max() + pad)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Easting (m, UTM 12N)")
    ax.set_ylabel("Northing (m, UTM 12N)")
    ax.set_title(title or f"Predicted tree species map ({len(df)} clusters)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(save_path, filename)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved prediction map → {out}")
    return out