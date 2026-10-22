"""
plot_prediction_map.py
────────────────────────────────────────────────────────────────────────────────
Top-down map of every detected cluster, drawn as crown-sized circles coloured
by predicted species (pinyons split by predicted drought tolerance), on top of
a true-colour top-down image.

Basemap sources, in priority order:
    1. ortho_path   — an orthomosaic GeoTIFF (e.g. exported from PixMapper4D),
                      if you have one. Sharpest result.
    2. pcd_path     — the raw coloured point cloud, rasterised top-down
                      (highest point per cell wins). Cached to disk after the
                      first run so later runs are instant.
    3. chm_path     — greyscale CHM fallback.

Requirements: numpy, pandas, matplotlib, rasterio, open3d, scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

STYLE = {
    "pinyon_tolerant":    ("#2e9e4f", "Pinyon – drought tolerant"),
    "pinyon_susceptible": ("#d9342b", "Pinyon – drought susceptible"),
    "pinyon_unknown":     ("#9467bd", "Pinyon – drought n/a"),
    "pinyon":             ("#e05c5c", "Pinyon"),
    "juniper":            ("#5b8dd9", "Juniper"),
    "ponderosa":          ("#f5a623", "Ponderosa"),
}


# ── basemap builders ──────────────────────────────────────────────────────────

def build_topdown_basemap(pcd_path, cache_path=None, resolution=0.15,
                          max_fill_px=4, force=False):
    """
    Rasterise a coloured point cloud into a top-down RGB image.

    For each grid cell the HIGHEST point's colour is used, which is what a
    camera looking straight down would see (canopy on top, ground between).
    Small holes are filled from the nearest filled cell.

    Args:
        pcd_path (str): .ply with colours (PATHS['Raw_pcd']).
        cache_path (str | None): .npz to cache the result. Reused next run.
        resolution (float): metres per pixel. 0.1-0.2 is a good range;
            smaller = sharper but more memory.
        max_fill_px (int): holes further than this many pixels from any
            point stay blank (so scan edges aren't smeared outward).
        force (bool): rebuild even if the cache exists.

    Returns:
        img (np.ndarray): (H, W, 3) uint8, row 0 = north.
        extent (tuple): (left, right, bottom, top) in UTM metres.
    """
    if cache_path and os.path.exists(cache_path) and not force:
        z = np.load(cache_path)
        print(f"  Loaded cached basemap ← {cache_path}")
        return z["img"], tuple(z["extent"])

    import open3d as o3d
    from scipy.ndimage import distance_transform_edt

    print(f"  Building top-down basemap from {pcd_path} ...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_colors():
        raise ValueError("Point cloud has no colours — can't build RGB basemap.")

    # thin the cloud first: no point having 8 cm points for a 15 cm raster
    pcd = pcd.voxel_down_sample(voxel_size=resolution / 2)
    pts = np.asarray(pcd.points)
    rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    n_cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    n_rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    col = np.clip(((pts[:, 0] - x_min) / resolution).astype(int), 0, n_cols - 1)
    row = np.clip(((y_max - pts[:, 1]) / resolution).astype(int), 0, n_rows - 1)

    # low → high, so the highest point in each cell is written last and wins
    order = np.argsort(pts[:, 2], kind="stable")
    img = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    filled = np.zeros((n_rows, n_cols), dtype=bool)
    img[row[order], col[order]] = rgb[order]
    filled[row, col] = True

    # fill small holes from the nearest filled pixel
    dist, (ri, ci) = distance_transform_edt(~filled, return_indices=True)
    holes = (~filled) & (dist <= max_fill_px)
    img[holes] = img[ri[holes], ci[holes]]
    valid = filled | holes

    # unscanned area → white
    img[~valid] = 255

    extent = (x_min, x_min + n_cols * resolution, y_max - n_rows * resolution, y_max)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, img=img, extent=np.array(extent))
        print(f"  Cached basemap → {cache_path}  ({n_cols}×{n_rows}px)")

    return img, extent


def _load_ortho(ortho_path, max_dim=5000):
    """Read an RGB(A) orthomosaic GeoTIFF, decimated to <= max_dim px."""
    import rasterio
    with rasterio.open(ortho_path) as src:
        step = max(1, max(src.width, src.height) // max_dim)
        bands = min(3, src.count)
        arr = src.read(list(range(1, bands + 1)),
                       out_shape=(bands, src.height // step, src.width // step))
        b = src.bounds
    img = np.moveaxis(arr, 0, -1)
    if img.dtype != np.uint8:
        img = np.clip(img / img.max() * 255, 0, 255).astype(np.uint8)
    return img, (b.left, b.right, b.bottom, b.top)


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_prediction_map(df_deep, df_clusters, save_path, chm_path=None,
                        pcd_path=None, ortho_path=None, basemap_cache=None,
                        basemap_resolution=0.15,
                        drought_col="predicted_drought_class",
                        min_confidence=None, show_ground_truth=True,
                        circle_alpha=0.30, min_circle_radius=0.5,
                        filename="prediction_map.png", title=None):
    """
    Save a top-down map of predicted tree labels, as circles on a colour
    aerial-style image.

    Args:
        df_deep (pd.DataFrame): needs "file" and "predicted_label"; optional
            `drought_col`, "radius", "prob_<species>".
        df_clusters (pd.DataFrame): needs "file", "x_pos", "y_pos"; optional "Name".
        save_path (str): directory for the PNG.
        pcd_path (str | None): raw coloured .ply (PATHS['Raw_pcd']) used to
            render the basemap.
        ortho_path (str | None): orthomosaic GeoTIFF; takes priority over pcd_path.
        basemap_cache (str | None): .npz path to cache the rendered basemap.
        basemap_resolution (float): metres/pixel for the point-cloud render.
        chm_path (str | None): greyscale fallback if no colour source works.
        min_confidence (float | None): clusters below this top-species
            probability are drawn as small faint grey dots instead of circles.
        circle_alpha (float): fill opacity of the circles (edges are solid).
        min_circle_radius (float): smallest circle drawn, in metres, so tiny
            clusters stay visible.

    Returns:
        str: path of the saved PNG.
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

    low_conf = pd.Series(False, index=df.index)
    if min_confidence is not None:
        prob_cols = [c for c in df.columns
                     if c.startswith("prob_") and not c.startswith("prob_drought_")]
        if prob_cols:
            low_conf = df[prob_cols].max(axis=1) < min_confidence

    # circle radius in METRES (data units), so circles match real crown size
    if "radius" in df.columns:
        radii = df["radius"].clip(lower=min_circle_radius, upper=8).values
    else:
        radii = np.full(len(df), 1.5)

    pad = 10
    xlim = (df["x_pos"].min() - pad, df["x_pos"].max() + pad)
    ylim = (df["y_pos"].min() - pad, df["y_pos"].max() + pad)

    fig, ax = plt.subplots(figsize=(14, 14))

    # ── basemap ───────────────────────────────────────────────────────────
    basemap_drawn = False
    try:
        img = extent = None
        if ortho_path and os.path.exists(ortho_path):
            img, extent = _load_ortho(ortho_path)
        elif pcd_path and os.path.exists(pcd_path):
            img, extent = build_topdown_basemap(
                pcd_path, cache_path=basemap_cache, resolution=basemap_resolution)
        if img is not None:
            ax.imshow(img, extent=extent, zorder=0, interpolation="bilinear")
            basemap_drawn = True
    except Exception as e:
        print(f"  ⚠  Could not build colour basemap ({e}) — trying CHM fallback.")

    if not basemap_drawn and chm_path and os.path.exists(chm_path):
        try:
            import rasterio
            with rasterio.open(chm_path) as src:
                step = max(1, max(src.width, src.height) // 4000)
                arr = src.read(1, out_shape=(src.height // step, src.width // step))
                b = src.bounds
            arr = np.ma.masked_invalid(arr)
            ax.imshow(arr, extent=(b.left, b.right, b.bottom, b.top), cmap="Greys",
                      vmin=0, vmax=float(np.nanpercentile(arr.filled(np.nan), 99)),
                      zorder=0)
        except Exception as e:
            print(f"  ⚠  CHM basemap failed too ({e}) — drawing without basemap.")

    # ── circles ───────────────────────────────────────────────────────────
    from matplotlib.colors import to_rgba
    handles = []

    if low_conf.any():
        ax.scatter(df.loc[low_conf, "x_pos"], df.loc[low_conf, "y_pos"],
                   s=8, c="#dddddd", edgecolors="black", linewidths=0.3,
                   alpha=0.7, zorder=1)
        handles.append(Line2D([], [], marker="o", ls="", mfc="#dddddd", mec="black",
                              ms=5, label=f"Low confidence (<{min_confidence}) "
                                          f"(n={int(low_conf.sum())})"))

    for key, (color, label) in STYLE.items():
        m = ((df["_cat"] == key) & ~low_conf).values
        if not m.any():
            continue
        patches = [Circle((x, y), r) for x, y, r in
                   zip(df["x_pos"].values[m], df["y_pos"].values[m], radii[m])]
        ax.add_collection(PatchCollection(
            patches,
            facecolor=to_rgba(color, circle_alpha),
            edgecolor=to_rgba(color, 1.0),
            linewidths=1.6, zorder=2))
        handles.append(Line2D([], [], marker="o", ls="", mfc=to_rgba(color, circle_alpha),
                              mec=color, mew=1.6, ms=10,
                              label=f"{label} (n={int(m.sum())})"))

    # ── GPS ground truth ring ─────────────────────────────────────────────
    if show_ground_truth and "Name" in df.columns:
        g = df["Name"].isin(["pinyon", "juniper", "ponderosa"])
        if g.any():
            ax.scatter(df.loc[g, "x_pos"], df.loc[g, "y_pos"], s=18, c="white",
                       edgecolors="black", linewidths=0.8, zorder=3)
            handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec="black",
                                  ms=5, label=f"GPS-labeled (n={int(g.sum())})"))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Easting (m, UTM 12N)")
    ax.set_ylabel("Northing (m, UTM 12N)")
    ax.set_title(title or f"Predicted tree species map ({len(df)} clusters)")
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    out = os.path.join(save_path, filename)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved prediction map → {out}")
    return out