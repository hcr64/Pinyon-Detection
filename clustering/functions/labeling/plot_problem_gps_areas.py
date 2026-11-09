import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import KDTree


def plot_problem_gps_areas(csv_coords, df_labels, df_clusters, point_cloud,
                           save_path, max_distance=4.0, view_radius=8.0,
                           statuses=("no_match", "multi"), page_size=12,
                           n_cols=4, point_size=2, dpi=130, combine_pdf=True):
    """
    Save top-down views of every GPS point that failed 1:1 matching.

    Each panel shows the raw point cloud (true colour, highest point on top)
    around one GPS label, the max_distance match radius (red dashed circle),
    the GPS point (yellow star), and every cluster centroid in view (cyan x,
    annotated with its df_clusters "file" index).

    Panels are split into pages of page_size and all pages are written, so
    no manual paging is needed.

    Args:
        csv_coords (np.ndarray): (M, 2) UTM GPS coords, aligned to df_labels
            (post coverage filter, as in match_labels_to_clusters()).
        df_labels (pd.DataFrame): Label frame with "Name".
        df_clusters (pd.DataFrame): Needs "x_pos", "y_pos", "file".
        point_cloud (o3d.geometry.PointCloud): Raw coloured cloud.
        save_path (str): Output directory; existing page_*.png are removed.
        max_distance (float): Match radius, drawn as the red circle and used
            to classify status.
        view_radius (float): Half-width of each panel in metres.
        statuses (tuple[str]): Which statuses to plot: "no_match" (0 clusters
            in range) and/or "multi" (2+).
        page_size (int): Panels per page.
        n_cols (int): Panels per row.
        point_size (float): Scatter marker size for cloud points.
        dpi (int): PNG resolution.
        combine_pdf (bool): Also write all pages to one problem_gps_areas.pdf.

    Returns:
        pd.DataFrame: One row per plotted GPS point (gps_idx, Name, x, y,
            status, n_clusters_in_range).

    Requirements:
        numpy, pandas, matplotlib, scipy.spatial.KDTree
    """
    cl_xy = df_clusters[["x_pos", "y_pos"]].values
    cl_tree = KDTree(cl_xy)
    n_in_range = np.array([len(m) for m in
                           cl_tree.query_ball_point(csv_coords, r=max_distance)])

    status = np.select([n_in_range == 0, n_in_range == 1],
                       ["no_match", "perfect"], default="multi")

    df_bad = pd.DataFrame({
        "gps_idx": np.arange(len(csv_coords)),
        "Name": df_labels["Name"].values,
        "x": csv_coords[:, 0],
        "y": csv_coords[:, 1],
        "status": status,
        "n_clusters_in_range": n_in_range,
    })
    df_bad = df_bad[df_bad["status"].isin(statuses)].reset_index(drop=True)

    if len(df_bad) == 0:
        print("plot_problem_gps_areas: no problem GPS points, nothing to plot.")
        return df_bad

    # raw cloud -> arrays; sort by Z so the highest point is drawn last
    pts = np.asarray(point_cloud.points)
    order = np.argsort(pts[:, 2], kind="stable")
    pts = pts[order]
    cols = (np.asarray(point_cloud.colors)[order] if point_cloud.has_colors()
            else np.full((len(pts), 3), 0.5))

    # points visible in each panel (sqrt(2) so the square corners fill too)
    pt_tree = KDTree(pts[:, :2])
    member_idx = pt_tree.query_ball_point(df_bad[["x", "y"]].values,
                                          r=view_radius * np.sqrt(2))
    member_idx = [np.sort(np.asarray(m, dtype=int)) for m in member_idx]  # keeps Z order

    def draw_area(ax, i):
        row = df_bad.iloc[i]
        gx, gy = row["x"], row["y"]
        idx = member_idx[i]
        if len(idx):
            ax.scatter(pts[idx, 0], pts[idx, 1], c=cols[idx], s=point_size)
        ax.add_patch(Circle((gx, gy), max_distance, fill=False,
                            ec="red", ls="--", lw=1))
        ax.scatter([gx], [gy], marker="*", s=180, c="yellow",
                   ec="black", zorder=5)

        near = df_clusters[np.hypot(df_clusters["x_pos"] - gx,
                                    df_clusters["y_pos"] - gy) <= view_radius * np.sqrt(2)]
        ax.scatter(near["x_pos"], near["y_pos"], marker="x", c="cyan",
                   s=50, zorder=4)
        for r in near.itertuples():
            ax.annotate(int(r.file), (r.x_pos, r.y_pos), color="white",
                        fontsize=7, xytext=(3, 3), textcoords="offset points",
                        bbox=dict(fc="black", alpha=0.5, pad=0.5, lw=0))

        ax.set_xlim(gx - view_radius, gx + view_radius)
        ax.set_ylim(gy - view_radius, gy + view_radius)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"#{int(row.gps_idx)} {row.Name} [{row.status}]\n"
                     f"{int(row.n_clusters_in_range)} clusters in range",
                     fontsize=8)

    # clear old pages
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            if f.startswith("page_") and f.endswith(".png"):
                os.remove(os.path.join(save_path, f))
    else:
        os.makedirs(save_path)

    n_pages = int(np.ceil(len(df_bad) / page_size))
    pdf = PdfPages(os.path.join(save_path, "problem_gps_areas.pdf")) if combine_pdf else None

    for page in range(n_pages):
        sel = range(page * page_size, min((page + 1) * page_size, len(df_bad)))
        n_rows = int(np.ceil(len(sel) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, i in zip(axes, sel):
            draw_area(ax, i)
        for ax in axes[len(sel):]:
            ax.set_visible(False)
        fig.suptitle(f"Problem GPS areas, page {page + 1}/{n_pages}", fontsize=10)
        plt.tight_layout()

        fig.savefig(os.path.join(save_path, f"page_{page:02d}.png"), dpi=dpi)
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

    if pdf is not None:
        pdf.close()

    print(f"Saved {len(df_bad)} problem GPS areas "
          f"({(df_bad['status'] == 'no_match').sum()} no_match, "
          f"{(df_bad['status'] == 'multi').sum()} multi) "
          f"across {n_pages} page(s) → {save_path}")
    return df_bad