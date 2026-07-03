import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pyproj import Transformer
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment


def match_labels_to_clusters(csv_path, df_clusters, utm_zone=12, max_distance=3.0,
                              gps_sigma=3.0, job_id="",
                              graph_subtitle="No Title",
                              graph_save_path="/home/hcr64/Pinyon-Detection/images/",
                              multi_match_save_path=None,
                              clusters=None,
                              csv_path_2=None):
    """
    Match GPS species labels to detected clusters via optimal one-to-one assignment,
    weighted by a Gaussian GPS error model.

    Loads GPS labels from one or two CSVs, converts coordinates from WGS84
    lat/lon to UTM, and assigns each GPS label to its nearest cluster centroid
    using scipy.optimize.linear_sum_assignment (the Hungarian algorithm).

    Cost matrix uses a Gaussian error model rather than raw Euclidean distance:

        cost(i, j) = 1 - exp( -d² / (2 * gps_sigma²) )

    where d is the Euclidean distance between GPS point i and cluster j, and
    gps_sigma is the assumed 1-sigma GPS horizontal error in metres. This means:
        - A match at d=0 has cost 0 (perfect)
        - A match at d=gps_sigma has cost ~0.39 (plausible)
        - A match at d=2*gps_sigma has cost ~0.86 (unlikely)
        - Assignments further than max_distance are still discarded as before

    The Gaussian cost makes the Hungarian algorithm prefer tight matches much
    more strongly than the raw-distance version, where a 1m match and a 2m
    match have costs 1 and 2 — a 2× difference. Under the Gaussian with
    sigma=3m, those same matches have costs 0.055 and 0.200 — a 3.6× difference
    that more accurately reflects GPS uncertainty.

    GPS labels are normalised before matching: dead trees and survey points are
    excluded, species names are standardised to pinyon / juniper / ponderosa,
    and labels outside the point cloud bounding box are dropped. Rows from
    csv_path_2 (if provided) are merged and deduplicated with the primary CSV.

    Saves two diagnostic plots to graph_save_path:
        coordinate_overlap.png — GPS points coloured by match quality
        GPS_Clusters_<job_id>.png — species scatter overlaid on all cluster
            centroids

    Args:
        csv_path (str): Path to the primary GPS label CSV. Must have columns
            "Name", "Longitude", "Latitude".
        df_clusters (pd.DataFrame): Cluster geometry DataFrame from
            clusters_to_dataframe(). Must have columns "x_pos" and "y_pos"
            (UTM, metres).
        utm_zone (int): UTM zone for coordinate conversion. Default 12
            (Arizona / Sunset Crater, EPSG:26912).
        max_distance (float): Maximum GPS-to-cluster distance in metres for
            a label assignment to be accepted. Assignments beyond this are
            discarded even if they are the best available match. Default 3.0.
        gps_sigma (float): Assumed 1-sigma horizontal GPS error in metres.
            Controls the shape of the Gaussian cost function — smaller values
            penalise distant matches more aggressively. 3.0m is appropriate
            for a typical consumer/survey GPS under tree canopy. Default 3.0.
        job_id (str): Appended to the graph filename for traceability across
            SLURM sweep runs. Default "".
        graph_subtitle (str): Subtitle shown on the species scatter plot.
            Default "No Title".
        graph_save_path (str): Directory for output plots.
        csv_path_2 (str | None): Optional second GPS label CSV. Default None.

    Returns:
        df_clusters (pd.DataFrame): Input DataFrame with columns added:
            "Name"           — assigned species string, or "unknown"
            "label_distance" — distance in metres to the matched GPS point,
                               or inf for unmatched clusters
        score (float): Matching score — fraction of GPS labels with exactly
            one cluster within max_distance. Range [0, 1].

    Requirements:
        numpy, pandas, matplotlib, pyproj, scipy.spatial.KDTree,
        scipy.optimize.linear_sum_assignment
    """

    graph_title = f"GPS_Clusters_{job_id}"

    # ── load and normalize labels ─────────────────────────────────────────────

    def load_and_filter(path):
        df = pd.read_csv(path)
        df = df[~df["Name"].str.lower().str.contains("dead", na=False)]
        df = df[~df["Name"].str.lower().str.contains("point", na=False)]
        df["Name"] = df["Name"].str.split().str[0].str.lower().str.strip()
        df["Name"] = df["Name"].replace({"junioer": "juniper"})
        df = df[df["Name"].isin(["pinyon", "juniper", "ponderosa"])].reset_index(drop=True)
        return df

    df_labels = load_and_filter(csv_path)

    if csv_path_2 is not None:
        df_labels_2 = load_and_filter(csv_path_2)
        df_labels = pd.concat([df_labels, df_labels_2], ignore_index=True)
        print(f"Merged labels from both CSVs.")

    print("Num Filtered Labels:", len(df_labels))
    print(df_labels["Name"].value_counts())

    # ── convert lat/lon to UTM ────────────────────────────────────────────────

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26912", always_xy=True)
    easting, northing = transformer.transform(
        df_labels["Longitude"].values,
        df_labels["Latitude"].values
    )
    csv_coords     = np.column_stack((easting, northing))
    cluster_coords = np.array(list(zip(df_clusters["x_pos"], df_clusters["y_pos"])))

    # ── drop GPS labels outside the scanned area ──────────────────────────────

    coverage_radius = 10.0
    cluster_tree = KDTree(cluster_coords)
    distances_to_nearest, _ = cluster_tree.query(csv_coords)
    in_coverage = distances_to_nearest <= coverage_radius

    n_dropped = (~in_coverage).sum()
    if n_dropped > 0:
        print(f"Coverage filter: dropped {n_dropped} GPS labels with no cluster "
              f"within {coverage_radius}m (were outside scan area)")

    df_labels  = df_labels[in_coverage].reset_index(drop=True)
    csv_coords = csv_coords[in_coverage]

    # ── diagnostic overlap plot ───────────────────────────────────────────────

    plot_gps_cluster_overlap(csv_coords,
        cluster_coords,
        df_labels,
        max_distance=max_distance,
        save_path=graph_save_path
        )

    # ── matching score ────────────────────────────────────────────────────────

    score = calculate_matching_score(
        csv_coords,
        cluster_coords,
        max_distance=max_distance
        )

    # ── save clusters behind ambiguous "multiple clusters" GPS matches ───────
    if clusters is not None and multi_match_save_path is not None:
        from functions.labeling.save_multimatch_clusters import save_multimatch_clusters
        save_multimatch_clusters(
            csv_coords, cluster_coords, clusters, df_labels, df_clusters,
            save_path=multi_match_save_path, max_distance=max_distance
        )

    # ── species scatter plot ──────────────────────────────────────────────────

    plt.figure(figsize=(10, 10))
    colors = {"pinyon": "red", "juniper": "blue", "ponderosa": "orange"}
    for species in ["pinyon", "juniper", "ponderosa"]:
        mask = df_labels["Name"] == species
        plt.scatter(csv_coords[mask, 0], csv_coords[mask, 1],
                    c=colors[species], label=species, s=20)

    print("Num Graph Points:", len(csv_coords))

    plt.scatter(cluster_coords[:,0], cluster_coords[:,1],
                c="green", label="clusters", s=1, alpha=0.3)
    plt.legend()
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.title(graph_title)
    plt.suptitle(graph_subtitle)

    if not os.path.exists(graph_save_path):
        os.makedirs(graph_save_path)

    plt.savefig(graph_save_path + graph_title + ".png")
    print(f"Saved {graph_title}.png")

    # ── Gaussian-weighted cost matrix ─────────────────────────────────────────
    # Raw Euclidean distance treats a 1m and 4m match as costs 1 and 4 — a
    # 4× difference. Under a Gaussian with sigma=gps_sigma, the same matches
    # have costs that reflect the actual likelihood ratio given GPS error.
    # cost = 1 - exp(-d² / (2σ²)) so:
    #   d = 0         → cost = 0.000  (certain match)
    #   d = 1*sigma   → cost = 0.393  (plausible)
    #   d = 2*sigma   → cost = 0.865  (unlikely)
    #   d = 3*sigma   → cost = 0.989  (very unlikely)
    # The Hungarian algorithm then strongly prefers tight matches over loose
    # ones, which raw distance does not do proportionally enough.

    print(f"\nBuilding Gaussian cost matrix  (gps_sigma={gps_sigma}m, "
          f"max_distance={max_distance}m)...")

    n_gps      = len(csv_coords)
    n_clusters = len(cluster_coords)
    cost_matrix = np.zeros((n_gps, n_clusters))

    for i, gps in enumerate(csv_coords):
        d_sq = np.sum((cluster_coords - gps) ** 2, axis=1)   # squared distances
        cost_matrix[i] = 1.0 - np.exp(-d_sq / (2.0 * gps_sigma ** 2))

    # ── one-to-one optimal assignment ─────────────────────────────────────────

    gps_idx, cluster_idx = linear_sum_assignment(cost_matrix)

    df_clusters["Name"]           = "unknown"
    df_clusters["label_distance"] = np.inf

    n_assigned = 0
    n_rejected = 0

    for g, c in zip(gps_idx, cluster_idx):
        # compute actual Euclidean distance for the max_distance gate
        actual_dist = np.linalg.norm(csv_coords[g] - cluster_coords[c])
        if actual_dist <= max_distance:
            df_clusters.loc[c, "Name"]           = df_labels["Name"].iloc[g]
            df_clusters.loc[c, "label_distance"]  = actual_dist
            n_assigned += 1
        else:
            n_rejected += 1

    print(f"Assignments accepted (≤ {max_distance}m): {n_assigned}")
    print(f"Assignments rejected (> {max_distance}m): {n_rejected}")

    return df_clusters, score


# ── scoring ───────────────────────────────────────────────────────────────────

def calculate_matching_score(csv_coords, cluster_coords, max_distance=5.0):
    """
    Compute the fraction of GPS labels with exactly one nearby cluster.

    For each GPS point, counts clusters within max_distance using a KDTree
    ball query. The score is perfect_matches / total_gps_points.

    Scoring categories:
        Perfect (score contribution): exactly one cluster within max_distance
        No match (penalised):         zero clusters within max_distance
        Multiple (penalised):         two or more clusters within max_distance

    Args:
        csv_coords (np.ndarray): (M, 2) UTM XY coordinates of GPS labels.
        cluster_coords (np.ndarray): (N, 2) UTM XY cluster centroids.
        max_distance (float): Search radius in metres. Default 5.0.

    Returns:
        float: Matching score in [0, 1].

    Requirements:
        numpy, scipy.spatial.KDTree
    """

    tree = KDTree(cluster_coords)
    results = tree.query_ball_point(csv_coords, r=max_distance)

    n_gps       = len(csv_coords)
    perfect     = 0
    no_match    = 0
    multi_match = 0

    for matches in results:
        n = len(matches)
        if n == 0:
            no_match += 1
        elif n == 1:
            perfect += 1
        else:
            multi_match += 1

    score = perfect / n_gps

    print(f"GPS points:              {n_gps}")
    print(f"Perfect matches (1:1):   {perfect}  ({100*perfect/n_gps:.1f}%)")
    print(f"No match:                {no_match}  ({100*no_match/n_gps:.1f}%)")
    print(f"Multiple clusters:       {multi_match}  ({100*multi_match/n_gps:.1f}%)")
    print(f"Matching score:          {score:.3f}")

    return score


# ── overlap diagnostic plot ───────────────────────────────────────────────────

def plot_gps_cluster_overlap(csv_coords, cluster_coords, df_labels,
                             max_distance=5.0,
                             save_path="/home/hcr64/Pinyon-Detection/coordinate_overlap.png"):
    """
    Save a diagnostic scatter plot of GPS points vs cluster centroids,
    coloured by match quality (perfect / no match / multiple nearby).

    Args:
        csv_coords (np.ndarray): (M, 2) UTM XY GPS coordinates.
        cluster_coords (np.ndarray): (N, 2) UTM XY cluster centroids.
        df_labels (pd.DataFrame): Label DataFrame with "Name" column,
            aligned to csv_coords by index.
        max_distance (float): Radius used to classify match quality.
            Default 5.0.
        save_path (str): Directory to save the PNG into. Default path is
            the Monsoon image folder.

    Returns:
        None

    Requirements:
        numpy, matplotlib, scipy.spatial.KDTree
    """

    tree    = KDTree(cluster_coords)
    results = tree.query_ball_point(csv_coords, r=max_distance)

    perfect  = []
    no_match = []
    multi    = []

    for i, matches in enumerate(results):
        n = len(matches)
        if n == 0:
            no_match.append(i)
        elif n == 1:
            perfect.append(i)
        else:
            multi.append(i)

    perfect  = np.array(perfect)
    no_match = np.array(no_match)
    multi    = np.array(multi)

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
               c="lightgreen", s=1, alpha=0.3, label="Clusters")

    if len(perfect):
        ax.scatter(csv_coords[perfect, 0], csv_coords[perfect, 1],
                   c="blue", s=40, label="Perfect match (1 cluster)", zorder=3)

    if len(no_match):
        ax.scatter(csv_coords[no_match, 0], csv_coords[no_match, 1],
                   c="red", s=60, marker="x", label="No cluster nearby",
                   zorder=3, linewidths=2)

    if len(multi):
        ax.scatter(csv_coords[multi, 0], csv_coords[multi, 1],
                   c="orange", s=60, marker="^",
                   label="Multiple clusters nearby", zorder=3)

    for i, (x, y) in enumerate(csv_coords):
        ax.annotate(df_labels["Name"].iloc[i], (x, y),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title(f"GPS Points vs Clusters (max_distance={max_distance}m)\n"
                 f"Perfect: {len(perfect)}  |  No match: {len(no_match)}  "
                 f"|  Multiple: {len(multi)}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path + "coordinate_overlap.png", dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")