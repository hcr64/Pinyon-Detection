import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pyproj import Transformer
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment


def match_labels_to_clusters(csv_path, df_clusters, utm_zone=12, max_distance=3.0, job_id="",
                            graph_subtitle="No Title", graph_save_path="/home/hcr64/Pinyon-Detection/images/"):
    """ 
    Desc:
        Creates a .csv/spreadsheet of advanced metrics on clusters to train models on. Takes a folder path with all the clusters. 
        Also filters through the labels csv to make labels standardized. Can also make a graph of clusters and GPS labels and saves it.
    Args:
        csv_path str: The path to the geoLabel CSV file
        df_clusters, df: The simple df returned from clusters_to_dataframe.
        utm_zone, int: For coordinate translation.
        max_distance, int: Maximum distance a cluster can be from a label to assign it. In meters, usually not more than 3.0.
        graph_save_path, str: where to save the graph of clusters and GPS labels.

    Returns:
        df_clusters: The simple df returned from clusters_to_dataframe, with added columns for labeled tree type.
        
    Requirements:
        numpy, pandas, open3d, matplotlib.pylot, pyproj, scipy.spatial
    """

    df_labels = pd.read_csv(csv_path)

    # the graph title
    graph_title = f"GPS_Clusters_{job_id}"

    # filter labels
    df_labels = df_labels[~df_labels["Name"].str.lower().str.contains("dead", na=False)]
    df_labels = df_labels[~df_labels["Name"].str.lower().str.contains("point", na=False)]
    df_labels["Name"] = df_labels["Name"].str.split().str[0].str.lower().str.strip()
    df_labels["Name"] = df_labels["Name"].replace({"junioer": "juniper"})
    df_labels = df_labels[df_labels["Name"].isin(["pinyon", "juniper", "ponderosa"])].reset_index(drop=True)
    
    # check how many labels are left
    print("Num Filtered Labels:", len(df_labels) )
    print(df_labels["Name"].value_counts())

    # convert lat/lon to UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26912", always_xy=True)
    easting, northing = transformer.transform(
        df_labels["Longitude"].values,
        df_labels["Latitude"].values
    )
    csv_coords     = np.column_stack((easting, northing))
    cluster_coords = np.array(list(zip(df_clusters["x_pos"], df_clusters["y_pos"])))

    # get a graph of points with too many/none clusters 
    plot_gps_cluster_overlap(csv_coords, 
        cluster_coords, 
        df_labels, 
        max_distance=max_distance, 
        save_path=graph_save_path
        )

    # calculate the score
    score = calculate_matching_score(
        csv_coords, 
        cluster_coords, 
        max_distance=max_distance
        )

    # plot overlap
    plt.figure(figsize=(10, 10))
    colors = {"pinyon": "red", "juniper": "blue", "ponderosa": "orange"}
    for species in ["pinyon", "juniper", "ponderosa"]:
        mask = df_labels["Name"] == species
        plt.scatter(csv_coords[mask, 0], csv_coords[mask, 1],
                    c=colors[species], label=species, s=20)

    print("Num Graph Points:", len(csv_coords) )

    # make the plot of GPS coords and clsuter locations
    plt.scatter(cluster_coords[:,0], cluster_coords[:,1], c="green", label="clusters", s=1, alpha=0.3)
    plt.legend()
    plt.xlabel("Easting")
    plt.ylabel("Northing")

    plt.title( graph_title )
    plt.suptitle( graph_subtitle )

    # if the path does not exist,
    if not os.path.exists( graph_save_path ):
        # create it
        os.makedirs( graph_save_path )

    # save it to the desired path
    plt.savefig(graph_save_path + graph_title + ".png")
    print(f"Saved {graph_title}.png")



    # NEW - one-to-one optimal assignment

    # build cost matrix (distances between every GPS point and every cluster)
    cost_matrix = np.zeros((len(csv_coords), len(cluster_coords)))
    for i, gps in enumerate(csv_coords):
        for j, clust in enumerate(cluster_coords):
            cost_matrix[i, j] = np.linalg.norm(gps - clust)

    # optimal one-to-one assignment
    gps_idx, cluster_idx = linear_sum_assignment(cost_matrix)

    # default everything to unknown
    df_clusters["Name"] = "unknown"
    df_clusters["label_distance"] = np.inf

    # only keep assignments within max_distance
    for g, c in zip(gps_idx, cluster_idx):
        if cost_matrix[g, c] <= max_distance:
            df_clusters.loc[c, "Name"] = df_labels["Name"].iloc[g]
            df_clusters.loc[c, "label_distance"] = cost_matrix[g, c]

    return df_clusters, score

# for calculating cluster / GPS points matching
def calculate_matching_score(csv_coords, cluster_coords, max_distance=5.0):
    """
    Score is based on:
    - GPS points with exactly one nearby cluster = good
    - GPS points with no nearby cluster = bad
    - GPS points with multiple nearby clusters = bad
    """

    tree = KDTree(cluster_coords)
    
    # for each GPS point, find all clusters within max_distance
    results = tree.query_ball_point(csv_coords, r=max_distance)
    
    n_gps         = len(csv_coords)
    perfect       = 0  # exactly 1 cluster nearby
    no_match      = 0  # 0 clusters nearby
    multi_match   = 0  # 2+ clusters nearby

    for matches in results:
        n = len(matches)
        if n == 0:
            no_match += 1
        elif n == 1:
            perfect += 1
        else:
            multi_match += 1

    score = perfect / n_gps  # 0.0 to 1.0

    print(f"GPS points:              {n_gps}")
    print(f"Perfect matches (1:1):   {perfect}  ({100*perfect/n_gps:.1f}%)")
    print(f"No match:                {no_match}  ({100*no_match/n_gps:.1f}%)")
    print(f"Multiple clusters:       {multi_match}  ({100*multi_match/n_gps:.1f}%)")
    print(f"Matching score:          {score:.3f}")

    return score


# supplemental function to make the overlay graph 
def plot_gps_cluster_overlap(csv_coords, cluster_coords, df_labels, max_distance=5.0, save_path="/home/hcr64/Pinyon-Detection/coordinate_overlap.png"):

    tree = KDTree(cluster_coords)
    results = tree.query_ball_point(csv_coords, r=max_distance)

    # categorize each GPS point
    perfect    = []
    no_match   = []
    multi      = []

    for i, matches in enumerate(results):
        n = len(matches)
        if n == 0:
            no_match.append(i)
        elif n == 1:
            perfect.append(i)
        else:
            multi.append(i)

    perfect    = np.array(perfect)
    no_match   = np.array(no_match)
    multi      = np.array(multi)

    fig, ax = plt.subplots(figsize=(12, 12))

    # plot clusters in background
    ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
               c="lightgreen", s=1, alpha=0.3, label="Clusters")

    # plot GPS points colored by match quality
    if len(perfect):
        ax.scatter(csv_coords[perfect, 0], csv_coords[perfect, 1],
                   c="blue", s=40, label="Perfect match (1 cluster)", zorder=3)

    if len(no_match):
        ax.scatter(csv_coords[no_match, 0], csv_coords[no_match, 1],
                   c="red", s=60, marker="x", label="No cluster nearby", zorder=3, linewidths=2)

    if len(multi):
        ax.scatter(csv_coords[multi, 0], csv_coords[multi, 1],
                   c="orange", s=60, marker="^", label=f"Multiple clusters nearby", zorder=3)

    # annotate with species name
    for i, (x, y) in enumerate(csv_coords):
        ax.annotate(df_labels["Name"].iloc[i], (x, y),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title(f"GPS Points vs Clusters (max_distance={max_distance}m)\n"
                 f"Perfect: {len(perfect)}  |  No match: {len(no_match)}  |  Multiple: {len(multi)}")

    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path + "coordinate_overlap.png", dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")