import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial import KDTree

def match_labels_to_clusters(csv_path, df_clusters, utm_zone=12, max_distance=3.0, eps=2.0,
    max_radius=2.0, graph_save_path="/home/hcr64/Pinyon-Detection/images/"):
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
        eps & max_distance, dbl: Not used for any calculations, just to include on graphs for reference.

    Returns:
        df_clusters: The simple df returned from clusters_to_dataframe, with added columns for labeled tree type.
        
    Requirements:
        numpy, pandas, open3d, matplotlib.pylot, pyproj, scipy.spatial
    """

    df_labels = pd.read_csv(csv_path)

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

    # plot overlap
    plt.figure(figsize=(10, 10))
    colors = {"pinyon": "red", "juniper": "blue", "ponderosa": "orange"}
    for species in ["pinyon", "juniper", "ponderosa"]:
        mask = df_labels["Name"] == species
        plt.scatter(csv_coords[mask, 0], csv_coords[mask, 1],
                    c=colors[species], label=species, s=20)

    print("Num Graph Points:", len(csv_coords) )

    # generate a title for the graph
    graph_title = f"EPS:{eps}-MAX_RADIUS:{max_radius}"

    # make the plot of GPS coords and clsuter locations
    plt.scatter(cluster_coords[:,0], cluster_coords[:,1], c="green", label="clusters", s=1, alpha=0.3)
    plt.legend()
    plt.xlabel("Easting")
    plt.ylabel("Northing")

    plt.title(graph_title)

    # save it to the desired path
    plt.savefig(graph_save_path + graph_title + ".png")
    print(f"Saved {graph_title}.png")

    # match labels to clusters
    tree = KDTree(csv_coords)
    distances, indices = tree.query(cluster_coords)

    df_clusters["Name"]           = df_labels["Name"].iloc[indices].values
    df_clusters["label_distance"] = distances
    df_clusters.loc[df_clusters["label_distance"] > max_distance, "Name"] = "unknown"

    return df_clusters