import open3d as o3d
import os

def save_clusters(clusters, save_path):

    # delete everything in the folder before saving
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))
    else:
        os.makedirs(save_path)  # create folder if it doesn't exist

    # save the clusters
    for index in range(len(clusters)):
        o3d.io.write_point_cloud(f"{save_path}cluster{index}.ply", clusters[index])
        print(f"Cluster {index} saved.")

    print(f"All {len(clusters)} clusters saved to {save_path}")
