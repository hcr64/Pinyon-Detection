import open3d as o3d
import os

def save_clusters(clusters, save_path):
    """ 
    Desc:
        Saves the given array of pointcloud clusters to the given path.
        
    Args:
        clusters, array of poincloud (.ply) clusters: An array of pointcloud (.ply) clusters. 
        save_path, str: Location to save the clusters.

    Returns:
        Void
        
    Requirements:
        open3d, os
    """

    # delete everything in the folder before saving
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))
    else:
        os.makedirs(save_path)  # create folder if it doesn't exist

    # save the clusters
    for index in range(len(clusters)):
        o3d.io.write_point_cloud(f"{save_path}cluster{index}.ply", clusters[index])
        # print(f"Cluster {index} saved.")

    print(f"All {len(clusters)} clusters saved to {save_path}")

def load_clusters( save_path ):
    # the array of clusters to return 
    clusters = []

    # for each saved cluster file in the file
    for file in os.listdir( save_path ):
        # make sure a pointcloud is being read
        if file.endswith(".ply"):

            # get the pointcloud object
            pcd = o3d.io.read_point_cloud( os.path.join( save_path, file ) )
            
            # add it to the array
            clusters.append( pcd )

    # return the cluster array
    return clusters
