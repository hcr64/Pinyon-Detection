import open3d as o3d
import os

def save_clusters(clusters, save_path):
    """
    Save a list of point cloud clusters to disk as individual .ply files.
 
    Clears the destination folder before writing so it always reflects the
    current cluster list. Files are named cluster0.ply, cluster1.ply, …
 
    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to save.
        save_path (str): Directory to write .ply files into. Created
            automatically if it does not exist.
 
    Returns:
        None
 
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
    """
    Load all .ply cluster files from a directory into a list of PointClouds.
 
    Args:
        save_path (str): Directory containing .ply cluster files written by
            save_clusters().
 
    Returns:
        list of o3d.geometry.PointCloud: One entry per .ply file found.
 
    Requirements:
        open3d, os
    """
    
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
