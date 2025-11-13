# import packages
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import pandas as pd
import laspy
import open3d as o3d
import os

# the folder address
FOLDER = '/content/drive/My Drive/Sunset_Crater_trial/point_cloud/'

# the function itself, 
def las_folder_to_pointcloud( folder_path=FOLDER ):

  # the combined, final point cloud
  combined_pcd = o3d.geometry.PointCloud()

  # get all of the files in the folder
  folder_contents = os.listdir(folder)

  # go through each ifile in the folder
  for file in folder_contents:
    print(file)

    # Load the LAS file
    las = laspy.read(folder + file)

    # Extract XYZ coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # append the new point cloud onto the largeone
    combined_pcd.points = o3d.utility.Vector3dVector(
      np.vstack( ( np.asarray(combined_pcd.points), np.asarray(pcd.points) ) )
      )
    
    print( combined_pcd )
    
