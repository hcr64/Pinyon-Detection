# include other folders
print("Loading packages...")

# import libraries
import numpy as np
import pandas as pd
import laspy
import open3d as o3d
import os
from scipy.spatial import KDTree

# include the constants folder
from constants import * 

# include my custom functions
from functions.las_folder_to_pointcloud import las_folder_to_pointcloud

# point cloud processing
from functions.clean_up_pointcloud import clean_up_pointcloud
from functions.cluster_pointcloud import cluster_pointcloud
from functions.get_pointcloud_stats import clusters_to_dataframe
from functions.match_labels_to_clusters import match_labels_to_clusters

# working with clusters
from functions.save_clusters import save_clusters
from functions.split_large_clusters import split_large_clusters

# model training
from functions.get_deep_cluster_features import make_deep_dataframe
from functions.train_tree_classifier import train_tree_classifier

print("Sucessfully loaded all packages.")

# the main function
def main():
	# print general settings
	print("#### SETTINGS ####")
	print("EPS:", EPS)
	print("MIN_POINTS:", MIN_POINTS)
	print("GREEN_THRESHOLD:", GREEN_THRESHOLD)
	print()
	print()

	# read in the point cloud from .las files.
	# The cleaned pointcloud can be saved, if so this does not need to be repeated, can be time consuming, ~10 minutes. 
	# If GREEN_THRESHOLD or VOXEL_SIZE get changed, this needs to get ran again. Clustering variables do not matter here.
	if STEPS['Clean_Pointcloud']:
		# get the point cloud from the las files in the drive
		print('Loading point cloud...')
		point_cloud = las_folder_to_pointcloud( 
			PATHS['Data'], 
			silent=SILENT, 
			downsize_pcd=DOWNSIZE,
			v_size=VOXEL_SIZE )

		print('Point cloud successfully loaded.\n')
		print()

		# check the points
		print(point_cloud)

		# 'clean up' the point cloud, remove noise, the floor, etc
		print("Cleaning up point cloud...")
		point_cloud = clean_up_pointcloud( point_cloud, green_threshold=GREEN_THRESHOLD )
		print("Point cloud successfully cleaned.\n")
		print()

		# check points again
		print(point_cloud)

		# save the cleaned pointcloud to monsoon
		print("Saving cleaned pointcloud...")
		o3d.io.write_point_cloud( PATHS['Cleaned_pcd'], point_cloud )
		print("Cleaned pointcloud saved.")
		print()

	# since the pointcloud is not being processed, load it in instead.
	else:
		# read the saved point cloud in
		print(f"Reading in cleaned pointcloud... (from {PATHS['Cleaned_pcd']})")
		point_cloud = o3d.io.read_point_cloud( PATHS['Cleaned_pcd'] )
		print("Cleaned point cloud read in.")
		print()



	# cluster the 'cleaned' pointcloud into tree clusters.
	# Sometimes unecesary if clusters are already saved. Also can take ~3 minutues to run, so can be worth it to skip.
	if STEPS['Make_Clusters']:
		# get clusters from the point cloud
		print("Clustering point cloud...")
		clusters, labels = cluster_pointcloud( point_cloud, eps=EPS, min_points=MIN_POINTS )
		print("Point cloud clustered.")
		print()

		# split clusters into smaller if too large
		print("Splitting clusters...")
		clusters = split_large_clusters(clusters, max_radius=MAX_RADIUS)
		print("Clusters split.")
		print()

		# save the clusters in the clusters folder
		print("Saving clusters...")
		save_clusters( clusters, PATHS['Clusters'] )
		print("Clusters saved.")
		print()
	
	# since the clusters are not being processed, read them in instead, NOT CURRENTLY NEEDED 
	# Functions after this access clusters from their path, so uneeded so far. 
	else:
		pass

	# get the cluster df
	print("Making cluster df...")
	df_clusters = clusters_to_dataframe( PATHS['Clusters'] )
	print("Cluster df made.")
	print()

	# get the deep data cluster df
	print("Making deep data cluster df...")
	df_deep_clusters = make_deep_dataframe( PATHS['Clusters'] )
	print("Deep data cluster df made.")
	print()

	# assign lables to clusters
	print("Assigning labels to clusters...")
	df_clusters = match_labels_to_clusters( 
		PATHS['Labels'], 
		df_clusters, 
		max_distance=MAX_DISTANCE,
		eps=EPS,
		max_radius=MAX_RADIUS
		 )
	print("Labels assigned.")
	print()

	# training the model, never too time consuming, generally less than a minute or two.
	if STEPS['Train_Model']:
		# model = train_tree_classifier(df_deep_clusters, df_clusters)
		model, features = train_tree_classifier(df_deep_clusters, df_clusters)

		# only label as pinyon if model is >80% confident
		probs = model.predict_proba(df_deep_clusters[features])
		pinyon_idx = list(model.classes_).index("pinyon")

		df_deep_clusters["pinyon_confidence"] = probs[:, pinyon_idx]
		df_deep_clusters["predicted_label"] = model.predict(df_deep_clusters[features])

		# filter to only high confidence pinyons
		confirmed_pinyons = df_deep_clusters[
			(df_deep_clusters["predicted_label"] == "pinyon") &
			(df_deep_clusters["pinyon_confidence"] > 0.80)
		]

		print(f"High confidence pinyons: {len(confirmed_pinyons)}")

		print(df_deep_clusters[["file", "predicted_label"]])

	print("Program complete.")

# call the main function
main()
