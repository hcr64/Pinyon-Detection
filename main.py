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

# saving clusters
from functions.save_clusters import save_clusters

# model training
from functions.get_deep_cluster_features import make_deep_dataframe
from functions.train_tree_classifier import train_tree_classifier

# include my constants
# import constants
import open3d as o3d

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

	if MAKE_CLUSTERS:
		# get the point cloud from the las files in the drive
		print('Loading point cloud...')
		point_cloud = las_folder_to_pointcloud( 
			DATA_PATH, 
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

		# get clusters from the point cloud
		print("Clustering point cloud...")
		clusters, labels = cluster_pointcloud( point_cloud, eps=EPS, min_points=MIN_POINTS )
		print("Point cloud clustered.")
		print()

		# save the clusters in the clusters folder
		print("Saving clusters...")
		save_clusters( clusters, CLUSTER_PATH )
		print("Clusters saved.")
		print()

	# get the cluster df
	print("Making cluster df...")
	df_clusters = clusters_to_dataframe( CLUSTER_PATH )
	print("Cluster df made.")
	print()

	# get the deep data cluster df
	print("Making deep data cluster df...")
	df_deep_clusters = make_deep_dataframe( CLUSTER_PATH )
	print("Deep data cluster df made.")
	print()

	# clean up the labels' names

	# assign lables to clusters
	print("Assigning labels to clusters...")
	df_clusters = match_labels_to_clusters( LABELS_PATH, df_clusters, max_distance=MAX_DISTANCE )
	print("Labels assigned.")
	print()

	# train the model 
	if TRAIN_MODEL:
		# usage
		# model = train_tree_classifier(df_deep_clusters, df_clusters)
		model, features = train_tree_classifier(df_deep_clusters, df_clusters)

		# predict on all clusters including unlabeled ones
		features = [
			"height", "radius", "n_points",
			"obb_extent_x", "obb_extent_y", "obb_extent_z",
			"eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
			"linearity", "planarity", "sphericity",
			"mean_r", "mean_g", "mean_b",
			"std_r", "std_g", "std_b"
		]

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

		df_deep_clusters["predicted_label"] = model.predict(df_deep_clusters[features])
		print(df_deep_clusters[["file", "predicted_label"]])

	print("Program complete.")

# call the main function
main()
