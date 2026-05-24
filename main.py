# include other folders
print("Loading packages...")

# just for printing the start time
from datetime import datetime

# import libraries
import numpy as np
import pandas as pd
import laspy
import open3d as o3d
import os
import argparse
import csv

from scipy.spatial import KDTree

# include the constants folder
from constants import * 

# include my custom functions
from functions.las_folder_to_pointcloud import las_folder_to_pointcloud

# point cloud cleaning
from functions.clean_up_pointcloud import clean_up_pointcloud
from functions.build_chm import normalize_heights_by_ground

# clustering
from functions.cluster_pointcloud import cluster_pointcloud

# getting stats/labels
from functions.get_pointcloud_stats import clusters_to_dataframe
from functions.match_labels_to_clusters import match_labels_to_clusters

# working with clusters
from functions.save_clusters import save_clusters
from functions.save_clusters import load_clusters
from functions.split_large_clusters import split_large_clusters
from functions.filter_clusters import filter_cluster

# model training
from functions.get_deep_cluster_features import make_deep_dataframe
from functions.train_tree_classifier import train_tree_classifier

print("Sucessfully loaded all packages.")

# the main function
def main():

	# get command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--eps",            type=float, required=True)
	parser.add_argument("--green_threshold",type=float, required=True)
	parser.add_argument("--max_radius",     type=float, required=True)
	parser.add_argument("--max_distance",     type=float, required=True)
	parser.add_argument("--min_points",     type=int, required=True)
	parser.add_argument("--voxel_size", type=float, required=True)

	# for files and such, also a command line arg
	parser.add_argument("--trial_name",     type=str, required=True)

	# get list of command line args
	args = parser.parse_args()

	# assign arguments to variables
	EPS             = args.eps
	GREEN_THRESHOLD = args.green_threshold
	MAX_RADIUS      = args.max_radius
	MAX_DISTANCE    = args.max_distance
	MIN_POINTS		= args.min_points
	VOXEL_SIZE      = args.voxel_size

	# for saving/reading files off location. 
	TRIAL_NAME		= args.trial_name

	NORMALIZE_HEIGHTS=True


	# print general settings
	print()
	print()
	print("#### SETTINGS ####")
	print("EPS:", EPS)
	print("MIN_POINTS:", MIN_POINTS)
	print("GREEN_THRESHOLD:", GREEN_THRESHOLD)
	print("TRIAL_NAME:", TRIAL_NAME)

	# print the start time
	print("Program Start Time: " + datetime.now().strftime("%H:%M:%S"))
	print()
	print()

	# variables

	# the location of where the cleaned pcd should be
	CLEANED_PCD_PATH = f"{PATHS['Pointclouds'].format( TRIAL_NAME )}/cleaned_pcd.ply"
	RAW_PCD_PATH = f"{PATHS['Pointclouds'].format( TRIAL_NAME )}/raw_pcd.ply"

	# read in the point cloud from .las files.
	# The cleaned pointcloud can be saved, if so this does not need to be repeated, can be time consuming, ~10 minutes. 
	# Uses PATHS, DOWNSIZE, VOXEL_SIZE
	if STEPS['Load_Pointcloud']:

		# get the point cloud from the las files in the drive
		print('Loading point cloud...')
		point_cloud = las_folder_to_pointcloud( 
			PATHS['Data'].format( TRIAL_NAME ), 
			silent=SILENT, 
			downsize_pcd=DOWNSIZE,
			v_size=VOXEL_SIZE )

		print('Point cloud successfully loaded.\n')
		print()

		# save the pointcloud to a folder 
		print(f"Saving raw point cloud... ( to { RAW_PCD_PATH }")

		# make sure the dir exists first
		if not os.path.exists( PATHS['Pointclouds'].format( TRIAL_NAME ) ):
			# if it does not, create it
			os.makedirs( PATHS['Pointclouds'].format( TRIAL_NAME ) )

		# now load it in
		o3d.io.write_point_cloud( f"{ RAW_PCD_PATH }", point_cloud)
		print('Raw point cloud successfully saved.\n')
		print()

	# load the pointcloud in from save
	else:
		# read the saved point cloud in
		print(f"Reading in raw pointcloud... (from {PATHS['Pointclouds'].format( TRIAL_NAME )})")

		# make sure the path exists first
		if os.path.exists( f"{PATHS['Pointclouds'].format( TRIAL_NAME )}" ):
			point_cloud = o3d.io.read_point_cloud( RAW_PCD_PATH )
			print("Raw point cloud read in.")
		
		# if it doesnt,
		else:
			# print error message and quit the program
			print(f"Could not find pointcloud save path ({RAW_PCD_PATH}). Exiting program...")
			return 1

		print()


	# downsize the pointcloud do this regardless of loading it in or not
	print("Downsizing pointcloud...")
	point_cloud = point_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
	print("Done downsizing.")


	# 'Clean' the read in point cloud, doesn't take too long. 
	# Uses PATHS, TRIAL_NAME, GREEN_THRESHOLD
	if STEPS['Clean_Pointcloud']:
		
		# check pcd diagnostics first
		points = np.asarray(point_cloud.points)
		print(f"Point count:  {len(points)}")
		print(f"XYZ min: {points.min(axis=0)}")
		print(f"XYZ max: {points.max(axis=0)}")
		print(f"Any NaN: {np.any(np.isnan(points))}")

		# get CHM features
		print("Normalizing heights by ground surface...")
		point_cloud = normalize_heights_by_ground(point_cloud, resolution=0.5)
		print("Heights normalized.")

		# 'clean up' the point cloud, remove noise, the floor, etc
		print("Cleaning up point cloud...")
		point_cloud = clean_up_pointcloud( point_cloud, green_threshold=GREEN_THRESHOLD )
		print("Point cloud successfully cleaned.\n")
		print()

		# check points again
		print(point_cloud)

		# save the cleaned pointcloud to monsoon
		print("Saving cleaned pointcloud...")

		# if the path does not exist,
		if not os.path.exists( os.path.dirname( PATHS['Pointclouds'].format( TRIAL_NAME ) ) ):
			os.makedirs( os.path.dirname( PATHS['Pointclouds'].format( TRIAL_NAME ) ) )

		# write to the folder
		o3d.io.write_point_cloud( CLEANED_PCD_PATH, point_cloud )
		print("Cleaned pointcloud saved.")
		print()

	# since the pointcloud is not being processed, load it in instead.
	else:
		
		# read the saved point cloud in
		print(f"Reading in cleaned pointcloud... (from { CLEANED_PCD_PATH })")

		# make sure the path exists first
		if os.path.exists( f"{PATHS['Pointclouds'].format( TRIAL_NAME ) }" ):
			point_cloud = o3d.io.read_point_cloud( CLEANED_PCD_PATH )
			print("Cleaned point cloud read in.")
		
		# if it doesnt,
		else:
			# print error message and quit the program
			print(f"Could not find pointcloud save path ({ CLEANED_PCD_PATH }). Exiting program...")
			return 1

		print()


	# cluster the 'cleaned' pointcloud into tree clusters.
	# Sometimes unecesary if clusters are already saved. Also can take ~3 minutues to run, so can be worth it to skip.
	# Uses variables: EPS, MIN_POINTS, MAX_RADIUS
	if STEPS['Make_Clusters']:
		# get clusters from the point cloud
		print("Clustering point cloud...")
		clusters, labels = cluster_pointcloud( point_cloud, eps=EPS, min_points=MIN_POINTS )
		print("Point cloud clustered.")
		print()

		# save the clusters in the clusters folder
		print("Saving clusters...")
		save_clusters( clusters, PATHS['Clusters'].format( TRIAL_NAME ) )
		print("Clusters saved.")
		print()
	
	# If clusters will not be computed, try to load them in instead
	else:
		# Print message that clusters are beaing loaded 
		print(f"Reading in clusters... (from {PATHS['Clusters'].format( TRIAL_NAME )})")

		# make sure the path exists first
		if os.path.exists( PATHS['Clusters'].format( TRIAL_NAME ) ):
			# if it does, load them in
			clusters = load_clusters( PATHS['Clusters'].format( TRIAL_NAME ) )
			print("Clusters read in.")
		
		# if it doesnt exist
		else:
			# print error message and quit the program
			print(f"Could not find clusters save path ({PATHS['Clusters'].format( TRIAL_NAME )}). Exiting program...")
			return 1

		print()
		
	# split the clusters if desired, 
	# no else statement needed here, not too time consuming
	if STEPS['Split_clusters']:
		# split clusters into smaller if too large
		print("Splitting clusters...")
		clusters = split_large_clusters(clusters, min_points=MIN_POINTS, max_radius=MAX_RADIUS)
		print("Clusters split.")
		print()

		# filter out non-tree clusters  <-- add here
		print("Filtering clusters...")
		clusters = [c for c in clusters if filter_cluster(c, min_height=1.0, min_radius=0.3)]
		print(f"{len(clusters)} clusters remaining after filter.")
		print()


	# get the cluster df
	print("Making cluster df...")
	df_clusters = clusters_to_dataframe( clusters )
	print("Cluster df made.")
	print()

	# get the deep data cluster df
	print("Making deep data cluster df...")
	df_deep_clusters = make_deep_dataframe( clusters )
	print("Deep data cluster df made.")
	print()

	# assign lables to clusters
	print("Assigning labels to clusters...")
	df_clusters, score = match_labels_to_clusters( 
		PATHS['Labels'], 
		df_clusters, 
		max_distance=MAX_DISTANCE,
		eps=EPS,
		max_radius=MAX_RADIUS,
		min_points=MIN_POINTS,
		green_threshold=GREEN_THRESHOLD
		 )
	print("Labels assigned.")
	print()
	
	# test the accuracy of assigning GPS points to Clusters
	if STEPS['Cluster_accuracy']:
		
		# what to save to the results csv
		results = {
			"eps":             EPS,
			"green_threshold": GREEN_THRESHOLD,
			"max_radius":      MAX_RADIUS,
			"max_distance":	   MAX_DISTANCE,
			"min_points":	   MIN_POINTS,
			"voxel_size":	   VOXEL_SIZE,
			"matching_score":  score,
			"Split_clusters":  STEPS['Split_clusters'],
			"normalize_heights":NORMALIZE_HEIGHTS,
		}

		# append to shared CSV safely
		results_path = PATHS['GPS_results']
		os.makedirs(os.path.dirname(results_path), exist_ok=True)
		file_exists = os.path.exists(results_path)

		with open(results_path, "a", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=results.keys())
			if not file_exists:
				writer.writeheader()
			writer.writerow(results)

		print(f"Results saved to {results_path}")

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

	# Print time at finish too
	print("Time at Completion: " + datetime.now().strftime("%H:%M:%S"))


# call the main function
main()
