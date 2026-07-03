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

# include all my custom functions
from functions import *

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
	parser.add_argument("--min_peak_distance", type=float, required=True)
	parser.add_argument("--k", type=int, required=True)
	parser.add_argument("--min_height",       type=float, required=True)
	parser.add_argument("--search_radius_m",  type=float, required=True)
	# parser.add_argument("--min_density_ratio", type=float, required=True)


	# for files and such, also a command line arg
	parser.add_argument("--trial_name",     type=str, required=True)
	parser.add_argument("--job_id",     type=str, required=True)


	# get list of command line args
	args = parser.parse_args()

	# assign arguments to variables
	EPS             = args.eps
	GREEN_THRESHOLD = args.green_threshold
	MAX_RADIUS      = args.max_radius
	MAX_DISTANCE    = args.max_distance
	MIN_POINTS		= args.min_points
	VOXEL_SIZE      = args.voxel_size
	MIN_PEAK_DISTANCE = args.min_peak_distance
	K = args.k

	MIN_DENSITY_RATIO = 1.5 # args.min_density_ratio
	S_SIGMA = 1.0

	MIN_HEIGHT       = args.min_height
	SEARCH_RADIUS_M  = args.search_radius_m


	# for saving/reading files off location. 
	TRIAL_NAME		= args.trial_name

	# for identifying what graphs come from which jobs
	JOB_ID = args.job_id

	NORMALIZE_HEIGHTS=True

	# get the paths 
	PATHS = get_paths( TRIAL_NAME )
	
	# print general settings / command line args
	print()
	print()
	print("#### SETTINGS ####")
	print("EPS:", EPS)
	print("MIN_POINTS:", MIN_POINTS)
	print("GREEN_THRESHOLD:", GREEN_THRESHOLD)
	print("MAX_DISTANCE:", MAX_DISTANCE)
	print("MAX_RADIUS:", MAX_RADIUS)
	print("VOXEL_SIZE:", VOXEL_SIZE)
	print("MIN_PEAK_DISTANCE:", MIN_PEAK_DISTANCE)
	print("K:", K)	
	print("MIN_DESNITY_RATIO:", MIN_DENSITY_RATIO)

	# set k to min points
	K = MIN_POINTS

	# non-calculation vars
	print("TRIAL_NAME:", TRIAL_NAME)
	print("JOB_ID:", JOB_ID)


	# print the start time
	print("Program Start Time: " + datetime.now().strftime("%H:%M:%S"))
	print()
	print()


	# read in the point cloud from .las files.
	# The cleaned pointcloud can be saved, if so this does not need to be repeated, can be time consuming, ~10 minutes. 
	# Uses PATHS, DOWNSIZE, VOXEL_SIZE
	if STEPS['Load_Pointcloud']:

		# get the point cloud from the las files in the drive
		print('Loading point cloud...')
		point_cloud = las_folder_to_pointcloud( 
			PATHS['Data'], 
			silent=SILENT, 
			downsize_pcd=DOWNSIZE,
			v_size=VOXEL_SIZE )

		print('Point cloud successfully loaded.\n')
		print()

		# save the pointcloud to a folder 
		print(f"Saving raw point cloud... ( to { PATHS['Raw_pcd'] }")

		# make sure the dir exists first
		if not os.path.exists( os.path.dirname( PATHS['Raw_pcd'] ) ):
			# if it does not, create it
			os.makedirs( os.path.dirname( PATHS['Raw_pcd'] ) )

		# now load it in
		o3d.io.write_point_cloud( f"{ PATHS['Raw_pcd'] }", point_cloud)
		print('Raw point cloud successfully saved.\n')
		print()


	# load the pointcloud in from save, if will be cleaned or used in the CHM, or used in clustering, 
	# otherwise, the cleaned pointcloud is being read in. Just read it in, its genrally used
	else:

		# read the saved point cloud in
		print(f"Reading in raw pointcloud... (from {PATHS['Raw_pcd']})")

		# make sure the path exists first
		if os.path.exists( os.path.dirname( PATHS['Raw_pcd'] ) ):
			point_cloud = o3d.io.read_point_cloud( PATHS['Raw_pcd'] )
			print(f"Raw point cloud read in. ({PATHS['Raw_pcd']})")
		
		# if it doesnt,
		else:
			# print error message and quit the program
			print(f"Could not find pointcloud save path ({PATHS['Raw_pcd']}). Exiting program...")
			return 1

		print()



	# only do this if the CHM is being made, or if the pointcloud will be cleaned. 
	# Otherwise this whill have no effect on the program
	if STEPS['Make_CHM'] or STEPS['Clean_Pointcloud']:
		# downsize the pointcloud do this regardless of loading it in or not
		print("Downsizing pointcloud...")
		point_cloud = point_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
		print("Done downsizing.")
		print()



	# make the cfm model if desired
	if STEPS['Make_CHM']:
		print('Making CHM...')
		chm, transform, crs = build_chm(
			point_cloud,
			resolution=0.5,
			ground_percentile=5,
			save_path=PATHS['CHM']
			)
		print(f"CHM complete. (Saved to {PATHS['CHM']})")
		print()

	# otherwise, load it in from disk,
	# always do one or another, no other condition needed
	else:
		# check if the path is legit
		if os.path.exists( PATHS['CHM'] ):
			print(f"Loading in CHM... (from { PATHS['CHM'] })")
			chm, transform, crs = load_chm( PATHS['CHM'] )
			
		else:
			# print an error message and quit
			print(f"Cannot find CHM save path. ({ PATHS['CHM'] }) Exiting program...")
			return 1

		print('CMF loaded in successfully.')

	# only do this if clustering the pointcloud after, otherwise there is no point and it wastes time
	if STEPS['Make_Clusters']:
		# get the peaks from the pointcloud
		# do either way
		print("Finding peaks in CHM...")
		peak_coords, peak_heights = find_chm_peaks(
			chm, 
			transform, 
			min_height=MIN_HEIGHT, 
			search_radius_m=SEARCH_RADIUS_M,
			smooth_sigma=S_SIGMA)
		print()


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
		if NORMALIZE_HEIGHTS:
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
		if not os.path.exists( os.path.dirname( PATHS['Cleaned_pcd'] ) ):
			os.makedirs( os.path.dirname( PATHS['Cleaned_pcd'] ) )

		# write to the folder
		o3d.io.write_point_cloud( PATHS['Cleaned_pcd'], point_cloud )
		print("Cleaned pointcloud saved.")
		print()


	# since the pointcloud is not being processed, load it in instead.
	# only do so if the clusters will be made. Otherwise the cleaned pcd will not be used
	elif STEPS['Make_Clusters'] and False:
		
		# read the saved point cloud in
		print(f"Reading in cleaned pointcloud... (from { PATHS['Cleaned_pcd'] })")

		# make sure the path exists first
		if os.path.exists( os.path.dirname(PATHS['Cleaned_pcd']) ):
			point_cloud = o3d.io.read_point_cloud( PATHS['Cleaned_pcd'] )
			print("Cleaned point cloud read in.")
		
		# if it doesnt,
		else:
			# print error message and quit the program
			print(f"Could not find pointcloud save path ({ PATHS['Cleaned_pcd'] }). Exiting program...")
			return 1

		print()


	# if none of these cases hit, print message saying the pointcloud will not be cleaned
	else:
		print("Unprocessed pointcloud being used. Pointcloud is not being 'cleaned.'")
		print()



	# cluster the 'cleaned' pointcloud into tree clusters.
	# Sometimes unecesary if clusters are already saved. Also can take ~3 minutues to run, so can be worth it to skip.
	# Uses variables: EPS, MIN_POINTS, MAX_RADIUS
	if STEPS['Make_Clusters']:
		# get clusters from the point cloud
		print("Clustering point cloud...")

		# clusters, labels = cluster_pointcloud( point_cloud, eps=EPS, min_points=MIN_POINTS )
		clusters = cluster_by_chm_peaks(
			point_cloud,
			peak_coords,
			chm=chm,           # ← new
			transform=transform, # ← new
			crown_radius=MAX_RADIUS,
			min_points=MIN_POINTS
		)
		print("Point cloud clustered.")
		print()

		# after clustering, filter points by height above ground
		MIN_POINT_HEIGHT = 1.5  # metres above local ground
		clusters = [
			c for c in clusters 
			if np.asarray(c.points)[:, 2].max() - np.asarray(c.points)[:, 2].min() > MIN_POINT_HEIGHT
		]

		# filter out non-vegetation clusters by checking the greenness of their
		# most-green points — removes rock patches and bare-ground returns that
		# passed the height filter but have weak greenness throughout
		clusters = filter_clusters_by_green_crown(
			clusters,
			top_fraction=0.20,   # inspect the greenest 20% of points
			min_exg=0.05,        # mean ExG of that slice must clear this bar
		)

		# save the clusters in the clusters folder
		print("Saving clusters...")
		save_clusters( clusters, PATHS['Clusters'] )
		print("Clusters saved.")
		print()

		# only do this if the pointcloud is NOT being 'cleaned' first, 
		if not STEPS['Clean_Pointcloud']:
			# get rid of the floor of the clusters
			print("Stripping ground from clusters...")
			clusters = strip_ground_from_clusters(clusters, ground_percentile=10, min_height_above_ground=0.5)
			print("Ground stripped.")

		# split clusters into smaller if too large
		print("Splitting clusters...")
		clusters = split_large_clusters(clusters, 
			min_points=MIN_POINTS, 
			max_radius=MAX_RADIUS, 
			min_peak_distance=MIN_PEAK_DISTANCE,
			k=K,
			min_density_ratio=MIN_DENSITY_RATIO
			)
		print("Clusters split.")
		print()

		# filter out non-tree clusters  <-- add here
		print("Filtering clusters...")
		before = len(clusters)
		clusters = [c for c in clusters if filter_cluster(c, min_height=1.0, min_radius=0.3)]
		print(f"Filtered {before - len(clusters)} non-tree clusters, {len(clusters)} remaining")
		print()

	
	# If clusters will not be computed, try to load them in instead
	else:
		# Print message that clusters are beaing loaded 
		print(f"Reading in clusters... (from {PATHS['Clusters']})")

		# make sure the path exists first
		if os.path.exists( PATHS['Clusters'] ):
			# if it does, load them in
			clusters = load_clusters( PATHS['Clusters'] )
			print("Clusters read in.")
		
		# if it doesnt exist
		else:
			# print error message and quit the program
			print(f"Could not find clusters save path ({PATHS['Clusters']}). Exiting program...")
			return 1

		print()
	

		# only do this if the pointcloud is NOT being 'cleaned' first, 
		if not STEPS['Clean_Pointcloud']:
			# get rid of the floor of the clusters
			print("Stripping ground from clusters...")
			clusters = strip_ground_from_clusters(clusters, ground_percentile=10, min_height_above_ground=0.5)
			print("Ground stripped.")



	#------------------------------------------------------------------------
	### GET FEATURES
	#------------------------------------------------------------------------
	# can take several minutes, worth reading in from disk when training models
	# only NEED to do when changing clusters, either plitting or making them
	if STEPS['Make_Clusters']:

		# get the cluster df
		print("Making cluster df...")
		df_clusters = clusters_to_dataframe( clusters, k=K )
		print("Cluster df made.")
		print()

		# get the deep data cluster df
		print("Making deep data cluster df...")
		df_deep_clusters = make_deep_dataframe( clusters )
		df_deep_clusters = engineer_features(df_deep_clusters)
		print("Deep data cluster df made.")
		print()

		# save the dataframes to the disk
		save_dataframes(df_clusters, df_deep_clusters, PATHS['Dataframes'])

		# save the labeled clusters to a folder
		print("saving Labeled clusters...")
		save_labeled_clusters(
			clusters, 
			df_clusters, 
			save_path=PATHS['Labeled_clusters']
			)
		print("Labeled clusters saved.")
		print()
	
	# read in dataframes, labeled clusters are not being used currently
	else:
		print("Reading in feature dataframes...")
		df_clusters, df_deep_clusters = load_dataframes(PATHS['Dataframes'])
		print("Dataframes read in.")
		print()



	# assign lables to clusters
	print("Assigning labels to clusters...")
	df_clusters, score = match_labels_to_clusters( 
		PATHS['Labels'],
		df_clusters, 
		csv_path_2=os.path.dirname( PATHS['Labels'] ) + '/sunsetCraterMay26.csv', 
		max_distance=MAX_DISTANCE,
		graph_save_path = PATHS['Images'],
		gps_sigma=4.0,
		job_id=JOB_ID,
		clusters=clusters,
		multi_match_save_path=PATHS["MM_save_path"],
		graph_subtitle=f"EPS:{EPS}-MR:{MAX_RADIUS}-GT:{GREEN_THRESHOLD}-MPts:{MIN_POINTS}-MD:{MAX_DISTANCE}-K:{K}-MPD:{MIN_PEAK_DISTANCE}"
		)
	print("Labels assigned.")
	print()
	


	# test the accuracy of assigning GPS points to Clusters
	if STEPS['Cluster_accuracy']:
		
		# what to save to the results csv
		results = {
			"eps":               EPS,
			"green_threshold":   GREEN_THRESHOLD,
			"max_radius":        MAX_RADIUS,
			"max_distance":      MAX_DISTANCE,
			"min_points":        MIN_POINTS,
			"voxel_size":        VOXEL_SIZE,
			"min_peak_distance": MIN_PEAK_DISTANCE,
			"k":                 K,
			"min_height":        MIN_HEIGHT,        # add
			"search_radius_m":   SEARCH_RADIUS_M,   # add
			"normalize_heights": NORMALIZE_HEIGHTS,
			"matching_score":    score,
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
		
		plot_feature_separability(
			df_deep_clusters,
			df_clusters,
			save_path=PATHS['Images']
		)
		
		if True:
			model, features = train_tree_classifier(
				df_deep_clusters,
				df_clusters,
				save_confusion_matrix_path=PATHS['Images'] + 'confusion_matrix.png'
			)
		else:
			model, features = run_advanced_classifiers(
				df_deep_clusters,
				df_clusters,
				save_path=PATHS['Images']
			)

			# ── semi-supervised label spreading ──────────────────────────────────────
			print("Running semi-supervised label spreading...")
			df_deep_clusters = run_label_spreading(
				df_deep_clusters,
				df_clusters,
				save_path=PATHS['Images'],
				alpha=0.2,          # how much labeled nodes can drift — keep low
				gamma=0.5,          # RBF neighbourhood tightness
				confidence_threshold=0.80
			)
			print("Label spreading complete.")

		# only label as pinyon if model is >80% confident
		if "prob_pinyon" in df_deep_clusters.columns:
			confirmed_pinyons = df_deep_clusters[
				(df_deep_clusters["predicted_label"] == "pinyon") &
				(df_deep_clusters["prob_pinyon"] > 0.80)
			]
			print(f"High confidence pinyons: {len(confirmed_pinyons)}")

		print(df_deep_clusters[["file", "predicted_label"]])


	print("Program complete.")

	# Print time at finish too
	print("Time at Completion: " + datetime.now().strftime("%H:%M:%S"))


# call the main function
main()
