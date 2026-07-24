"""
run_clustering.py
────────────────────────────────────────────────────────────────────────────────
Point cloud → clusters → GPS-labeled clusters.

Everything expensive lives here: loading .las files, building the CHM,
cleaning/normalizing the point cloud, watershed segmentation, ground
stripping, splitting oversized clusters, feature extraction, and GPS label
matching. Ends by saving df_clusters (now labeled) and df_deep_clusters to
disk via save_dataframes(), so train_model.py never has to touch a point
cloud.

Split out of main.py so that:
  - sweep jobs (pinyon_sweep.sh) don't also run a full classifier comparison
    on every array task, which was bloating log output for no reason (the
    labeled set doesn't change between sweep iterations)
  - classifier iteration doesn't require re-running clustering

CLI args are unchanged from the old main.py, so pinyon_sweep.sh / pinyons.sh
work as-is — just point them at this file instead of main.py.
"""

print("Loading packages...")

from datetime import datetime
import numpy as np
import pandas as pd
import laspy
import open3d as o3d
import os
import argparse
import csv

from scipy.spatial import KDTree

from functions import *

# import global files, includes constants
import sys
sys.path.insert(0, "/home/hcr64/Pinyon-Detection")
from global_files import * 

print("Successfully loaded all packages.")


def str2bool(v):
    """argparse helper — accepts True/False, true/false, 1/0 etc."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{v}'")


def main():

    # ── CLI args ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps",               type=float, required=True)
    parser.add_argument("--green_threshold",   type=float, required=True)
    parser.add_argument("--max_radius",        type=float, required=True)
    parser.add_argument("--max_distance",      type=float, required=True)
    parser.add_argument("--min_points",        type=int,   required=True)
    parser.add_argument("--voxel_size",        type=float, required=True)
    parser.add_argument("--min_peak_distance", type=float, required=True)
    parser.add_argument("--k",                 type=int,   required=True)
    parser.add_argument("--min_height",        type=float, required=True)
    parser.add_argument("--search_radius_m",   type=float, required=True)
    parser.add_argument("--gps_sigma",         type=float, required=True)
    parser.add_argument("--smooth_sigma",      type=float, required=True)

    parser.add_argument("--trial_name", type=str, required=True)
    parser.add_argument("--job_id",     type=str, required=True)

    # gates ALL disk writes of clusters/dataframes/pcds/CHM/pre-split diagnostics.
    # Default True so single runs (pinyons.sh) behave exactly as before.
    # Sweep jobs (pinyon_sweep.sh) should pass --save False — matching_score
    # still gets appended to GPS_results regardless, since that's the whole
    # point of a sweep; this only silences the heavy artifact writes.
    parser.add_argument("--save", type=str2bool, default=True)

    args = parser.parse_args()

    EPS               = args.eps
    GREEN_THRESHOLD   = args.green_threshold
    MAX_RADIUS        = args.max_radius
    MAX_DISTANCE      = args.max_distance
    MIN_POINTS        = args.min_points
    VOXEL_SIZE        = args.voxel_size
    MIN_PEAK_DISTANCE = args.min_peak_distance
    K                 = args.k

    MIN_DENSITY_RATIO = 1.5

    MIN_HEIGHT      = args.min_height
    SEARCH_RADIUS_M = args.search_radius_m
    GPS_SIGMA       = args.gps_sigma
    SMOOTH_SIGMA    = args.smooth_sigma

    TRIAL_NAME = args.trial_name
    JOB_ID     = args.job_id
    SAVE       = args.save

    NORMALIZE_HEIGHTS = True

    PATHS = get_paths(TRIAL_NAME)

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
    print("MIN_DENSITY_RATIO:", MIN_DENSITY_RATIO)
    print("GPS_SIGMA:", GPS_SIGMA)
    print("SMOOTH_SIGMA:", SMOOTH_SIGMA)
    print("SAVE:", SAVE)

    K = MIN_POINTS

    print("TRIAL_NAME:", TRIAL_NAME)
    print("JOB_ID:", JOB_ID)
    print("Program Start Time: " + datetime.now().strftime("%H:%M:%S"))
    print()
    print()

    # ── load raw point cloud ─────────────────────────────────────────────
    if STEPS['Load_Pointcloud']:
        print('Loading point cloud...')
        point_cloud = las_folder_to_pointcloud(
            PATHS['Data'],
            silent=SILENT,
            downsize_pcd=DOWNSIZE,
            v_size=VOXEL_SIZE)
        print('Point cloud successfully loaded.\n')

        if SAVE:
            print(f"Saving raw point cloud... ( to { PATHS['Raw_pcd'] }")
            if not os.path.exists(os.path.dirname(PATHS['Raw_pcd'])):
                os.makedirs(os.path.dirname(PATHS['Raw_pcd']))
            o3d.io.write_point_cloud(f"{ PATHS['Raw_pcd'] }", point_cloud)
            print('Raw point cloud successfully saved.\n')
        else:
            print("SAVE=False — skipping raw point cloud write.\n")

    else:
        print(f"Reading in raw pointcloud... (from {PATHS['Raw_pcd']})")
        if os.path.exists(os.path.dirname(PATHS['Raw_pcd'])):
            point_cloud = o3d.io.read_point_cloud(PATHS['Raw_pcd'])
            print(f"Raw point cloud read in. ({PATHS['Raw_pcd']})")
        else:
            print(f"Could not find pointcloud save path ({PATHS['Raw_pcd']}). Exiting program...")
            return 1
        print()

    if STEPS['Make_CHM'] or STEPS['Clean_Pointcloud']:
        print("Downsizing pointcloud...")
        point_cloud = point_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print("Done downsizing.\n")

    # ── CHM ───────────────────────────────────────────────────────────────
    if STEPS['Make_CHM']:
        print('Making CHM from external PixMapper4D DSM...')
        chm, transform, crs = build_chm_from_external_dsm(
            dsm_path=PATHS['DSM'] + TRIAL_NAME + '_dsm.tif',
            point_cloud=point_cloud,
            dtm_path=None,
            ground_percentile=5,
            save_path=PATHS['CHM'] if SAVE else None
        )
        if SAVE:
            print(f"CHM complete. (Saved to {PATHS['CHM']})\n")
        else:
            print("CHM complete. (SAVE=False — not written to disk.)\n")
    else:
        if os.path.exists(PATHS['CHM']):
            print(f"Loading in CHM... (from { PATHS['CHM'] })")
            chm, transform, crs = load_chm(PATHS['CHM'])
        else:
            print(f"Cannot find CHM save path. ({ PATHS['CHM'] }) Exiting program...")
            return 1
        print('CHM loaded in successfully.')

    CHM_RESOLUTION = abs(transform.a)
    print(f"CHM resolution: {CHM_RESOLUTION} m/px")

    if STEPS['Make_Clusters']:
        print("Finding peaks in CHM...")
        peak_coords, peak_heights = find_chm_peaks(
            chm,
            transform,
            min_height=MIN_HEIGHT,
            search_radius_m=SEARCH_RADIUS_M,
            resolution=CHM_RESOLUTION,
            smooth_sigma=SMOOTH_SIGMA)
        print()

    # ── clean pointcloud ──────────────────────────────────────────────────
    if STEPS['Clean_Pointcloud']:
        points = np.asarray(point_cloud.points)
        print(f"Point count:  {len(points)}")
        print(f"XYZ min: {points.min(axis=0)}")
        print(f"XYZ max: {points.max(axis=0)}")
        print(f"Any NaN: {np.any(np.isnan(points))}")

        if NORMALIZE_HEIGHTS:
            print("Normalizing heights by ground surface...")
            point_cloud = normalize_heights_by_ground(point_cloud, resolution=0.5)
            print("Heights normalized.")

        print("Cleaning up point cloud...")
        point_cloud = clean_up_pointcloud(point_cloud, green_threshold=GREEN_THRESHOLD)
        print("Point cloud successfully cleaned.\n")

        print(point_cloud)

        if SAVE:
            print("Saving cleaned pointcloud...")
            if not os.path.exists(os.path.dirname(PATHS['Cleaned_pcd'])):
                os.makedirs(os.path.dirname(PATHS['Cleaned_pcd']))
            o3d.io.write_point_cloud(PATHS['Cleaned_pcd'], point_cloud)
            print("Cleaned pointcloud saved.\n")
        else:
            print("SAVE=False — skipping cleaned pointcloud write.\n")

    else:
        print("Unprocessed pointcloud being used. Pointcloud is not being 'cleaned.'\n")

    # ── cluster / load clusters ──────────────────────────────────────────
    if STEPS['Make_Clusters']:
        print("Clustering point cloud...")
        clusters = cluster_by_chm_peaks(
            point_cloud,
            peak_coords,
            chm=chm,
            transform=transform,
            crown_radius=MAX_RADIUS,
            min_points=MIN_POINTS
        )
        print("Point cloud clustered.\n")

        MIN_POINT_HEIGHT = 1.5
        clusters = [
            c for c in clusters
            if np.asarray(c.points)[:, 2].max() - np.asarray(c.points)[:, 2].min() > MIN_POINT_HEIGHT
        ]

        clusters = filter_clusters_by_green_crown(
            clusters,
            top_fraction=0.20,
            min_exg=0.05,
        )

        # ── ground strip + split + final filter (Make_Clusters branch only) ──
        # clusters loaded from disk in the else branch below have already been
        # through this — .ply files are only ever saved post-processing, so
        # reloading them never needs to redo it.
        if not STEPS['Clean_Pointcloud']:
            print("Stripping ground from clusters...")
            clusters = strip_ground_from_clusters(clusters, ground_percentile=10, min_height_above_ground=0.5)
            print("Ground stripped.\n")

        print("Splitting clusters...")
        clusters = split_large_clusters(
            clusters,
            min_points=MIN_POINTS,
            max_radius=MAX_RADIUS,
            min_peak_distance=MIN_PEAK_DISTANCE,
            k=K,
            min_density_ratio=MIN_DENSITY_RATIO,
            # split_large_clusters() calls save_clusters_descriptive() internally
            # whenever save_pre_split_path is not None — pass None during sweeps
            # to skip that write entirely rather than gating it after the fact
            save_pre_split_path=PATHS['PS_clusters'] if SAVE else None
        )
        print("Clusters split.\n")

        print("Filtering clusters...")
        before = len(clusters)
        clusters = [c for c in clusters if filter_cluster(c, min_height=1.0, min_radius=0.3)]
        print(f"Filtered {before - len(clusters)} non-tree clusters, {len(clusters)} remaining\n")

        # make dataframes
        print("Making cluster df...")
        df_clusters = clusters_to_dataframe(clusters, k=K)
        print("Cluster df made.\n")

        print("Making deep data cluster df...")
        df_deep_clusters = make_deep_dataframe(clusters)
        df_deep_clusters = engineer_features(df_deep_clusters)
        print("Deep data cluster df made.\n")

        # save the clusters after all processing has been done, dataframes get saved later on, more changes are made
        if SAVE:
            # save clusters
            print("Saving clusters...")
            save_clusters(clusters, PATHS['Clusters'])
            print("Clusters saved.\n")

        else:
            print("SAVE=False — skipping cluster .ply writes.\n")


    # if make clusters == false, load them in instead
    # also load in dataframes too 
    else:
        # trey loading in clusters
        print(f"Reading in clusters... (from {PATHS['Clusters']})")
        if os.path.exists(PATHS['Clusters']):
            clusters = load_clusters(PATHS['Clusters'])
            print("Clusters read in.")
        else:
            print(f"Could not find clusters save path ({PATHS['Clusters']}). Exiting program...")
            return 1

        # try loading in dataframes
        print("Reading in feature dataframes...")
        if os.path.exists(PATHS['Dataframes']):
            df_clusters, df_deep_clusters = load_dataframes(PATHS['Dataframes'])
            print("Dataframes read in.\n")
        else:
            print(f"Could not find dataframes save path ({PATHS['Dataframes']}). Exiting program...")
            return 1

        print()


    # ── GPS label matching ────────────────────────────────────────────────
    print("Assigning labels to clusters...")
    df_clusters, score = match_labels_to_clusters(
        PATHS['Labels'],
        df_clusters,
        csv_path_2=os.path.dirname(PATHS['Labels']) + '/sunsetCraterMay26.csv',
        max_distance=MAX_DISTANCE,
        graph_save_path=PATHS['Images'],
        gps_sigma=GPS_SIGMA,
        job_id=JOB_ID,
        clusters=clusters,
        # save_multimatch_clusters() writes one .ply folder per ambiguous
        # GPS label — skip during sweeps the same way as the other writes
        multi_match_save_path=PATHS["MM_save_path"] if SAVE else None,
        graph_subtitle=f"EPS:{EPS}-MR:{MAX_RADIUS}-GT:{GREEN_THRESHOLD}-MPts:{MIN_POINTS}-MD:{MAX_DISTANCE}-K:{K}-MPD:{MIN_PEAK_DISTANCE}-GPSs:{GPS_SIGMA}-SMs:{SMOOTH_SIGMA}"
    )
    print("Labels assigned.\n")

    # save labeled clusters
    if SAVE:
        print("Saving labeled clusters...")
        save_labeled_clusters(
            clusters,
            df_clusters,
            save_path=PATHS['Labeled_clusters']
        )
        print("Labeled clusters saved.\n")

        print("Saving feature dataframes...")
        save_dataframes(df_clusters, df_deep_clusters, PATHS['Dataframes'])
        print("Dataframes saved.\n")
    else:
        print("SAVE=False — skipping labeled cluster and dataframe writes.\n")

    # ── log matching score ────────────────────────────────────────────────
    # Always runs regardless of SAVE — this one-row CSV append is the actual
    # output a sweep exists to produce, and is cheap/safe under concurrent
    # array tasks (unlike the multi-MB cluster/CHM writes above).
    if STEPS['Cluster_accuracy'] and not SAVE:
        results = {
            "eps":               EPS,
            "green_threshold":   GREEN_THRESHOLD,
            "max_radius":        MAX_RADIUS,
            "max_distance":      MAX_DISTANCE,
            "min_points":        MIN_POINTS,
            "voxel_size":        VOXEL_SIZE,
            "min_peak_distance": MIN_PEAK_DISTANCE,
            "k":                 K,
            "min_height":        MIN_HEIGHT,
            "search_radius_m":   SEARCH_RADIUS_M,
            "gps_sigma":         GPS_SIGMA,
            "smooth_sigma":      SMOOTH_SIGMA,
            "normalize_heights": NORMALIZE_HEIGHTS,
            "matching_score":    score,
        }

        results_path = PATHS['GPS_results']
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        file_exists = os.path.exists(results_path)

        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

        print(f"Results saved to {results_path}")

    print("Clustering + labeling complete.")
    print("Time at Completion: " + datetime.now().strftime("%H:%M:%S"))


main()