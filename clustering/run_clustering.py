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
import time

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

def append_result_row(path, row, preamble=None, lock_timeout=60):
    """Append one row to a per-sweep CSV. The first writer creates the file with
    the preamble (# comment lines) and the header. Uses an atomic mkdir lock
    instead of flock, which isn't reliable on some cluster filesystems."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_dir = path + ".lock"

    start = time.time()
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            if time.time() - start > lock_timeout:
                raise TimeoutError(f"Could not get {lock_dir} in {lock_timeout}s. "
                                   f"If no task is writing, delete it and rerun.")
            time.sleep(0.2)

    try:
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            if new_file and preamble:
                f.write("\n".join(preamble) + "\n")
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    finally:
        os.rmdir(lock_dir)

REQUIRED = object()   # sentinel: no default, flag must be passed

# name: (type, default). Order here == order of columns in the results CSV.
PARAMS = {
    # swept / required
    "max_radius":        (float, REQUIRED),
    "max_distance":      (float, REQUIRED),
    "min_points":        (int,   REQUIRED),
    "min_peak_distance": (float, REQUIRED),
    "k":                 (int,   REQUIRED),
    "min_height":        (float, REQUIRED),
    "search_radius_m":   (float, REQUIRED),
    "smooth_sigma":      (float, REQUIRED),

    # dead / no effect on matching_score, optional so pinyons.sh still works
    "eps":               (float, 2.0),
    "green_threshold":   (float, 0.025),
    "voxel_size":        (float, 0.08),
    "gps_sigma":         (float, 4.0),

    # live args that used to be hardcoded
    "crown_radius":       (float,    8.0),
    "min_density_ratio":  (float,    1.5),
    "use_color":          (str2bool, False),
    "color_weight":       (float,    10.0),
    "min_exg":            (float,    0.05),
    "min_cluster_height": (float,    1.5),
}

# not CLI args, not swept
FIXED = {"mask_chm_filled": 0.3}

# columns written to the results CSV, in order (dead params left out on purpose)
RESULT_KEYS = [
    "max_radius", "crown_radius", "max_distance", "min_points",
    "min_peak_distance", "k", "min_density_ratio", "min_height",
    "search_radius_m", "smooth_sigma", "use_color", "color_weight",
    "min_exg", "min_cluster_height",
]

def main():
    # get the arg parser
    parser = argparse.ArgumentParser()

    # get sweepable params
    for name, (typ, default) in PARAMS.items():
        if default is REQUIRED:
            parser.add_argument(f"--{name}", type=typ, required=True)
        else:
            parser.add_argument(f"--{name}", type=typ, default=default)

    # technical vars, not part of the param dict
    parser.add_argument("--trial_name", type=str, required=True)
    parser.add_argument("--job_id",     type=str, required=True)
    parser.add_argument("--notes",      type=str, default="N/A")
    parser.add_argument("--results_path", type=str, default="N/A")
    parser.add_argument("--save",       type=str2bool, default=True)
    args = parser.parse_args()

    # make the param dict
    P = {name: getattr(args, name) for name in PARAMS}
    P.update(FIXED)

    # technical vars
    TRIAL_NAME, JOB_ID, RESULTS_PATH, SAVE = args.trial_name, args.job_id, args.results_path, args.save

    # print the settings
    print("\n\n#### SETTINGS ####")
    for name, val in P.items():
        print(f"{name.upper()}: {val}")
    print("SAVE:", SAVE)
    print("TRIAL_NAME:", TRIAL_NAME)
    print("JOB_ID:", JOB_ID)
    print("Program Start Time: " + datetime.now().strftime("%H:%M:%S"))
    print("\n")

    SWEEP_ID = os.environ.get("SLURM_ARRAY_JOB_ID", JOB_ID)
    TASK_ID  = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    PATHS = get_paths(TRIAL_NAME)

    # if  want to downsize the raw pointcloud or not
    # this variable is on the chopping block, it is usually set to true
    DOWNSIZE = True

    # to get more in-depth printed messages
    SILENT = True

    # ── load raw point cloud ─────────────────────────────────────────────
    if STEPS['Load_Pointcloud'] and STEPS['Make_Raw_Clusters']:
        print('Loading point cloud...')
        point_cloud = las_folder_to_pointcloud(
            PATHS['Data'],
            silent=SILENT,
            downsize_pcd=DOWNSIZE,
            v_size=P['voxel_size'])
        print('Point cloud successfully loaded.\n')

        if SAVE:
            print(f"Saving raw point cloud... ( to { PATHS['Raw_pcd'] }")
            if not os.path.exists(os.path.dirname(PATHS['Raw_pcd'])):
                os.makedirs(os.path.dirname(PATHS['Raw_pcd']))
            o3d.io.write_point_cloud(f"{ PATHS['Raw_pcd'] }", point_cloud)
            print('Raw point cloud successfully saved.\n')
        else:
            print("SAVE=False — skipping raw point cloud write.\n")

    # do not load in the raw pointcloud, instead read it in from disk
    else:
        print(f"Reading in raw pointcloud... (from {PATHS['Raw_pcd']})")
        if os.path.exists(os.path.dirname(PATHS['Raw_pcd'])):
            point_cloud = o3d.io.read_point_cloud(PATHS['Raw_pcd'])
            print(f"Raw point cloud read in. ({PATHS['Raw_pcd']})")
        else:
            print(f"Could not find pointcloud save path ({PATHS['Raw_pcd']}). Exiting program...")
            return 1
        print()


    # ── CHM ───────────────────────────────────────────────────────────────
    if STEPS['Make_CHM'] and STEPS['Make_Raw_Clusters']:
        # downsize the pointcloud, can take a while
        print("Downsizing pointcloud...")
        point_cloud = point_cloud.voxel_down_sample(voxel_size=P['voxel_size'])
        print("Done downsizing.\n")

        # make the chm
        print('Making CHM from external PixMapper4D DSM...')
        chm, transform, crs = build_chm_from_external_dsm(
            dsm_path=PATHS['DSM'],
            point_cloud=point_cloud,
            dtm_path=None,
            ground_percentile=5,
            save_path=PATHS['CHM'] if SAVE else None
        )
        if SAVE:
            print(f"CHM complete. (Saved to {PATHS['CHM']})\n")
        else:
            print("CHM complete. (SAVE=False — not written to disk.)\n")

    # read in the chm from memory
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


    ### With evertyhting loaded in, the clustering begins
    ### Very time consuming, can take up to an hour or longer in some cases

    # make the clusters
    if STEPS['Make_Raw_Clusters']:
        # find peaks in the caonpy height model, use to cluster later
        print("Finding peaks in CHM...")
        peak_coords, peak_heights = find_chm_peaks(
            chm,
            transform,
            min_height=P['min_height'],
            search_radius_m=P['search_radius_m'],
            resolution=CHM_RESOLUTION,
            smooth_sigma=P['smooth_sigma'])
        print()


        # cluster the chm by the peaks. Takes a while
        print("Clustering point cloud...")
        clusters = cluster_by_chm_peaks(
            point_cloud,
            peak_coords,
            chm=chm,
            transform=transform,
            crown_radius=P['crown_radius'],
            min_points=P['min_points'],
            mask_chm_filled=P['mask_chm_filled'],
        )
        print("Point cloud clustered.\n")

        if SAVE:
            # save new clusters to the raw clusters path
            save_clusters(clusters, PATHS['Raw_Clusters'])

    # load in the raw clusters from disk
    else:
        print(f"Reading in raw clusters... (from {PATHS['Raw_Clusters']})")
        if os.path.exists(os.path.dirname(PATHS['Raw_Clusters'])):
            clusters = load_clusters(PATHS['Raw_Clusters'])
            print(f"Raw clusters cloud read in. ({PATHS['Raw_Clusters']})")
        else:
            print(f"Could not find raw cluster save path ({PATHS['Raw_Clusters']}). Exiting program...")
            return 1
        print()

    stages = {"raw": clusters} 

    # filter clusters by height
    clusters = [
        c for c in clusters
        if np.asarray(c.points)[:, 2].max() - np.asarray(c.points)[:, 2].min() > P['min_cluster_height']
    ]
    stages["height>1.5"] = clusters

    # filter clusters by how green their most green points are, my custom function
    clusters = filter_clusters_by_green_crown(
        clusters, 
        top_fraction=0.20, 
        min_exg=P['min_exg']
        )
    stages["green_crown"] = clusters


    # strip ground from the clusters, make sure it is just tree
    print("Stripping ground from clusters...")
    clusters = strip_ground_from_clusters(clusters, ground_percentile=10, min_height_above_ground=0.5)
    print("Ground stripped.\n")
    stages["ground_strip"] = clusters


    # splits clusters that are likely multiple trees
    # uses point density cores to find multiple clusters within one
    print("Splitting clusters...")
    clusters = split_large_clusters(
        clusters,
        min_points=P['min_points'], 
        max_radius=P['max_radius'],
        min_peak_distance=P['min_peak_distance'], 
        k=P['k'],
        min_density_ratio=P['min_density_ratio'],
        use_color=P['use_color'], 
        color_weight=P['color_weight'],
        save_pre_split_path=PATHS['PS_clusters'] if SAVE else None,
        silent=SILENT
    )
    print("Clusters split.\n")
    stages["split"] = clusters

    # filter out clusters by height and radius, make sure they are all likely trees
    print("Filtering clusters...")
    before = len(clusters)
    clusters = [c for c in clusters if filter_cluster(c, min_height=1.0, min_radius=0.3)]
    print(f"Filtered {before - len(clusters)} non-tree clusters, {len(clusters)} remaining\n")
    stages["final"] = clusters                       # must be last
    
    # make dataframes, these are less complex that can also be sued for graphing
    print("Making cluster df...")
    df_clusters = clusters_to_dataframe(clusters, k=P['k'])
    print("Cluster df made.\n")

    # get a deeper feature df specifically for model training
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


    # ── GPS label matching ────────────────────────────────────────────────
    print("Assigning labels to clusters...")
    df_clusters, score, match_counts = match_labels_to_clusters(
        PATHS['Labels'],
        df_clusters,
        csv_path_2=os.path.dirname(PATHS['Labels']) + '/sunsetCraterMay26.csv',
        max_distance=P['max_distance'],
        graph_save_path=PATHS['Images'],
        gps_sigma=P['gps_sigma'],
        job_id=JOB_ID,
        clusters=clusters,
        # save_multimatch_clusters() writes one .ply folder per ambiguous
        # GPS label — skip during sweeps the same way as the other writes
        multi_match_save_path=PATHS["MM_save_path"] if SAVE else None,
        graph_subtitle=f"{P}",
        stage_clusters=None, #stages if SAVE else None,
        diagnostic_save_path=PATHS['Dataframes'] + 'gps_loss_diagnosis.csv' if SAVE else None,
        centroid_k=P['k'],
        mm_cluster_stats_path=PATHS['Dataframes'] + 'mm_clusters_stats.csv',
        silent=SILENT
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
    # Always runs, for every run — sweep task or full manual run via
    # pinyons.sh alike. The score computation itself is already
    # unconditional (see match_labels_to_clusters() above); this just
    # appends one cheap row to the results CSV so every run's score gets
    # a permanent record, not only sweep runs.
    results = {
        "task": TASK_ID,
        **{key: P[key] for key in RESULT_KEYS},
        "matching_score": score,
        **match_counts,
    }

    append_result_row(RESULTS_PATH, results )
    print(f"Results saved to {RESULTS_PATH}")

    # print ending message
    print("Clustering + labeling complete.")
    print("Time at Completion: " + datetime.now().strftime("%H:%M:%S"))


main()