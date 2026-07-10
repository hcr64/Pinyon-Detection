# if we want to downsize or not
DOWNSIZE = True

# to get more printed messages
SILENT = True

# for doing/skipping sections of the main function
STEPS = {
    # If you want to load the entire 'raw' pointcloud.
    # once it has been done & saved, does not need to be doen again.
    # Takes more than 10 minutes if true usually
    'Load_Pointcloud':False,

    # making the canopy from height model with the unprocesses pcd
    # once done and saved, does not need to be ran again
    # takes less than 2 minutes 
    'Make_CHM':False,

    # if the large pointcloud needs to be cleaned again
    # very time consuming, more than 10 minutes
    'Clean_Pointcloud':False,

    # if the clusters should be created again
    # a little time consuming, usually at least 3 minutes or so
    'Make_Clusters':True,

    # Evaluate the GPS points to cluster accuracy 
    # A few seconds at most
    'Cluster_accuracy':True,
    
    # if a model should be trained
    # not very time consuming, usually a few seconds or less
    'Train_Model':True
}

# path builder 
def get_paths(trial_name, scratch_dir='/scratch/hcr64/Pinyon-Detection/'):
    """
    Build all pipeline I/O paths for a given trial name.

    Args:
        trial_name (str): Name of the trial subfolder under trial_data/.
            Passed in from --trial_name at the command line.
        scratch_dir (str): Root scratch directory for large read-only inputs
            (raw .las files). Defaults to the Monsoon scratch path.

    Returns:
        dict: Path strings keyed by role (Data, Raw_pcd, CHM, etc.)
    """
    trial_dir = f'/home/hcr64/Pinyon-Detection/trial_data/{trial_name}/'

    return {
        'Data':             scratch_dir + f'data/{trial_name}/point_cloud/',
        'Pointclouds':      trial_dir + 'pointclouds/',
        'Raw_pcd':          trial_dir + 'pointclouds/raw_pcd.ply',
        'Cleaned_pcd':      trial_dir + 'pointclouds/cleaned_pcd.ply',
        'CHM':              trial_dir + 'CFMs/chm.tif',
        'Labels':           trial_dir + 'labels/sunset-sfm.csv',
        'Images':           trial_dir + 'images/',
        'GPS_results':      trial_dir + 'results/results_Jul4.csv',
        'Clusters':         trial_dir + 'clusters/',
        'Labeled_clusters': trial_dir + 'labeled_clusters/',
        'Dataframes':       trial_dir + 'dataframes/',
        'MM_save_path':     trial_dir + 'multi_match_clusters/'
    }

