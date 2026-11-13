# for doing/skipping sections of the main function
STEPS = {
    # If you want to load the entire 'raw' pointcloud.
    # once it has been done & saved, does not need to be doen again.
    # Takes more than 10 minutes if true usually
    'Load_Pointcloud':False,

    # making the canopy from height model with the unprocesses pcd
    # once done and saved, does not need to be ran again
    # takes less than 2 minutes 
    'Make_CHM':True,

    # makes all clusters, before any filtering is done. 
    'Make_Raw_Clusters':False,
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
    local_trial_dir = f'/home/hcr64/Pinyon-Detection/trial_data/{trial_name}/'
    scratch_dir=f'/scratch/hcr64/Pinyon-Detection/data/{trial_name}/'

    return {

        # scratch data folder paths
        # these can both take in external data
        # Data needs .las files, DSM can take an available dsm if possible
        'Data':             scratch_dir + 'point_cloud/', # .las files need to be here for processing
        'DSM':              scratch_dir + '1_dsm/' + trial_name + '_dsm.tif', # a DSM can be here if available, but if not one can be made


        # everything below are for writing to disk. If already saved, 
        # they can speed up the duration of the program by several minutes.
        'Pointclouds':      scratch_dir + 'pointclouds/',
        'Raw_pcd':          scratch_dir + 'pointclouds/raw_pcd.ply',
        'Cleaned_pcd':      scratch_dir + 'pointclouds/cleaned_pcd.ply',
        'CHM':              scratch_dir + 'CFMs/chm.tif',
        'Raw_Clusters':     scratch_dir + 'raw_clusters/',
        'Clusters':         scratch_dir + 'clusters/',
        'PS_clusters':      scratch_dir + 'pre_split_clusters/',
        'Labeled_clusters': scratch_dir + 'labeled_clusters/',
        'MM_save_path':     scratch_dir + 'multi_match_clusters/',


        # local folder paths
        'Label_sources': [
            local_trial_dir + 'labels/sunset-sfm.csv',
            local_trial_dir + 'labels/sunsetCraterMay26.csv',
            local_trial_dir + 'labels/Sunset_crater_oct292026.csv',
        ],
        'Labels': local_trial_dir + 'labels/processed_labels.csv',
        
        'Images':           local_trial_dir + 'images/',
        'GPS_results':      local_trial_dir + 'GPS_results/',
        'Dataframes':       local_trial_dir + 'dataframes/',
        'Models':           local_trial_dir + 'models/',
    }

