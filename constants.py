# the trial name
TRIAL_NAME = 'Sunset_sfm_trial'

# the path of trial dir
TRIAL_DIR = '/home/hcr64/Pinyon-Detection/trial_data/{}/'.format( TRIAL_NAME )

# various paths that get accessed in the main function
PATHS = {
    # the data folder address with .las files
    'Data': TRIAL_DIR + 'data/point_cloud/',

    # where the complete unprocessed pointcloud and processed pointclouds are
    'Pointclouds': TRIAL_DIR + 'pointclouds/',

    # where to save/load the raw complete pointcloud from
    'Raw_pcd':TRIAL_DIR + 'pointclouds/raw_pcd.ply',
    'Cleaned_pcd':TRIAL_DIR + 'pointclouds/cleaned_pcd.ply',

    # where to save/load the canopy height model from
    'CHM': TRIAL_DIR + 'CFMs/chm.tif',

    # Where the CSV of labels path is
    'Labels':TRIAL_DIR + 'labels/sunset-sfm.csv',

    # where to save images
    'Images': TRIAL_DIR + 'images/',

    # Where to log parameters & gps / label matching scores
    'GPS_results':TRIAL_DIR + "results/newester_results.csv",

    # folder path for the cluster save
    'Clusters':TRIAL_DIR + 'clusters/',

    # folder for labled clusters
    'Labeled_clusters':TRIAL_DIR + 'labeled_clusters/'
}

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

    # if clusters need to be split or not
    # not too time consuming
    'Split_clusters':True,

    # Evaluate the GPS points to cluster accuracy 
    # A few seconds at most
    'Cluster_accuracy':True,
    
    # if a model should be trained
    # not very time consuming, usually a few seconds or less
    'Train_Model':True
}
