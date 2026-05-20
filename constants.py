# various paths that get accessed in the main function
PATHS = {
    # the data folder address with .las files
    'Data':'/home/hcr64/mydata/{}/point_cloud/',

    # where the complete unprocessed pointcloud and processed pointclouds are
    'Pointclouds':'/home/hcr64/Pinyon-Detection/pointclouds/{}/',

    # Where the CSV of labels path is
    'Labels':'/home/hcr64/Pinyon-Detection/sunset-sfm.csv',

    # The log of variable changes
    'GPS_results':"/home/hcr64/Pinyon-Detection/results/results_minp.csv",

    # folder path for the cluster save
    'Clusters':'/home/hcr64/Pinyon-Detection/clusters/{}/'

}

# the larger the number, the less inclusive to greenish points
# GREEN_THRESHOLD = 0.03

# if we want to downsize or not
DOWNSIZE = True

# when downsizing, voxel size
VOXEL_SIZE = 0.10

# for splitting larger clusters into smaller ones
# MAX_RADIUS = 3.5

# for assigning labels (2-5)
MAX_DISTANCE = 2.7

# to get more printed messages
SILENT = True

# for doing/skipping sections of the main function
STEPS = {
    'Load_Pointcloud':False,

    # if the large pointcloud needs to be cleaned again
    # very time consuming, more than 10 minutes
    'Clean_Pointcloud':True,

    # if the clusters should be created again
    # a little time consuming, usually at least 3 minutes or so
    'Make_Clusters':True,

    # Evaluate the GPS points to cluster accuracy 
    # A few seconds at most
    'Cluster_accuracy':False,
    
    # if a model should be trained
    # not very time consuming, usually a few seconds or less
    'Train_Model':False
}
