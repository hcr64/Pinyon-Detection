# various paths that get accessed in the main function
PATHS = {
    # the data folder address with .las files
    'Data':'/home/hcr64/mydata/point_cloud/',

    # where the cleaned pointcloud is
    'Cleaned_pcd':'/home/hcr64/Pinyon-Detection/pointclouds/cleaned_pointcloud.ply',

    # Where the CSV of labels path is
    'Labels':'/home/hcr64/Pinyon-Detection/sunset-sfm.csv',

    # folder path for the cluster save
    'Clusters':'/home/hcr64/Pinyon-Detection/clusters/'

}

# the larger the number, the less inclusive to greenish points
GREEN_THRESHOLD = 0.03

# if we want to downsize or not
DOWNSIZE = True

# when downsizing, voxel size
VOXEL_SIZE = 0.10

# for clustering
EPS = 2.0
MIN_POINTS = 35

# for splitting larger clusters into smaller ones
MAX_RADIUS = 2.7

# for assigning labels (2-5)
MAX_DISTANCE = 2.7

# to get more printed messages
SILENT = True

# for doing/skipping sections of the main function
STEPS = {

    # if the large pointcloud needs to be cleaned again
    'Clean_Pointcloud':False,

    # if the clusters should be created again
    'Make_Clusters':True,
    
    # if a model should be trained
    'Train_Model':False
}
