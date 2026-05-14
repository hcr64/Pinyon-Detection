# the folder address
DATA_PATH = '/home/hcr64/mydata/point_cloud/'

# Where the CSV of labels path is
LABELS_PATH = '/home/hcr64/Pinyon-Detection/sunset-sfm.csv'

# folder path for the cluster save
CLUSTER_PATH = '/home/hcr64/Pinyon-Detection/clusters/'

# the larger the number, the less inclusive to greenish points
GREEN_THRESHOLD = 0.05

# if we want to downsize or not
DOWNSIZE = True

# when downsizing, voxel size
VOXEL_SIZE = 0.10

# for clustering
EPS = 2.5
MIN_POINTS = 45

# for assigning labels (2-5)
MAX_DISTANCE = 3.0

# to get more printed messages
SILENT = True

# if the clusters should be created again
MAKE_CLUSTERS = True

# if a model should be trained
TRAIN_MODEL = True
