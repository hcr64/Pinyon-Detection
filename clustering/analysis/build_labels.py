import argparse, sys
sys.path.insert(0, "/home/hcr64/Pinyon-Detection")
from global_files import get_paths
from clustering.functions.labeling.build_label_csv import build_label_csv

p = argparse.ArgumentParser()
p.add_argument("--trial_name", default="Sunset_sfm_trial")
p.add_argument("--dedup_distance", type=float, default=1.0)
a = p.parse_args()

PATHS = get_paths(a.trial_name)
build_label_csv(PATHS["Label_sources"], PATHS["Labels"],
                dedup_distance=a.dedup_distance)