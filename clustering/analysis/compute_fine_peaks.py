import argparse, os, sys
import numpy as np

sys.path.insert(0, "/home/hcr64/Pinyon-Detection")
from global_files import get_paths
from functions.preprocessing.build_chm import load_chm
from functions.detection.find_chm_peaks import find_chm_peaks

p = argparse.ArgumentParser()
p.add_argument("--trial_name", default="Sunset_sfm_trial")
p.add_argument("--smooth_sigma", type=float, required=True)   # pixels, at native res
p.add_argument("--search_radius_m", type=float, required=True)
p.add_argument("--min_height", type=float, default=0.5)
p.add_argument("--tag", required=True)                        # used in the output filename
a = p.parse_args()

PATHS = get_paths(a.trial_name)
chm, transform, crs = load_chm(PATHS["CHM"])
res = abs(transform.a)
print(f"CHM {chm.shape}, {res:.4f} m/px")

coords, heights = find_chm_peaks(
    chm, transform, min_height=a.min_height,
    search_radius_m=a.search_radius_m, resolution=res,
    smooth_sigma=a.smooth_sigma,
)

out_dir = PATHS["Dataframes"]
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f"fine_peaks_{a.tag}.npz")
np.savez(out, coords=coords, heights=heights, res=res,
         smooth_sigma=a.smooth_sigma, search_radius_m=a.search_radius_m,
         min_height=a.min_height)
print(f"Saved {len(coords)} peaks -> {out}")