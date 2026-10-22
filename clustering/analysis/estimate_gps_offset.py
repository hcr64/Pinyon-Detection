import sys
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

D = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/hcr64/Pinyon-Detection/trial_data/Sunset_sfm_trial/dataframes/"
MAX_DIST = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

gps = pd.read_csv(D + "gps_loss_diagnosis.csv")
cl = pd.read_csv(D + "clusters.csv")
G = gps[["x", "y"]].values
tree = KDTree(cl[["x_pos", "y_pos"]].values)


def n_perfect(coords):
    hits = tree.query_ball_point(coords, r=MAX_DIST)
    return sum(len(h) == 1 for h in hits)


# ── 1. direction of the nearest cluster for no-match points ──────────────
nm = gps[gps["final_status"] == "no_match"]
_, idx = tree.query(nm[["x", "y"]].values)
vec = cl[["x_pos", "y_pos"]].values[idx] - nm[["x", "y"]].values
unit = vec / np.linalg.norm(vec, axis=1, keepdims=True)
R = np.linalg.norm(unit.mean(axis=0))
print(f"No-match points: {len(nm)}")
print(f"Mean vector to nearest cluster: dx={vec[:,0].mean():+.2f} m, "
      f"dy={vec[:,1].mean():+.2f} m  (+x east, +y north)")
print(f"Direction consistency R={R:.2f}  "
      f"(random directions give about {1/np.sqrt(len(nm)):.2f}; above ~0.5 is systematic)\n")

# ── 2. grid search: which GPS shift maximises 1:1 matches ────────────────
steps = np.arange(-8, 8.01, 0.25)
res = [(dx, dy, n_perfect(G + [dx, dy])) for dx in steps for dy in steps]
res = pd.DataFrame(res, columns=["dx", "dy", "perfect"]).sort_values(
    "perfect", ascending=False)
base = n_perfect(G)
print(f"Perfect matches with no shift: {base}/{len(G)}")
print("Top shifts (add to GPS coords):")
print(res.head(5).to_string(index=False))
print(f"Median over all shifts: {res['perfect'].median():.0f}\n")

# ── 3. split-half check: fit on half the points, score the other half ────
rng = np.random.default_rng(0)
for rep in range(3):
    mask = rng.random(len(G)) < 0.5
    A, B = G[mask], G[~mask]
    best = max(((dx, dy) for dx in steps for dy in steps),
               key=lambda s: sum(len(h) == 1 for h in
                                 tree.query_ball_point(A + s, r=MAX_DIST)))
    gain = (sum(len(h) == 1 for h in tree.query_ball_point(B + best, r=MAX_DIST))
            - sum(len(h) == 1 for h in tree.query_ball_point(B, r=MAX_DIST)))
    print(f"split {rep}: fit shift ({best[0]:+.2f}, {best[1]:+.2f}) "
          f"-> held-out gain {gain:+d} of {len(B)} points")