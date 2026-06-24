# Pipeline Overview

This document walks through every stage of the Pinyon-Detection pipeline in execution order, with the function responsible for each step, its inputs and outputs, and notes on why it works the way it does.

---

## Stage Map

```
Raw .las files
      │
      ▼
┌─────────────────────┐
│  1. Load Point Cloud │  las_folder_to_pointcloud()
└─────────────────────┘
      │  raw Open3D PointCloud (absolute Z, UTM metres)
      ▼
┌─────────────────────┐
│  2. Build CHM        │  build_chm()
└─────────────────────┘
      │  CHM raster (float32 GeoTIFF) + Affine transform
      ▼
┌─────────────────────┐
│  3. Find CHM Peaks   │  find_chm_peaks()
└─────────────────────┘
      │  (N, 2) UTM peak coordinates — one per candidate tree top
      ▼
┌─────────────────────┐
│  4. Clean Point Cloud│  clean_up_pointcloud()  [+ optional normalize_heights_by_ground()]
└─────────────────────┘
      │  green-filtered PointCloud
      ▼
┌─────────────────────┐
│  5. Segment Crowns   │  cluster_by_chm_peaks()   (watershed)
└─────────────────────┘
      │  list of per-tree PointCloud clusters
      ▼
┌─────────────────────┐
│  6. Strip Ground     │  strip_ground_from_clusters()
└─────────────────────┘
      │  clusters with sub-canopy ground points removed
      ▼
┌─────────────────────┐
│  7. Split Large      │  split_large_clusters()   (Mean Shift)
│     Clusters         │
└─────────────────────┘
      │  refined clusters — merged crowns separated
      ▼
┌─────────────────────┐
│  8. Extract Features │  clusters_to_dataframe()  +  make_deep_dataframe()
└─────────────────────┘
      │  df_clusters (geometry/position)  +  df_deep_clusters (shape/color/PCA)
      ▼
┌─────────────────────┐
│  9. Match GPS Labels │  match_labels_to_clusters()
└─────────────────────┘
      │  df_clusters with "Name" column + matching score
      ▼
┌─────────────────────┐
│ 10. Train Classifier │  train_tree_classifier()
└─────────────────────┘
      │  trained model + predict_proba columns on df_deep_clusters
      ▼
┌─────────────────────┐
│ 11. Save Outputs     │  save_clusters()  /  save_labeled_clusters()
└─────────────────────┘
```

---

## Stage Details

### 1. Load Point Cloud
**Function:** `io/las_folder_to_pointcloud.py` → `las_folder_to_pointcloud()`

Reads all `.las` files from the data folder and merges them into one Open3D `PointCloud`. RGB is extracted and normalised from 16-bit (0–65535) to float (0.0–1.0). The combined cloud is optionally voxel-downsampled.

- The raw cloud is saved to `pointclouds/raw_pcd.ply` and reused on subsequent runs (`STEPS['Load_Pointcloud'] = False`).
- Coordinates are in UTM metres (EPSG:26912, zone 12N for Sunset Crater).

---

### 2. Build Canopy Height Model (CHM)
**Function:** `preprocessing/build_chm.py` → `build_chm()`

Rasterises the raw point cloud at 0.5 m/pixel. Each cell gets:
- **DSM** — maximum Z (top of canopy or ground, whichever is highest)
- **DTM** — low-percentile Z (proxy for ground surface, default 5th percentile)
- **CHM** = DSM − DTM, clipped to [0, ∞)

Output is a float32 GeoTIFF saved to `CFMs/chm.tif` and reused on subsequent runs. The rasterio `Affine` transform is returned alongside the array so pixel indices can be converted back to UTM coordinates anywhere in the pipeline.

> **Why use the raw cloud for the CHM?** The green filter and height normalisation happen *after* CHM building. The CHM needs ground returns to compute a DTM — filtering them out first would break the ground estimate.

---

### 3. Find CHM Peaks
**Function:** `detection/find_chm_peaks.py` → `find_chm_peaks()`

Applies a `scipy.ndimage.maximum_filter` across the CHM to find local maxima. Each pixel that equals the local maximum *and* exceeds `min_height` is a candidate tree top. Flat-topped regions (multiple equal pixels) are collapsed to their centroid. Pixel indices are converted to UTM coordinates via the rasterio transform.

Returns `(N, 2)` UTM peak coordinates — one per detected tree top. These become the watershed markers in Stage 5.

Key parameters:
- `min_height` — minimum canopy height in metres (1.0 m worked best at Sunset Crater)
- `search_radius_m` — local max window radius; controls how close two peaks can be (3.0 m worked best)

---

### 4. Clean Point Cloud
**Functions:** `preprocessing/build_chm.py` → `normalize_heights_by_ground()` (optional),  
`preprocessing/clean_up_pointcloud.py` → `clean_up_pointcloud()`

**Height normalisation (optional):** Subtracts a per-cell ground estimate from each point's Z so heights are relative to local terrain rather than sea level. Useful for sloped sites but found to *hurt* matching scores at Sunset Crater (`normalize_heights=False` is the current setting).

**Green filter:** Keeps only points where the green channel dominates both red and blue by at least `green_threshold`. This removes bare ground, rock, and sky while retaining vegetation. The threshold of 0.025 was found to be the sweet spot in parameter sweeps.

The cleaned cloud is saved to `pointclouds/cleaned_pcd.ply`.

---

### 5. Segment Crowns (Watershed)
**Function:** `detection/cluster_by_chm_peaks.py` → `cluster_by_chm_peaks()`

Uses marker-controlled watershed segmentation on the CHM, then assigns each 3D point to its watershed region. This replaces the earlier radius-based nearest-peak assignment.

Steps inside the function:
1. Convert peak UTM coordinates to raster pixel indices
2. Build a marker image (each peak = one integer label)
3. Slightly dilate markers (3×3 footprint) so watershed seeds are more robust
4. Run `skimage.segmentation.watershed` on the *inverted* CHM — high canopy becomes a deep basin, watershed floods uphill from each seed
5. Map each 3D point's XY → raster label
6. Apply a hard `crown_radius` cap: points further than `crown_radius` metres from any peak get label 0 (discarded)
7. Build one `PointCloud` per label; discard clusters below `min_points`

> **Why watershed instead of radius assignment?** Radius assignment draws a circle around each peak and assigns points to the nearest peak. Watershed respects the actual ridgeline between adjacent crowns, handles non-circular crowns, and prevents one large tree from stealing points from a smaller neighbour.

A simple height filter (cluster Z range > 1.5 m) runs immediately after to drop flat ground patches that slipped through.

---

### 6. Strip Ground from Clusters
**Function:** `preprocessing/strip_ground_from_clusters.py` → `strip_ground_from_clusters()`

For each cluster, estimates the local ground elevation as the `ground_percentile`-th percentile of Z values within that cluster, then discards all points below `ground + min_height_above_ground`. Done per-cluster so that trees on a slope each get their own local ground reference rather than a single global Z cutoff.

Default parameters: `ground_percentile=10`, `min_height_above_ground=0.5 m`.

---

### 7. Split Large Clusters
**Function:** `detection/split_large_clusters.py` → `split_large_clusters()`

Clusters whose XY radius exceeds `max_radius` are candidates for splitting — they likely contain two or more adjacent trees whose crowns merged during watershed segmentation.

Split logic:
1. **Gate 1 — size check:** skip clusters within `max_radius`
2. **Gate 2 — density peak check:** run `find_density_peaks()` using a KDTree. Only clusters with ≥ 2 well-separated density peaks proceed to splitting. This prevents splitting a single large tree that happens to have a wide crown.
3. **Mean Shift on XY:** fit `sklearn.cluster.MeanShift` with `bandwidth = min_peak_distance`. Mean Shift discovers the number of sub-crowns automatically — no `n_clusters` required.
4. **Validation:** each sub-cluster must pass `filter_cluster()` (minimum height 1.0 m, minimum radius 0.3 m). Sub-clusters that fail are discarded individually; the split is only fully reverted if *no* sub-clusters survive.

> **Why Mean Shift instead of KMeans?** KMeans requires specifying `n_clusters` up front. Mean Shift finds the number of modes automatically from point density, which is critical when the number of merged trees per cluster varies.

Pre-split clusters are saved to `pre_split_clusters/` for inspection.

---

### 8. Extract Features
**Functions:**  
`features/get_pointcloud_stats.py` → `clusters_to_dataframe()` (position/geometry)  
`features/get_deep_cluster_features.py` → `make_deep_dataframe()` (shape/color/PCA)

Two dataframes are built from the cluster list:

**df_clusters** — used for GPS label matching and scoring:

| Column | Description |
|---|---|
| `n_points` | Total point count |
| `height` | Z range of AABB |
| `radius` | Max XY half-width of AABB |
| `x_pos`, `y_pos` | Density-weighted centroid (XY) |
| `density` | Points per cubic metre |

The centroid uses `density_weighted_center()` — a KDTree-based density estimate that weights denser regions more heavily, so the centre is pulled toward the trunk rather than the sparse crown edge.

**df_deep_clusters** — used for model training:

| Column group | Features |
|---|---|
| Shape | `height`, `radius`, `n_points`, OBB extents (x/y/z) |
| PCA | `eigenvalue_1/2/3`, `linearity`, `planarity`, `sphericity` |
| Color | `mean_r/g/b`, `std_r/g/b` |

---

### 9. Match GPS Labels
**Function:** `labeling/match_labels_to_clusters.py` → `match_labels_to_clusters()`

Converts GPS coordinates (WGS84 lat/lon) to UTM using `pyproj`, then performs optimal one-to-one assignment between GPS labels and cluster centroids using `scipy.optimize.linear_sum_assignment` (the Hungarian algorithm). Labels further than `max_distance` metres from their assigned cluster are discarded.

Also computes the **matching score** — the fraction of GPS points with exactly one cluster within `max_distance`. Saves two diagnostic plots:
- `coordinate_overlap.png` — GPS points coloured by match quality (perfect / no match / multiple)
- `GPS_Clusters_<job_id>.png` — species scatter overlaid on cluster positions

GPS labels are loaded from up to two CSVs, merged, deduplicated, and normalised (dead trees excluded, species names standardised to `pinyon` / `juniper` / `ponderosa`).

---

### 10. Train Classifier
**Function:** `classification/train_tree_classifier.py` → `train_tree_classifier()`

Trains species classifiers on the labeled subset of `df_deep_clusters`. The labeled subset is the intersection of `df_deep_clusters` and clusters assigned a known species in Step 9.

The current implementation compares multiple model types (Gradient Boosting, SVM/RBF, Logistic Regression, KNN, and an ensemble/neural net), each wrapped in a `Pipeline` with `StandardScaler`. Cross-validation uses `StratifiedKFold` with fold count capped at `min(3, min_class_size)` to handle small juniper/ponderosa samples.

Outputs:
- Per-class precision / recall / F1
- Macro F1 and Cohen's Kappa (preferred over raw accuracy given class imbalance)
- Confusion matrix PNGs
- `predict_proba` columns appended to `df_deep_clusters`
- `predicted_label` column (best model's prediction for every cluster, including unlabeled ones)

In `main.py`, predictions are filtered by confidence: only clusters predicted as `pinyon` with > 80% confidence are counted as confirmed detections.

> **Class imbalance note:** Pinyon likely dominates the label set. Raw accuracy is misleading — always evaluate using per-class metrics and macro F1.

---

### 11. Save Outputs
**Functions:** `io/save_clusters.py` → `save_clusters()`, `io/save_labeled_clusters.py` → `save_labeled_clusters()`

- `save_clusters()` — saves the full cluster list as `cluster0.ply`, `cluster1.ply`, … Clears the folder before writing so it always reflects the current run.
- `save_labeled_clusters()` — saves only clusters with a confirmed GPS species label, named `pinyon_0.ply`, `juniper_0.ply`, etc.
- Labeled clusters are synced to Google Drive via `rclone` using `shells/save_clusters_to_drive.sh`.

---

## Step Flags

All stages are individually togglable in `constants.py` under `STEPS`. Expensive stages (loading `.las` files, building the CHM, cleaning the point cloud, making clusters) only need to run once — their outputs are saved to disk and reloaded on subsequent runs.

| Flag | Expensive? | Notes |
|---|---|---|
| `Load_Pointcloud` | Yes (~10 min) | Only needed when raw `.las` data changes |
| `Make_CHM` | No (~2 min) | Only needed when raw cloud or resolution changes |
| `Clean_Pointcloud` | Yes (~10 min) | Only needed when green threshold changes |
| `Make_Clusters` | Moderate (~3 min) | Re-run when CHM peaks or crown radius changes |
| `Split_clusters` | No | Fast; re-run freely |
| `Cluster_accuracy` | No | Appends one row to results CSV |
| `Train_Model` | No | Usually < 1 minute |
