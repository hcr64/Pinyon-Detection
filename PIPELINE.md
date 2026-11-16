# Pipeline Overview

Stage-by-stage walkthrough of the Pinyon-Detection pipeline in execution order: the function responsible for each step, its inputs and outputs, every tunable parameter, and notes on why things work the way they do. This file replaces the old `clustering/README.md` and `modelling/README.md`.

The pipeline has two independent halves:

- **Clustering** (`clustering/run_clustering.py`): expensive. Point cloud → CHM → watershed crowns → filtering/splitting → features → GPS label matching. Swept over SLURM array jobs to maximise `matching_score`.
- **Modelling** (`modelling/train_model.py`): cheap. Reads the dataframes clustering saved and trains species / drought classifiers. Runs in seconds to a few minutes and never touches the point cloud (except `--embeddings`).

Modelling requires clustering to have completed at least once for the trial, so `df_clusters` already has its `Name` column.

---

## Stage Map

```
Raw .las files (PixMapper4D export) + PixMapper4D DSM (.tif) + GPS label CSVs
      │
══════╪════════════════════  clustering/run_clustering.py  ═══════════════════
      ▼
┌──────────────────────┐
│ 1. Load Point Cloud  │  las_folder_to_pointcloud()
└──────────────────────┘
      │  raw Open3D PointCloud (absolute Z, UTM metres, EPSG:26912)
      ▼
┌──────────────────────┐
│ 2. Build CHM         │  build_chm_from_external_dsm()
└──────────────────────┘
      │  CHM raster (float32 GeoTIFF) + Affine transform
      ▼
┌──────────────────────┐
│ 3. Find CHM Peaks    │  find_chm_peaks()
└──────────────────────┘
      │  (N, 2) UTM peak coordinates, one per candidate tree top
      ▼
┌──────────────────────┐
│ 4. Segment Crowns    │  cluster_by_chm_peaks()  (watershed)
└──────────────────────┘
      │  raw clusters (one PointCloud per watershed region)
      ▼
┌──────────────────────┐
│ 5. Filter Clusters   │  height filter → filter_clusters_by_green_crown()
└──────────────────────┘
      ▼
┌──────────────────────┐
│ 6. Strip Ground      │  strip_ground_from_clusters()
└──────────────────────┘
      ▼
┌──────────────────────┐
│ 7. Split Large       │  split_large_clusters()  (density peaks + Mean Shift)
│    Clusters          │
└──────────────────────┘
      ▼
┌──────────────────────┐
│ 8. Final Shape Filter│  filter_cluster()
└──────────────────────┘
      ▼
┌──────────────────────┐
│ 9. Extract Features  │  clusters_to_dataframe() + make_deep_dataframe()
│                      │  + engineer_features()
└──────────────────────┘
      │  df_clusters (position/geometry) + df_deep_clusters (shape/colour/PCA)
      ▼
┌──────────────────────┐
│ 10. Match GPS Labels │  match_labels_to_clusters()
└──────────────────────┘
      │  df_clusters + Name / label_distance / drought_class / drought_level
      ▼
┌──────────────────────┐
│ 11. Save + Log       │  save_clusters / save_labeled_clusters / save_dataframes
└──────────────────────┘                 + one row appended to the results CSV

══════╪════════════════════  modelling/train_model.py  ════════════════════════
      ▼
┌──────────────────────┐
│ 12. Load + Prepare   │  load_dataframes(), regenerate missing features,
│     Features         │  select_best_features_rf(), optional embeddings
└──────────────────────┘
      ▼
┌──────────────────────┐
│ 13. Train Classifiers│  train_tree_classifier()  or  --advanced pipeline
│                      │  + --drought  (tolerant vs susceptible pinyons)
└──────────────────────┘
      ▼
┌──────────────────────┐
│ 14. Diagnostics      │  inspect_misclassified_clusters(), reality checks
└──────────────────────┘
      ▼
┌──────────────────────┐
│ 15. Outputs          │  plot_prediction_map(), save_predictions()
└──────────────────────┘
```

---

## Running it

```bash
# environment (once)
bash shells/setup_venv.sh
source shells/access_venv.sh            # module load miniforge3 + conda activate open3d_env

# clustering: single run with hand-tuned defaults
sbatch clustering/shells/pinyons.sh -m "notes about this run"

# clustering: parameter sweep (one array task per row of clustering/params.txt)
sbatch clustering/shells/pinyon_sweep.sh -m "notes about this sweep"

# modelling
sbatch modelling/shells/train_model.sh

# everything, start to finish, with plots collected into run_outputs/
sbatch shells/run_full_pipeline.sh
```

`-m` writes a notes line into the results CSV preamble (like `git commit -m`).

Direct invocation:

```bash
python -u clustering/run_clustering.py \
    --max_radius 4.25 --max_distance 4.0 --min_points 300 \
    --min_peak_distance 3.5 --k 400 --min_height 0.4 \
    --search_radius_m 2.5 --smooth_sigma 3 \
    --job_id test --trial_name Sunset_sfm_trial

python -u modelling/train_model.py --trial_name Sunset_sfm_trial --advanced --drought
```

---

# Part 1: Clustering

## Stage details

### 1. Load Point Cloud
**Function:** `io/las_folder_to_pointcloud.py` → `las_folder_to_pointcloud()`

Reads every `.las` file in `PATHS['Data']`, merges them into one Open3D `PointCloud`, and normalises RGB from 16-bit (0–65535) to float (0.0–1.0). Optionally voxel-downsamples (`voxel_size`).

- Only runs when `STEPS['Load_Pointcloud']` and `STEPS['Make_Raw_Clusters']` are both true. Otherwise the cloud is read from `pointclouds/raw_pcd.ply`.
- With `SAVE=True` the raw cloud is written to `pointclouds/raw_pcd.ply`.
- Coordinates are UTM metres (EPSG:26912, zone 12N for Sunset Crater).
- Slow (~10 min). Only needed when the raw `.las` data changes.

### 2. Build Canopy Height Model (CHM)
**Function:** `preprocessing/build_chm.py` → `build_chm_from_external_dsm()`

The CHM now comes from a **PixMapper4D DSM** (`PATHS['DSM']`, native ~3.67 cm/px) rather than being rasterised from the point cloud.

- **DSM**: read from the external GeoTIFF.
- **DTM**: if no `dtm_path` is given, computed from the point cloud at a coarse grid (`ground_grid_resolution=0.5` m, `ground_percentile=5`) and bilinearly resampled onto the DSM grid. The coarse grid matters: computing percentiles directly on a 3–4 cm grid finer than the (8 cm voxel) point spacing gives near-empty cells and single-point "percentiles".
- **CHM** = DSM − DTM, clipped to [0, ∞); NaN where either is missing.
- Saved as float32 LZW GeoTIFF to `CFMs/chm.tif` (when `SAVE=True`). The Affine transform is returned so pixel indices convert back to UTM anywhere downstream. `CHM_RESOLUTION = abs(transform.a)` is derived from it.

The point cloud is voxel-downsampled (`voxel_size`) before the DTM computation. `build_chm()` (rasterise-from-cloud) and `normalize_heights_by_ground()` still exist in the same file but are not called by `run_clustering.py`.

### 3. Find CHM Peaks
**Function:** `detection/find_chm_peaks.py` → `find_chm_peaks()`

1. Gaussian-smooth the CHM (`smooth_sigma`, in pixels; NaNs zeroed first and re-masked after).
2. `scipy.ndimage.maximum_filter` with a window of `2 * ceil(search_radius_m / resolution) + 1` pixels.
3. A pixel is a peak if it equals the local max and is ≥ `min_height` (checked on the *smoothed* CHM).
4. Flat-topped regions are collapsed to their centroid; pixels → UTM via the transform.

Returns `(N, 2)` peak coordinates and `(N,)` heights (read from the unsmoothed CHM). Peaks become the watershed markers in Stage 4.

`analysis/compute_fine_peaks.py` + `shells/fine_peaks.sh` compute and save peak sets at several `smooth_sigma` values for comparison without re-running everything.

### 4. Segment Crowns (Watershed)
**Function:** `detection/cluster_by_chm_peaks.py` → `cluster_by_chm_peaks()`

Marker-controlled watershed on the CHM, then each 3D point is assigned to its watershed region.

1. Convert peak UTM coords to raster pixel indices.
2. Build a marker image (one integer per peak) and dilate with a 3×3 footprint.
3. Run `skimage.segmentation.watershed` on the *inverted* CHM, masked to cells with CHM > `mask_chm_filled` (0.3 m) so flooding doesn't bleed into bare ground.
4. Look up each point's XY → raster label.
5. Hard cap: points further than `crown_radius` from any peak get label 0.
6. Build one `PointCloud` per label; drop clusters below `min_points`.

> **Why watershed instead of radius assignment?** It respects the ridgeline between adjacent crowns, handles non-circular crowns, and stops one large tree stealing points from a smaller neighbour.

With `SAVE=True` the raw clusters are written to `PATHS['Raw_Clusters']`; when `Make_Raw_Clusters` is false they are loaded from there instead.

### 5. Filter Clusters
Two filters run back to back in `run_clustering.py` to de-clutter non-tree clusters:

- **Height filter**: Z range must exceed `min_cluster_height` (default 1.5 m). Drops flat ground patches.
- **Green-crown filter** (`detection/filter_green_crown.py` → `filter_clusters_by_green_crown()`): ranks points by Excess Green (ExG = 2g − r − b), takes the top 20 %, and requires their mean ExG ≥ `min_exg`. Rock/bare-ground clusters that slipped through have weak top-greenness. Clusters without colour pass automatically.

There is no whole-cloud green filter in the current pipeline (`clean_up_pointcloud` lives in `legacy_files/`); vegetation filtering happens per cluster here.

### 6. Strip Ground from Clusters
**Function:** `preprocessing/strip_ground_from_clusters.py` → `strip_ground_from_clusters()`

Per cluster, estimates local ground as the 10th percentile of Z and discards points below `ground + 0.5 m`. Per-cluster rather than global so trees on the cinder cone's slope each get their own ground reference. Clusters left with < 5 points are dropped. Both values are hardcoded in `run_clustering.py` (`ground_percentile=10`, `min_height_above_ground=0.5`).

### 7. Split Large Clusters
**Function:** `detection/split_large_clusters.py` → `split_large_clusters()`

Clusters whose XY radius exceeds `max_radius` likely hold two or more merged crowns.

1. **Gate 1**: skip clusters under `min_points`.
2. **Gate 2, size**: skip clusters with radius ≤ `max_radius` (unless listed in `force_check_indices`; not currently wired in).
3. **Gate 3, density peaks**: `find_density_peaks()` uses a KDTree (`k` neighbours) and keeps points that beat all their neighbours and exceed `min_density_ratio` × mean density, with peaks closer than `min_peak_distance` suppressed. Fewer than 2 peaks → still one tree, pass through.
4. **Mean Shift** with `bandwidth = min_peak_distance` finds the number of sub-crowns automatically.
5. **Validation**: each sub-cluster must have ≥ `min_points` and pass `filter_cluster()` (height ≥ 1.0, radius ≥ 0.3). Failing sub-clusters are discarded individually; the split is reverted only if none survive.

With `use_color=True`, both the density-peak gate and Mean Shift run on XY + `color_weight` × shadow-robust chromaticity instead of XY alone, aimed at overlapping crowns of different colour. Colour splitting has not shown a clear matching-score gain in sweeps so far; it's the focus of the current "separate overlapping trees" work.

Pre-split clusters are saved to `pre_split_clusters/` for inspection (when `SAVE=True`).

> **Why Mean Shift instead of KMeans?** KMeans needs `n_clusters` up front; Mean Shift finds it from point density, which matters when the number of merged trees varies.

### 8. Final Shape Filter
`filter_cluster(min_height=1.0, min_radius=0.3)` once more over everything after splitting. Clusters under 1 m tall or 0.3 m radius are not trees.

### 9. Extract Features
**Functions:** `global_files/get_pointcloud_stats.py` → `clusters_to_dataframe()`,
`global_files/get_deep_cluster_features.py` → `make_deep_dataframe()` + `engineer_features()`

**`df_clusters`** (GPS matching and scoring):

| Column | Description |
|---|---|
| `file` | Cluster's integer index in the cluster list (join key everywhere) |
| `n_points` | Point count |
| `height` | Z range of AABB |
| `width_x`, `width_y`, `radius` | AABB extents; radius = max half-width |
| `x_pos`, `y_pos` | Density-weighted XY centroid (uses `k` neighbours), pulled toward the trunk rather than sparse crown edges |
| `density` | Points per m³ |

**`df_deep_clusters`** (classifier input). Groups:

| Group | Features |
|---|---|
| Shape | `height`, `radius`, `n_points`, `obb_extent_x/y/z` |
| PCA | `eigenvalue_1/2/3`, `linearity`, `planarity`, `sphericity` |
| Colour | `mean_r/g/b`, `std_r/g/b` |
| Structural | `verticality`, `flatness_ratio`, `crown_base_ratio` |
| Footprint | `hull_area_ratio`, `hull_volume_ratio`, `obb_aspect_ratio`, `n_density_peaks` |
| Vertical/radial profile | `z_skewness`, `z_kurtosis`, `trunk_gap_score`, `radial_density_gradient` |
| Colour, reframed | `hue_mean`, `hue_std`, `saturation_mean`, `bark_fraction` |
| Chromaticity (shadow-robust) | `chroma_r/g/b`, `chroma_g_std`, `exg_chroma` |
| Colour extremes (top 20 %) | `top_red_exr`, `top_blue_exb`, `top_yellow_exy`, `top_grey` |
| Engineered ratios | `height_to_radius`, `green_dominance`, `crown_volume`, `color_saturation` |

Not every computed feature is used by the models; see the `FEATURES` list in Part 2.

### 10. Match GPS Labels
**Function:** `labeling/match_labels_to_clusters.py` → `match_labels_to_clusters()`

1. Load labels from `labels/sunset-sfm.csv` and `labels/sunsetCraterMay26.csv`, merged. Names containing "dead" or "point" are dropped; species normalised to `pinyon` / `juniper` / `ponderosa` (rows like "pinyon juniper" fall out here).
2. **Drought tags**: a trailing `T` or `S`, optionally followed by 1–10 (e.g. `pinyon T4`), is parsed into `drought_class` (`tolerant` / `susceptible`) and `drought_level`. Lower number = more tolerant, higher = more susceptible. Only pinyons carry tags.
3. WGS84 lat/lon → UTM 12N (`pyproj`).
4. **Coverage filter**: GPS points with no cluster within 10 m are dropped (outside the irregular drone footprint).
5. **Matching score**: fraction of GPS points with exactly one cluster centroid within `max_distance` (KDTree ball query). Also returns `match_counts` (`n_gps`, `n_perfect`, `n_no_match`, `n_multi`).
6. **Assignment**: Hungarian algorithm (`linear_sum_assignment`) on a Gaussian cost `1 − exp(−d² / 2σ²)` with `σ = gps_sigma`. Assignments further than `max_distance` are rejected.
7. Adds `Name`, `label_distance`, `drought_class`, `drought_level` to `df_clusters` (`"unknown"` / `inf` where unmatched).

Diagnostics:
- `coordinate_overlap.png`: GPS points coloured perfect / no match / multiple.
- `GPS_Clusters.png`: species scatter over cluster centroids.
- `save_multimatch_clusters()`: prints and saves the clusters behind every ambiguous GPS point (`multi_match_clusters/`) plus a stats CSV (`mm_clusters_stats.csv`). The printed `smallest/largest n_points` ratio hints at fragment-of-one-tree (lopsided) vs two real crowns (similar).
- `diagnose_gps_losses()`: per-GPS-point nearest-cluster distance and cluster count after each stage, and which stage lost it. Currently disabled (`stage_clusters=None` in `run_clustering.py`); the `stages` dict is already built, so enabling it is a one-line change.
- `analysis/estimate_gps_offset.py`: checks for a systematic GPS offset using `gps_loss_diagnosis.csv`.

### 11. Save + Log
- `save_clusters()`: `cluster0.ply, cluster1.ply, …` (folder cleared first; indices match `df_clusters["file"]`).
- `save_labeled_clusters()`: GPS-labeled clusters only, as `pinyon_0.ply`, `juniper_0.ply`, …
- `save_dataframes()`: `clusters.csv` and `deep_clusters.csv`.
- Every run (sweep or single) appends one row to `RESULTS_PATH` via a mkdir-based lock (safe for parallel array tasks). Columns: `task`, all of `RESULT_KEYS` (see parameter table), `matching_score`, `n_gps`, `n_perfect`, `n_no_match`, `n_multi`.
- Labeled clusters can be synced to Google Drive with `rclone` (see `legacy_files/save_clusters_to_drive.sh`).

With `SAVE=False` (use for sweeps) none of the `.ply` / dataframe / CHM writes happen; only the results row is logged.

---

## Clustering parameters

CLI args of `run_clustering.py`. "Pinyons.sh" is the hand-tuned single-run default; "default" is the argparse default when a flag is omitted.

### Live parameters (swept)

| Param | pinyons.sh | Default | Stage | Meaning / findings |
|---|---|---|---|---|
| `max_radius` | 4.25 | required | 7 | XY radius (m) above which a cluster is checked for splitting. 3.5–5.0 all within noise. |
| `max_distance` | 4.0 | required | 10 | GPS-to-cluster radius (m) for both the matching score and accepting an assignment. 4.0 beat 3.5 (more no-matches) and 4.5 (more multi-matches). |
| `min_points` | 300 | required | 4, 7 | Min points per cluster/sub-cluster. 100–300 all ≈ 0.818–0.822; 800 is worse. |
| `min_peak_distance` | 3.5 | required | 7 | Density-peak suppression distance and Mean Shift bandwidth (≈ min trunk-to-trunk distance, m). 3.0–4.0 fine; 2.0 worse. |
| `k` | 400 | required | 7, 9 | Neighbour count for density peaks (split) and for the density-weighted centroid. Effect on score is small. |
| `min_height` | 0.4 | required | 3 | Min smoothed CHM height (m) for a peak. 0.3–0.5 equivalent; 0.75 slightly worse. |
| `search_radius_m` | 2.5 | required | 3 | Local-max window radius (m). **The most sensitive parameter.** 2.5 is best; 2.0 and 3.0+ both drop ≈ 0.03–0.05. |
| `smooth_sigma` | 3 | required | 3 | Gaussian σ (px, native grid) before peak detection. 2–4 equivalent. |
| `crown_radius` | not passed | 8.0 | 4 | Hard cap on point-to-nearest-peak distance (m). 5–10 made no difference. |
| `min_density_ratio` | not passed | 1.5 | 7 | A density peak must be this many × the cluster mean density. 1.2–2.5 no real effect. |
| `use_color` | not passed | False | 7 | Split on XY + chromaticity. No clear gain yet. |
| `color_weight` | not passed | 10.0 | 7 | Metres-equivalent weight of chromaticity when `use_color=True`. Sweeps of 5–40 showed no improvement. |
| `min_exg` | not passed | 0.05 | 5 | Min mean ExG of a cluster's greenest 20 %. 0.03–0.08 no real effect. |
| `min_cluster_height` | not passed | 1.5 | 5 | Min cluster Z range (m). |

### Dead / low-impact parameters
Kept so older shell scripts still work; none move `matching_score`.

| Param | Default | Notes |
|---|---|---|
| `eps` | 2.0 | DBSCAN epsilon. Unused since the CHM/watershed method. |
| `green_threshold` | 0.025 | Whole-cloud green filter margin. `clean_up_pointcloud` is no longer in the pipeline. |
| `voxel_size` | 0.08 | Voxel downsample size (m) for the raw cloud and DTM computation. |
| `gps_sigma` | 4.0 | Only reshapes the Hungarian cost used for *species assignment*; the matching score is a raw-distance KDTree query, so sweeping this doesn't move it. Watch the multi-match diagnostics instead. |

### Fixed in code
`mask_chm_filled = 0.3` (watershed mask), `ground_percentile = 10` and `min_height_above_ground = 0.5` (ground strip), green-crown `top_fraction = 0.20`, final `filter_cluster(min_height=1.0, min_radius=0.3)`, GPS `coverage_radius = 10.0`.

### Technical (non-swept) args
`--trial_name` (required), `--job_id` (required), `--notes`, `--results_path`, `--save` (True/False).

### Sweep file format
`clustering/params.txt`: one row per SLURM array task, 14 whitespace-separated columns, `#` comments allowed:

```
max_radius crown_radius max_distance min_points min_peak_distance k min_density_ratio min_height search_radius_m smooth_sigma use_color color_weight min_exg min_cluster_height
```

Update `#SBATCH --array=1-N` in `pinyon_sweep.sh` to match the number of rows. `use_color` accepts `1/0/True/False`. `pinyon_sweep.sh` runs only clustering; it never trains a classifier. It also runs `shells/clear_garbage_files.sh` to move finished jobs' logs out of the log folders.

Sweep results land in `trial_data/<trial>/GPS_results/sweep_<array_job_id>.csv`; single runs in `job_<job_id>.csv`.

### Matching score history
Best recorded on Sunset Crater: **~0.835** (July runs, older pipeline). Recent sweeps on the current watershed pipeline plateau around **0.81–0.82** (e.g. 193/236 perfect, ~31 no-match, ~12 multi). Remaining losses are mostly no-matches (GPS error / missed crowns) and overlapping-tree multi-matches.

---

# Part 2: Modelling

## Stage details

### 12. Load + Prepare Features
**Entry point:** `modelling/train_model.py`

1. `load_dataframes()` reads `clusters.csv` / `deep_clusters.csv`. Aborts if `df_clusters` has no `Name` column.
2. Stale `emb_*` columns are stripped from the cached CSV so embeddings never accumulate.
3. If any column in `FEATURES` is missing from the cache, raw clusters are reloaded and `make_deep_dataframe()` + `engineer_features()` are re-run (no clustering or GPS matching), then saved. This is how you add a new feature without a full clustering run.
4. `select_best_features_rf()` ranks features by cross-validated **permutation importance** (on held-out folds, scored on macro F1) and proposes a subset above the mean importance. It also compares full vs selected features including **juniper recall**, so a selection that trims noise but hurts juniper is visible.
5. **`--embeddings`**: trains (or loads from `PATHS['Models']/pointnet_ae.pt`) a small PointNet autoencoder on all clusters (self-supervised, no labels), extracts a 64-d embedding per cluster, and merges `emb_*` columns onto `df_deep_clusters`. `--retrain_embeddings` forces retraining. See `pointcloud_autoencoder.py` for why this was chosen over PointMLP/PointNeXt/etc. (≈ 166–236 labels makes end-to-end deep models a non-starter).
6. `plot_feature_separability()` saves scatter grids and per-feature boxplots by species.

### 13. Train Classifiers

**Default: `train_tree_classifier()`** (`classification/train_tree_classifier.py`)

Compares RF baseline, Gradient Boosting, SVM (RBF), Logistic Regression and KNN on a stratified 80/20 split plus `StratifiedKFold` CV (folds capped at `min(3, min_class_size)` for the tiny ponderosa class). Reports per-class precision/recall/F1, **macro F1** and **Cohen's kappa** (preferred over accuracy given the imbalance), saves a confusion matrix PNG for the best model, and appends `predicted_label` and `prob_<species>` columns for every cluster, labeled or not.

Optional enhancements (`DEFAULT_ENHANCEMENTS`; pass an `enhancements` dict to override):

| Toggle | Default | Effect |
|---|---|---|
| `SMOTE` | False | Synthetic oversampling of juniper/ponderosa inside each CV fold (no leakage). |
| `WEIGHTING` | True | Adds `RF_weighted`: "balanced" class weights with ponderosa ×2. |
| `SELECTION` | True | Adds `RF_selected`: drops features below mean RF importance. |
| `TUNING` | False | Adds `RF_tuned`: `RandomizedSearchCV` (30 iterations) over RF hyperparameters. Slowest. |

**`--advanced`**: `run_advanced_classifiers()` instead (XGBoost, LightGBM, soft-voting RF+GB+SVM, MLP, isotonic-calibrated SVM), followed by:
- `run_label_spreading()` (`alpha=0.2`, `gamma=0.5`, `confidence_threshold=0.80`): RBF similarity graph over all ~3,600 clusters, propagating species labels from the labeled ones. Adds `semi_label` / `semi_confidence`.
- `run_full_evaluation()` (`evaluate_semi_supervised.py`): the *honest* evaluation. Masks each labeled cluster before fitting, compares against an RF baseline on the same folds, runs a degradation curve (does LabelSpreading hold up better as labels shrink?) and a reliability diagram (does 0.80 confidence mean 80 % accuracy?). Trust this over the "labeled-node recovery" printout inside `run_label_spreading()`, which is in-sample.

**`--drought`**: `train_drought_classifier()` trains tolerant vs susceptible on the pinyon-labeled subset that carries a T/S tag (needs ≥ 6 per class; returns `None` otherwise), then predicts `predicted_drought_class` for every cluster predicted as pinyon. `inspect_drought_misclassified_clusters()` runs the stability check for it.

**`--combined`**: 4-class model (juniper / ponderosa / pinyon_tolerant / pinyon_susceptible) in `train_combined_classifier.py`. The call is currently commented out in `train_model.py`.

### 14. Diagnostics
- `inspect_misclassified_clusters()`: out-of-fold predictions over 5 seeds. A cluster misclassified in every seed is a real, stable error worth opening as a `.ply`; one flipped in 1–2 seeds is CV noise. Adds `label_distance` and reality-check flags (low points, low height, extreme height/radius). Saved to `pinyon_misclassification_report.csv`.
- `plot_cluster_reality_check()`: per-species histograms of height, radius, ratios, etc., to confirm labeled clusters look like real trees (is ponderosa taller/narrower than pinyon?). Flags clusters under 20 points or 0.5 m tall.
- `diagnose_cluster_bimodality()`: checks whether suspicious clusters are two overlapping trees that separate by colour but not XY. Fits 1- vs 2-component GMMs per point on chromaticity/height (BIC), then requires the two colour groups to also occupy different parts of the footprint (otherwise it's just sun/shade). Helpers: `calibrate_thresholds()`, `explain_why_not_flagged()`, `get_color_spatial_override_candidates()` (feeds `split_large_clusters(force_check_indices=...)`), `get_flagged_cluster_paths()`. Not called by `train_model.py`; run from a notebook.

### 15. Outputs
- Confidence rule used downstream: a cluster counts as a confirmed pinyon when `predicted_label == "pinyon"` and `prob_pinyon > 0.80`.
- `plot_prediction_map()` (`min_confidence=0.90`): top-down map of crown-sized circles coloured by predicted species (pinyons split by predicted drought class) over a true-colour basemap rasterised from the raw cloud (cached to `topdown_basemap.npz`; optional `ortho_path`; CHM fallback). Clusters below the confidence threshold are drawn as small grey dots; GPS-labeled clusters get a white ring. Saved as `prediction_map.png`.
- `save_predictions()`: `predictions.csv` (labels, probabilities, position, geometry, ground-truth `Name`) for visualising in Jupyter.
- `save_dataframes()` re-saves the dataframes with the new prediction columns.

---

## Modelling parameters

### CLI flags (`train_model.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--trial_name` | required | Trial folder name. |
| `--advanced` | off | Advanced classifiers + label spreading + honest evaluation instead of `train_tree_classifier()`. |
| `--embeddings` | off | Train/load PointNet autoencoder and merge `emb_*` columns. |
| `--retrain_embeddings` | off | Force autoencoder retraining. |
| `--embedding_dim` | 64 | Bottleneck size. Try 32 if loss is unstable, 128 if it plateaus early. |
| `--embedding_n_points` | 256 | Points resampled per cluster. |
| `--embedding_epochs` | 100 | Autoencoder epochs. |
| `--drought` | off | Also train the tolerant/susceptible pinyon classifier. |
| `--combined` | off | Accepted, but its code path is commented out. |

### Hardcoded in `train_model.py`
- Label spreading: `alpha=0.2` (clamping; 0 = labeled nodes fixed, raise toward 0.4 if unlabeled clusters get confidently wrong labels), `gamma=0.5` (RBF bandwidth on standardised features; lower = broader spread), `confidence_threshold=0.80`.
- Misclassification checks: `n_seeds=5`.
- Prediction map: `min_confidence=0.90`.

### `model_sweep.sh` / `model_params.txt`
Array job over classifier settings. Columns: `mode alpha gamma confidence_threshold eval_random_state smote weighting selection tuning`. `advanced` rows use the first five; `supervised` rows use the last four. Unused columns need a placeholder (0).

> This sweep script calls `train_model.py` with flags (`--alpha`, `--gamma`, `--smote`, `--results_path`, …) that the current `train_model.py` doesn't define, so it will fail as written. See Known Quirks.

### Active `FEATURES` (`modelling/functions/feature_config.py`)
Single source of truth for every classifier. Currently used: all shape, PCA, colour, chromaticity, colour-extreme, structural and engineered-ratio features, plus `obb_aspect_ratio`, `n_density_peaks`, `trunk_gap_score`, `radial_density_gradient`, `hue_mean`, `hue_std`, `saturation_mean`, `bark_fraction`.
Currently commented out: `hull_area_ratio`, `hull_volume_ratio`, `z_skewness`, `z_kurtosis`.
`emb_*` columns are not in `FEATURES`, so classifiers ignore embeddings unless you add them.

---

## Data bottleneck

~236 labeled clusters out of ~3,600 total (294 raw GPS recordings, some discarded for entry errors like "pinyon juniper"). **Ponderosa has only ~20 samples.** This has been the binding constraint: no model architecture, oversampling, weighting or feature-engineering approach tried so far has overcome it. More ground-truth labels are the highest-leverage next step.

A second issue is overlapping trees inside one cluster. They corrupt label assignment during matching and model training, and a handful of clusters are consistently misclassified for this reason (and likely hurt drought classification too).

---

## Step flags

`STEPS` in `global_files/constants.py` toggles expensive stages. Outputs are saved to disk and reloaded when a step is off.

| Flag | Cost | Notes |
|---|---|---|
| `Load_Pointcloud` | ~10 min | Only when raw `.las` change. |
| `Make_CHM` | ~2 min | Only when the raw cloud / DSM changes. |
| `Make_Raw_Clusters` | ~3 min+ | Peaks + watershed. Re-run when peak or crown params change. |

Everything after raw clustering (filter, strip, split, features, matching) always runs, since that is what the sweeps tune.

---

## Paths

`get_paths(trial_name)` in `global_files/constants.py`:

| Key | Location | Purpose |
|---|---|---|
| `Data` | scratch `…/<trial>/point_cloud/` | Input `.las` files |
| `DSM` | scratch `…/<trial>/1_dsm/<trial>_dsm.tif` | PixMapper4D DSM |
| `Raw_pcd`, `Cleaned_pcd` | scratch `…/pointclouds/` | Saved clouds |
| `CHM` | scratch `…/CFMs/chm.tif` | CHM GeoTIFF |
| `Raw_Clusters`, `Clusters` | scratch `…/clusters/` | Pre-/post-processing clusters |
| `PS_clusters`, `Labeled_clusters`, `MM_save_path` | scratch | Pre-split, labeled, multi-match `.ply` folders |
| `Labels` | local `labels/sunset-sfm.csv` | GPS labels (plus `sunsetCraterMay26.csv` alongside it) |
| `Images`, `Dataframes`, `Models`, `GPS_results` | local `trial_data/<trial>/…` | Plots, CSV caches, autoencoder checkpoint, sweep results |

Scratch is purged periodically; `pinyons.sh` `touch`es the input `.las` files so they aren't deleted.

---

## Debugging and utilities

```bash
bash legacy_files/debug.sh <JOB_ID>           # print a job's .txt and .err
bash shells/see_recent_jobs.sh          # recent jobs from sacct
bash shells/clear_garbage_files.sh      # move finished jobs' logs to scratch
```

Clustering logs go to `clustering/logs/`, modelling logs to `modelling/logs/`.

---

## Known quirks

Real current behaviours, worth knowing before you spend a sweep on them.

**Clustering**

- **`run_full_pipeline.sh` is out of sync.** Its parameters (`MIN_POINTS=40`, `K=40`, `MIN_HEIGHT=0.5`, `MIN_PEAK_DISTANCE=3.0`) are the old values, not the current `pinyons.sh` ones.
- **`gps_sigma` doesn't move `matching_score`** (see above).
- **`max_distance` plays two roles**: score radius and assignment cutoff. Raising it can trade no-matches for multi-matches rather than genuinely improving matching.
- **Height normalisation isn't wired in.** `normalize_heights_by_ground()` exists but nothing calls it (it hurt scores at Sunset Crater).

**Modelling**
- **`train_tree_classifier()` ignores its `features` argument.** `train_model.py` passes `selected_features`, but inside the function `available` is built from `FEATURES`, not `features`, so the permutation-importance selection never restricts the models. Only the function's own `SELECTION` enhancement (`RF_selected`) reduces features. One-line fix: build `available` from `features`.
- **`model_sweep.sh` is stale** relative to `train_model.py` (flags it passes don't exist).
- Embeddings aren't used by any classifier unless `emb_*` are added to `FEATURES`.
