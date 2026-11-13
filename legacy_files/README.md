## Detecting Drought Resistant Pinyon Pines using Aerial Structure-from-Motion

# Clustering & Labeling

This is the expensive half of the pipeline: raw `.las` files -> Canopy Height
Model -> watershed-segmented tree clusters -> Filter likely non-tree clusters -> GPS-labeled clusters. Everything
here runs via `run_clustering.py` and is meant to be swept over SLURM array
jobs to optimize `matching_score`.

To decultter the amount of tree clusters, clusters that are likely not trees get filtered out. Tree clusters with low total points, clusters with few green points, and any clusters shorter than 1m are removed to make sure labels are placed on probable trees. 

The predictive side of the project is in `../modelling/` with it's own README. — see `../modelling/README.md`.

---

## Structure
(may not be completely up to date)
```
clustering/
├── run_clustering.py         # entry point — clustering + GPS labeling only
├── compute_fine_peaks.py     # precomputes CHM peaks at several smoothing levels
├── params.txt                # sweep grid — one row per array task
│
├── functions/
│   ├── io/                   # las_folder_to_pointcloud, save_labeled_clusters
│   ├── preprocessing/        # CHM building (build_chm, build_chm_from_external_dsm,
│   │                         #   load_chm), normalize_heights_by_ground,
│   │                         #   per-cluster ground stripping
│   ├── detection/            # CHM peak finding, watershed segmentation,
│   │                         #   colour-augmented Mean-Shift splitting,
│   │                         #   green-crown filter
│   └── labeling/             # GPS label matching (Hungarian assignment),
│                             #   matching_score, multi-match + GPS-loss diagnostics
│
├── shells/
│   ├── pinyons.sh            # single run with hand-tuned defaults
│   ├── pinyon_sweep.sh       # parameter sweep (SLURM array job)
│   └── fine_peaks.sh         # runs compute_fine_peaks.py for several sigmas
│
├── logs/                     # SLURM stdout/stderr (job_%j.* and sweep_%A_%a.*)
└── trial_data/<trial_name>/  # small, local outputs (see "Where outputs go")
```

---

## Quick start

### Single run

Edit the parameter block (which will be set to the optimal values in the GitHub) at the top of `shells/pinyons.sh`, then on the command line:

```bash
sbatch shells/pinyons.sh
```

Or run directly:

```bash
python -u clustering/run_clustering.py \
    --max_radius        4.25 \
    --max_distance      4.0 \
    --min_points        300 \
    --min_peak_distance 3.5 \
    --k                 400 \
    --min_height        0.4 \
    --search_radius_m   2.5 \
    --smooth_sigma      3 \
    --job_id            test \
    --trial_name        Sunset_sfm_trial
```
All other variables not passed through are set as defaults in `PARAMS`, the ones above are sweepable and are proven to effect clusters. Other parameters are currently unused but still wired in.
There are more extensive variable descriptions in `pinyons.sh`.

This produces:
- Segmented clusters → `/scratch/[USER]/Pinyon-Detection/trial_data/<trial_name>/clusters/`
- GPS-labeled clusters → `/scratch/[USER]/Pinyon-Detection/trial_data/<trial_name>/labeled_clusters/`
- Feature dataframes (labeled) → `trial_data/<trial_name>/dataframes/`
- Diagnostic plots (GPS overlap, species scatter) → `trial_data/<trial_name>/images/`
- One row appended to the results CSV at `PATHS['GPS_results']`

### Parameter sweep

Edit `clustering/params.txt` (one row per array task, 12 columns — see below),
update `--array` in `pinyon_sweep.sh` to match the row count, then:

```bash
sbatch clustering/shells/pinyon_sweep.sh
```

`pinyon_sweep.sh` only calls `run_clustering.py` — it never touches the
classifier. That comparison used to run on every array task via the old
`main.py` and mostly just cluttered sweep logs, since the labeled set doesn't
change between sweep iterations. 

`clear_garbage_files.sh` removes all non-running `.out` and `.err` files in the log folder, and runs when a sweep starts.

---

## `params.txt` column format

```
max_radius  crown_radius  max_distance  min_points  min_peak_distance  k  min_density_ratio  min_height  search_radius_m  smooth_sigma  use_color  color_weight  min_exg  min_cluster_height
```

| # | Column | Meaning |
|---|---|---|
| 1 | `max_radius` | XY radius (m) above which a cluster is a candidate for splitting in `split_large_clusters` |
| 2 | `crown_radius` | Hard cap (m) on point-to-nearest-peak distance in `cluster_by_chm_peaks` |
| 3 | `max_distance` | Max GPS-to-cluster distance accepted in label matching (m) |
| 4 | `min_points` | Minimum points per valid cluster (also applied to sub-clusters after splitting) |
| 5 | `min_peak_distance` | Mean Shift bandwidth and minimum density-peak separation when splitting (m; operates in XY + colour space when `use_color` is on) |
| 6 | `k` | Neighbour count for local density — used by the split density-peak gate and by the density-weighted centroid that GPS matching measures against |
| 7 | `min_density_ratio` | A split peak must be this many times denser than the cluster mean |
| 8 | `min_height` | Minimum CHM height to count as a tree peak (m) |
| 9 | `search_radius_m` | Local-max window radius for peak detection (m) |
| 10 | `smooth_sigma` | Gaussian smoothing (px) applied to the CHM before peak detection |
| 11 | `use_color` | Split on XY + chromaticity instead of XY only (`True`/`False`/`1`/`0`) |
| 12 | `color_weight` | "Metres-equivalent" scale on chromaticity when `use_color` is on |
| 13 | `min_exg` | Minimum mean ExG of a cluster's greenest 20% of points (green-crown filter) |
| 14 | `min_cluster_height` | Minimum Z range (m) of a raw cluster |

---

## Known Quirks

These are current, real behaviors of the code — not necessarily bugs to fix
blindly, but worth knowing before you spend a sweep on them:

- **`gps_sigma` doesn't move `matching_score`.** `calculate_matching_score()`
  scores purely via a KDTree ball-query on raw distance — `gps_sigma` only
  reweights the Hungarian cost matrix used for *species label assignment*.
  Sweeping it will change which species get assigned in ambiguous
  multi-match cases, not the matching score itself. Watch the multi-match
  diagnostic output (`multi_match_clusters/`), not `matching_score`, when
  tuning this.
- **Height normalization is not currently wired in.** `normalize_heights_by_ground()`
  still exists in `preprocessing/build_chm.py` but nothing in
  `run_clustering.py` calls it — heights are used as-is (absolute UTM Z)
  throughout. This matches the finding that normalization hurt matching
  scores at Sunset Crater; re-enable by calling it explicitly inside the
  `Clean_Pointcloud` block if you want to test it again.


---
