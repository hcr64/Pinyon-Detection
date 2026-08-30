# Pinyon-Detection — Clustering & Labeling

This is the expensive half of the pipeline: raw `.las` files → Canopy Height
Model → watershed-segmented tree clusters → Filter likely non-tree clusters → GPS-labeled clusters. Everything
here runs via `run_clustering.py` and is meant to be swept over SLURM array
jobs to optimize `matching_score`.

To decultter the amount of tree clusters, clusters that are lilely not trees get filtered out. Tree clusters with low total points, clusters with few green points, and any clusters shorter than 1m are removed to make sure labels are placed on probable trees. 

If you're looking for the classifier (species prediction on labeled
clusters), that lives in `../modelling/` — see `../modelling/README.md`.

---

## Structure

```
clustering/
├── run_clustering.py         # entry point — clustering + GPS labeling only
├── constants.py              # STEPS flags, get_paths(trial_name) builder
│
├── functions/
│   ├── io/                   # las loading, cluster save/load, dataframe save/load
│   ├── preprocessing/        # green filter, CHM building, height normalization,
│   │                         # per-cluster ground stripping
│   ├── detection/            # CHM peak finding, watershed segmentation,
│   │                         # Mean-Shift cluster splitting, green-crown filter
│   └── labeling/             # GPS label matching (Hungarian assignment),
│                              # matching_score, multi-match diagnostics
│
├── shells/
│   ├── pinyons.sh             # single run with hand-tuned defaults
│   ├── pinyon_sweep.sh        # parameter sweep (SLURM array job)
│   ├── save_clusters_to_drive.sh
│   └── save_mm_clusters_to_drive.sh
│
└── parameters/
    └── params.txt             # sweep grid — one row per array task
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
python -u run_clustering.py \
    --eps               2.0 \
    --green_threshold   0.025 \
    --max_radius        3.0 \
    --max_distance      4.0 \
    --min_points        200 \
    --voxel_size        0.08 \
    --min_peak_distance 3.0 \
    --k                 40 \
    --min_height        1.0 \
    --search_radius_m   3.0 \
    --gps_sigma         4.0 \
    --smooth_sigma      1.0 \
    --job_id            test \
    --trial_name        Sunset_sfm_trial
```

There are more extensive variable descriptions in `pinyons.sh`.

This produces:
- Segmented clusters → `trial_data/<trial_name>/clusters/`
- GPS-labeled clusters → `trial_data/<trial_name>/labeled_clusters/`
- Feature dataframes (labeled) → `trial_data/<trial_name>/dataframes/`
- Diagnostic plots (GPS overlap, species scatter) → `trial_data/<trial_name>/images/`
- One row appended to the results CSV at `PATHS['GPS_results']`

### Parameter sweep

Edit `parameters/params.txt` (one row per array task, 12 columns — see below),
update `--array` in `pinyon_sweep.sh` to match the row count, then:

```bash
sbatch shells/pinyon_sweep.sh
```

`pinyon_sweep.sh` only calls `run_clustering.py` — it never touches the
classifier. That comparison used to run on every array task via the old
`main.py` and mostly just cluttered sweep logs, since the labeled set doesn't
change between sweep iterations. 

`clear_garbage_files.sh` removes all non-running `.out` and `.err` files in the log folder, and runs when a sweep starts.

---

## `params.txt` column format

```
eps  green_threshold  max_radius  max_distance  min_points  voxel_size  min_peak_distance  k  min_height  search_radius_m  gps_sigma  smooth_sigma
```

| # | Column | Meaning |
|---|---|---|
| 1 | `eps` | DBSCAN epsilon — unused with the CHM/watershed method, keep at 2.0 (Not used in CHM workflow) |
| 2 | `green_threshold` | Green-channel dominance margin for vegetation filter (Not used in CHM workflow) |
| 3 | `max_radius` | Crown radius cap in `cluster_by_chm_peaks` (metres) |
| 4 | `max_distance` | Max GPS-to-cluster distance accepted in label matching (metres) |
| 5 | `min_points` | Minimum points per valid cluster |
| 6 | `voxel_size` | Downsampling cell size (metres) |
| 7 | `min_peak_distance` | Min trunk-to-trunk distance for cluster splitting (metres) |
| 8 | `k` | Minimum amount of points in a centroid when splitting large clusters into two. |
| 9 | `min_height` | Minimum CHM height to count as a tree peak (metres) |
| 10 | `search_radius_m` | Local-max window radius for peak detection (metres) |
| 11 | `gps_sigma` | Assumed 1-sigma GPS error (metres) for the Gaussian label-matching cost |
| 12 | `smooth_sigma` | Gaussian smoothing (px) applied to CHM before peak detection |

---

## Matching Score

Reported after every run: the fraction of GPS-labeled trees with exactly one
cluster within `max_distance`.

(Numbers below are an example, not up to date)
```
Perfect matches (1:1):  58  (82.9%)
No match:                8  (11.4%)
Multiple clusters:        4   (5.7%)
Matching score:         0.829
```

Best achieved on Sunset Crater so far: **~0.83**.

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

## Debugging

```bash
bash shells/debug.sh <JOB_ID>        # (top-level shells/)
  -prints .err and .txt files from the job 

bash shells/see_recent_jobs.sh       # (top-level shells/)
  -prints recently completed jobs in a time frame, useful for looking back at jobs after closing terminal/IDE
```