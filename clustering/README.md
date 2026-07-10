# Pinyon-Detection — Clustering & Labeling

This is the expensive half of the pipeline: raw `.las` files → Canopy Height
Model → watershed-segmented tree clusters → GPS-labeled clusters. Everything
here runs via `run_clustering.py` and is meant to be swept over SLURM array
jobs to optimize `matching_score`.

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

Edit the parameter block at the top of `shells/pinyons.sh`, then:

```bash
sbatch shells/pinyons.sh
```

Or run directly:

```bash
python -u run_clustering.py \
    --eps 2.0 \
    --green_threshold 0.025 \
    --max_radius 3.0 \
    --max_distance 4.0 \
    --min_points 200 \
    --voxel_size 0.08 \
    --min_peak_distance 3.0 \
    --k 40 \
    --min_height 1.0 \
    --search_radius_m 3.0 \
    --gps_sigma 4.0 \
    --smooth_sigma 1.0 \
    --job_id test \
    --trial_name Sunset_sfm_trial
```

This produces:
- Segmented clusters → `trial_data/<trial>/clusters/`
- GPS-labeled clusters → `trial_data/<trial>/labeled_clusters/`
- Feature dataframes (labeled) → `trial_data/<trial>/dataframes/`
- Diagnostic plots (GPS overlap, species scatter) → `trial_data/<trial>/images/`
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

---

## `params.txt` column format

```
eps  green_threshold  max_radius  max_distance  min_points  voxel_size  min_peak_distance  k  min_height  search_radius_m  gps_sigma  smooth_sigma
```

| # | Column | Meaning |
|---|---|---|
| 1 | `eps` | DBSCAN epsilon — unused with the CHM/watershed method, keep at 2.0 |
| 2 | `green_threshold` | Green-channel dominance margin for vegetation filter |
| 3 | `max_radius` | Crown radius cap in `cluster_by_chm_peaks` (metres) |
| 4 | `max_distance` | Max GPS-to-cluster distance accepted in label matching (metres) |
| 5 | `min_points` | Minimum points per valid cluster |
| 6 | `voxel_size` | Downsampling cell size (metres) |
| 7 | `min_peak_distance` | Min trunk-to-trunk distance for cluster splitting (metres) |
| 8 | `k` | **Currently has no effect** — see Known Quirks below |
| 9 | `min_height` | Minimum CHM height to count as a tree peak (metres) |
| 10 | `search_radius_m` | Local-max window radius for peak detection (metres) |
| 11 | `gps_sigma` | Assumed 1-sigma GPS error (metres) for the Gaussian label-matching cost |
| 12 | `smooth_sigma` | Gaussian smoothing (px) applied to CHM before peak detection |

---

## Matching Score

Reported after every run: the fraction of GPS-labeled trees with exactly one
cluster within `max_distance`.

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

- **`k` is silently overwritten.** `run_clustering.py` sets `K = MIN_POINTS`
  right after parsing args, so whatever you pass via `--k` (or column 8 in
  `params.txt`) is discarded. If you want to sweep `k` independently, this
  line needs to change first.
- **`gps_sigma` doesn't move `matching_score`.** `calculate_matching_score()`
  scores purely via a KDTree ball-query on raw distance — `gps_sigma` only
  reweights the Hungarian cost matrix used for *species label assignment*.
  Sweeping it will change which species get assigned in ambiguous
  multi-match cases, not the matching score itself. Watch the multi-match
  diagnostic output (`multi_match_clusters/`), not `matching_score`, when
  tuning this.
- **Height normalization only runs if `Clean_Pointcloud` is on.**
  `NORMALIZE_HEIGHTS = True` is hardcoded in `run_clustering.py`, but it's
  nested inside `if STEPS['Clean_Pointcloud']:`. With that flag `False` (the
  current default in `constants.py`), normalization never executes
  regardless of `NORMALIZE_HEIGHTS`.
- **`filter_clusters_by_chm_peaks()`** exists in `detection/find_chm_peaks.py`
  but isn't called anywhere in the current pipeline — it's dead code unless
  you wire it in.
- **Cached dataframes and reloaded clusters can drift out of sync.** Ground
  stripping and cluster splitting now always run in the shared post-branch
  section regardless of `Make_Clusters`. If `Make_Clusters=False` loads
  cached dataframes via `load_dataframes()`, make sure they were built from
  clusters that went through the *same* split parameters — otherwise the
  `file` index between `df_clusters` and the freshly-loaded cluster list can
  misalign.

---

## Debugging

```bash
bash shells/debug.sh <JOB_ID>        # (top-level shells/)
bash shells/see_recent_jobs.sh       # (top-level shells/)
```