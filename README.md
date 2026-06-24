# Pinyon-Detection

A research pipeline for detecting and classifying pinyon, juniper, and ponderosa pine trees from drone-collected LiDAR/SfM point clouds. Developed for the **Sunset Crater** field site (Sunset_sfm_trial) and designed to run on the **Monsoon** SLURM HPC cluster.

The pipeline ingests raw `.las` files, builds a Canopy Height Model, detects tree tops as CHM peaks, segments individual crowns via watershed, and trains a species classifier on GPS-labeled clusters.

---

## Repository Structure

```
Pinyon-Detection/
├── main.py                  # Entry point — orchestrates the full pipeline
├── constants.py             # File paths, pipeline step flags, global settings
│
├── functions/               # All pipeline logic, organized by stage
│   ├── io/                  # Loading, saving, and converting data
│   ├── preprocessing/       # Point cloud cleaning, CHM building, ground stripping
│   ├── detection/           # CHM peak finding, crown segmentation, cluster splitting
│   ├── features/            # Per-cluster feature extraction for the classifier
│   ├── labeling/            # GPS label matching and scoring
│   └── classification/      # Model training and species prediction
│
├── shells/                  # SLURM job scripts and utilities
│   ├── pinyons.sh           # Single main run
│   ├── pinyon_sweep.sh      # Parameter sweep (job array)
│   ├── save_clusters_to_drive.sh
│   ├── setup_venv.sh
│   ├── debug.sh
│   └── see_recent_jobs.sh
│
├── parameters/
│   └── params.txt           # Parameter grid for sweep jobs
│
├── trial_data/
│   └── Sunset_sfm_trial/
│       ├── data/point_cloud/    # Raw .las input files
│       ├── pointclouds/         # Saved raw and cleaned .ply files
│       ├── CFMs/                # Canopy Height Model GeoTIFFs
│       ├── clusters/            # Segmented tree clusters (.ply)
│       ├── labeled_clusters/    # Clusters with GPS species labels
│       ├── labels/              # GPS label CSVs
│       ├── images/              # Diagnostic plots
│       └── results/             # Sweep result CSVs
│
└── results/                 # Aggregate result CSVs across sweeps
```

See [`PIPELINE.md`](PIPELINE.md) for a walkthrough of the full processing flow.

---

## Quick Start

### 1. Environment setup

```bash
bash shells/setup_venv.sh
source open3d_env/bin/activate
```

Requires Python 3.10. Key dependencies: `open3d`, `numpy`, `scipy`, `scikit-learn`, `rasterio`, `scikit-image`, `laspy`, `pyproj`, `pandas`, `matplotlib`.

### 2. Configure paths and steps

Edit `constants.py` to set:
- `TRIAL_NAME` — the subfolder under `trial_data/` to use
- `PATHS` — all input/output file paths (derived from `TRIAL_NAME`)
- `STEPS` — boolean flags controlling which pipeline stages run

### 3. Single run

```bash
sbatch shells/pinyons.sh
```

Or run directly (adjust parameters inline in the script first):

```bash
python -u main.py \
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
    --job_id test \
    --trial_name Sunset_sfm_trial
```

### 4. Parameter sweep

Edit `parameters/params.txt` (one parameter set per line), update the `--array` count in `pinyon_sweep.sh`, then:

```bash
sbatch shells/pinyon_sweep.sh
```

Results append to `trial_data/Sunset_sfm_trial/results/newester_results.csv`.

---

## Key Parameters

| Parameter | Description | Tuned value |
|---|---|---|
| `voxel_size` | Downsampling cell size (m) | 0.08 |
| `green_threshold` | Minimum green channel dominance for vegetation filter | 0.025 |
| `min_height` | Minimum CHM height to count as a tree peak (m) | 1.0 |
| `search_radius_m` | Local max window radius for peak detection (m) | 3.0 |
| `max_radius` | Maximum crown radius for watershed assignment (m) | 3.0–4.0 |
| `max_distance` | Maximum GPS-to-cluster distance for label matching (m) | 2.85–4.0 |
| `min_points` | Minimum points per valid cluster | 200 |
| `min_peak_distance` | Minimum trunk-to-trunk distance for cluster splitting (m) | 3.0 |
| `eps` | DBSCAN epsilon — unused with CHM method, keep at 2.0 | 2.0 |

**Findings from parameter sweeps:**
- `normalize_heights=False` consistently outperformed `True`
- `voxel_size=0.1` outperformed `0.08`; `eps=2.0` outperformed `2.5`
- `max_distance ≈ 2.85` is generally optimal for label matching
- `min_points` in the 35–50 range (pre-CHM clustering) or ~200 (post-CHM) for valid detections

---

## Matching Score

The pipeline reports a **matching score** after each run: the fraction of GPS-labeled trees that have exactly one cluster within `max_distance`. A perfect score is 1.0; the best achieved on Sunset Crater is **~0.83**.

```
Perfect matches (1:1):  58  (82.9%)
No match:                8  (11.4%)
Multiple clusters:        4   (5.7%)
Matching score:         0.829
```

---

## Field Site

**Sunset Crater**, Arizona. The scan area covers a cinder cone with significant terrain variation (~30 m relief). The irregular drone flight path means coverage is not a uniform rectangle — a `coverage_radius` proximity filter in the label matcher handles the non-rectangular footprint.

Species present: pinyon pine, juniper, ponderosa pine.

---

## Debugging

```bash
# Check a specific SLURM job
bash shells/debug.sh <JOB_ID>

# View recent pinyon jobs
bash shells/see_recent_jobs.sh
```

Diagnostic plots (GPS vs cluster overlap, species scatter) are saved to `trial_data/Sunset_sfm_trial/images/` after each run.
