# Detecting Drought Resistant Pinyon Pines using Aerial Structure-from-Motion

A research project intended to identify drought-tolerant pinyon pines from drone-collected Structure-From-Motion pointclouds processed using PixMapper4D. Currently, data has only been collected from a field site at **Sunset Crater, AZ**. The processing is done on the Monsoon SLURM HPC cluster at Northern Arizona University. 

Drought tolerance in pinyon pine (Pinus edulis) varies among individual trees growing side by side, and this variation is heritable ([Sthultz et al. 2009](https://doi.org/10.1111/j.1365-2486.2009.01901.x)).  Long-term monitoring shows that moth-susceptible, shrub-like trees on poor soils declined less during drought than resistant, upright trees ([Whipple et al. 2019](https://doi.org/10.3389/fpls.2019.00132)). Another part of the mechanism appears to be genetically determined differences in the ectomycorrhizal fungal communities that colonize a tree’s roots, which are passed from parent trees to seedlings ([Gehring et al. 2017](https://www.pnas.org/doi/full/10.1073/pnas.1704022114)).

> **In this README moth-susceptible = drought-tolerant**, and moth classes will not be mentioned again going forward.

 In the field, pinyons drought tolerance appeared to exist on a spectrum, so tree tags are recorded with numbers 1 (tolerant) through 10 (susceptible). The project itself currently ignores the exact number and only models on tolerant/susceptible pinyons. The number rankings are helpful as diagnostics for drought class model results. 

<div align="center">
  <img src="assets/root/SSC_pinyon1.jpg" alt="A drought susceptable pinyon pine at site A in Sunset Crater, AZ." width="75%">
</div>

>Above: A more drought tolerant pinyon pine at site A in Sunset Crater, AZ. Taken by myself. 

<div align="center">
  <img src="assets/root/tolerant_and_susceptible.jpeg" alt="A  very drought tolerant pinyon pine on the left, and a  very susceptible pinyon on the right. Taken by myself at site A in Sunset Crater, AZ." width="75%">
</div>

> Above: A  drought tolerant pinyon pine on the left, and a  susceptible pinyon on the right. Taken by myself at site A in Sunset Crater, AZ. 


The pipeline takes in raw `.las` files, builds a Canopy Height Model, detects
tree tops as CHM peaks, segments individual crowns via watershed, matches
GPS-tagged ground-truth species to clusters, and trains a species classifier
on the labeled subset. All `.las` used have been exported from PixMapper4D.

<div align="center">
  <img src="assets/root/PM4D_SC_aerial.png" alt="Screenshot of Structure-from-motion pointcloud of site A in Sunset Crater, AZ. Visualized using PixMapper4D." width="75%">
</div>

>Screenshot of Structure-from-motion pointcloud of site A in Sunset Crater, AZ. Visualized using PixMapper4D.

The project is split into two independent stages:

- **`clustering/`** — everything expensive: point cloud processing, CHM,
  watershed segmentation, GPS label matching. Run via `run_clustering.py`,
  usually swept over many parameter combinations to optimize the matching
  score. Depending on parameters, usually takes around 25 to 50 minutes (Not counting reading in .las files). More on runtime estimation in [`PIPELINE.md`](PIPELINE.md).

- **`modelling/`** — species classification on the labeled clusters, and drought tolerance on pinyons.
  `clustering/` produced. Run via `train_model.py`, fast enough to iterate
  on interactively.
  Takes a lot less time to run, usually less than 5 minutes. 

These used to be one script (`main.py`). They were split apart because clustering was taking too long and was not required for training the model. The main file became too cluttered, and was clearly two seperate parts. 

---
## AI Usage Disclaimer
All python and bash coding has been done by Anthropic's Claude Sonnet models (versions 4.x and 5.x). Some slight touch ups have been done by myself on code outputted from claude. The web browser version has been used, and outputs are copied into NAU's Monsoon's VSCode IDE.

---

## End Goals
Ultimately, accurate prediction maps contatining species information and pinyon drought tolerance will be output for any group of `.las` files inputted, without needing any GPS tags. On top of that, I would like to create a more interactive app that shows more in-depth information on potential trees. 

<div align="center">
  <img src="assets/root/prediction_map.png" alt="Example of prediction map output from train_model.py. Created Oct. 2026" width="75%">
</div>

>Example of prediction map output from train_model.py. Created Oct. 2026 

Prediction matrices below are from a run early Nov 2026, Macro F1 scores of 0.9 or higher are targeted. The first one is a specifier to distinguish between All Pinyons/Junipers/Ponderosas, then the one below is for pinyon drought classes. Multiple models get trained for specification with many performing well, and Multi-Layer Perceptron models performing the most consistently.


<div align="center">
  <img src="assets/root/cm_CalibratedSVM.png" alt="{Confusion Matrix of CSVM Specifier model as of Nov 2026}" width="75%">
</div>

>Above: Specifier confusion matrix results of a Support Vector Machine model.

<div align="center">
  <img src="assets/root/drought_confusion_matrix.png" alt="{Confusion Matrix of pinyon drought model as of Nov 2026}" width="75%">
</div>

>Above: Pinyon drought class modelling results and parameters used. 

I would also like to write and publish a scientific paper about detecting drought tolerance in pinyons via SfM, as well as their predicted growing patterns across elevation gradients. 

## Repository Structure
(May not be completely up-to-date)
```
Pinyon-Detection/
├── README.md                      # project overview
├── PIPELINE.md                    # stage-by-stage walkthrough + parameter reference
├── .gitignore                     # excludes venv, generated clouds/clusters/logs
│
├── global_files/                  # shared by clustering/ and modelling/
│   ├── __init__.py
│   ├── constants.py               # STEPS flags, get_paths(trial_name)
│   ├── get_pointcloud_stats.py    # clusters_to_dataframe, density-weighted centroid
│   ├── get_deep_cluster_features.py  # make_deep_dataframe, engineer_features
│   └── save_clusters.py           # save/load clusters + dataframes, save_predictions
│
├── clustering/                    # expensive half: point cloud → GPS-labeled clusters
│   ├── run_clustering.py          # entry point
│   ├── params.txt                 # sweep grid (one row per SLURM array task)
│   ├── functions/
│   │   ├── io/
│   │   │   ├── las_folder_to_pointcloud.py
│   │   │   └── save_labeled_clusters.py
│   │   ├── preprocessing/
│   │   │   ├── build_chm.py       # CHM from external DSM, load_chm
│   │   │   └── strip_ground_from_clusters.py
│   │   ├── detection/
│   │   │   ├── find_chm_peaks.py
│   │   │   ├── cluster_by_chm_peaks.py   # watershed segmentation
│   │   │   ├── split_large_clusters.py   # density peaks + Mean Shift
│   │   │   └── filter_green_crown.py
│   │   └── labeling/
│   │       ├── build_label_csv.py
│   │       ├── match_labels_to_clusters.py
│   │       ├── filter_gps_by_coverage.py
│   │       ├── diagnose_gps_losses.py
│   │       ├── save_multimatch_clusters.py
│   │       └── plot_problem_gps_areas.py
│   ├── analysis/                  # one-off utility scripts
│   │   ├── build_labels.py
│   │   ├── compute_fine_peaks.py
│   │   └── estimate_gps_offset.py
│   ├── shells/
│   │   ├── pinyons.sh             # single run, hand-tuned defaults
│   │   ├── pinyon_sweep.sh        # SLURM array parameter sweep
│   │   └── fine_peaks.sh
│   └── logs/                      # SLURM stdout/stderr
│
├── modelling/                     # cheap half: species + drought classifiers
│   ├── train_model.py             # entry point
│   ├── model_params.txt           # classifier sweep grid (stale, see PIPELINE.md)
│   ├── functions/
│   │   ├── feature_config.py      # FEATURES: single source of truth
│   │   ├── features/
│   │   │   ├── select_best_features_rf.py
│   │   │   ├── pointcloud_autoencoder.py
│   │   │   ├── plot_feature_separability.py
│   │   │   ├── plot_cluster_reality_check.py
│   │   │   ├── plot_prediction_map.py
│   │   │   └── diagnose_cluster_bimodality.py
│   │   └── classification/
│   │       ├── train_tree_classifier.py
│   │       ├── advanced_classifiers.py
│   │       ├── semi_supervised.py
│   │       ├── evaluate_semi_supervised.py
│   │       ├── train_drought_classifier.py
│   │       ├── train_combined_classifier.py
│   │       ├── inspect_misclassifications.py
│   │       └── inspect_drought_misclassifications.py
│   ├── shells/
│   │   ├── train_model.sh
│   │   └── model_sweep.sh
│   └── logs/
│
├── shells/                        # cross-cutting utilities
│   ├── setup_venv.sh
│   ├── access_venv.sh
│   ├── jupyter_kernel_env.sh
│   ├── run_full_pipeline.sh       # clustering → modelling → collect plots
│   ├── see_recent_jobs.sh
│   └── clear_garbage_files.sh
│
├── trial_data/
│   └── Sunset_sfm_trial/
│       ├── labels/                # raw + processed GPS label CSVs
│       ├── dataframes/            # clusters.csv, deep_clusters.csv, predictions.csv
│       ├── images/                # diagnostic plots, confusion matrices, prediction map
│       ├── GPS_results/           # sweep_<id>.csv / job_<id>.csv
│       ├── models/                # pointnet_ae.pt checkpoint
│       └── run_outputs/           # run_full_pipeline.sh collected outputs
│
├── assets/                        # README images
│
├── results/                       # early manual-trial CSVs
└── legacy_files/                  # retired code, old READMEs, old results
```

---

## Quick Start

### 1. Environment setup

```bash
bash shells/setup_venv.sh

module load miniforge3/26.3.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open3d_env
```

Requires Python 3.10. Key dependencies: `open3d`, `numpy`, `scipy`,
`scikit-learn`, `rasterio`, `scikit-image`, `laspy`, `pyproj`, `pandas`,
`matplotlib`, `xgboost`. 

`.gitignore` excludes `open3d_env/` and every generated pipeline artifact
(point clouds, clusters, CHMs, dataframes, images, SLURM logs) — these are
all regenerable from a `run_clustering.py` run and get large fast and are mainly used for debugging, so they
stay out of git history. `results/` CSVs are small and are the actual
experiment record, so those stay tracked.

### 2. Configure paths and steps

Edit `clustering/constants.py`:
- `get_paths(trial_name)` — all input/output paths, derived from the trial name
  - `PATHS["DATA"]` needs to have the .las files to run, at least for the first time.
  - `PATHS["DSM"]` can have a DSM if one is available, if not it can get created. 
- `STEPS` — boolean flags controlling which clustering stages re-run
  - Steps can only be skipped after being ran at least once, so the data can be saved.

### 3. Run clustering + labeling

```bash
sbatch clustering/shells/pinyons.sh
```

See [`PIPEPLINE.md`](PIPELINE.md) for CLI args, the
`params.txt` sweep format, and current known quirks. Both `pinyons.sh` and `pinyon_sweep.sh` accept optional `-m` string arguments for notes along with the run, similar to `git commit -m`. 

Job results for both sweeps and individual jobs can be found in `trial_data/{trial_name}/GPS_results/`. Log files for debugging or any printed outputs from any job are in `clustering/logs/`.
### 4. Train the classifier

```bash
sbatch shells/train_model.sh
```

See [`PIPEPLINE.md`](PIPELINE.md) again for the enhancement
toggles, the `--advanced` classifier comparison, and semi-supervised label
spreading.

### 5. (Optional) Do a full run
```
sbatch shells/run_full_pipeline.sh
```
Runs a clustering job then a modelling one, doing steps 3 and 4 with one script. Outputs of both sides go to `/run_outputs`, in a folder named `[TRIAL_NAME]_[JOB_ID]_[DATE]_[TIME]`. 

---

## Matching Score

`clustering/`'s output metric: the fraction of GPS-labeled trees with
exactly one detected cluster within `max_distance`.

<div align="center">
  <img src="assets/root/cluster_label_overlay.png" alt="Graph displaying clusters and labels collected from the field of site A in Sunset Crater, AZ." width="75%">
</div>

>Graph displaying clusters and labels collected from the field of site A in Sunset Crater, AZ. 

(Printed outputs from run_clustering.py below)

```
Perfect matches (1:1):   197  (83.5%)
No match:                26  (11.0%)
Multiple clusters:       13  (5.5%)
Matching score:          0.835
```

Best achieved on Sunset Crater so far: **~0.835**.

---

## The Data Bottleneck

The labeled dataset is small — **~319 labeled clusters**, with **ponderosa
at only ~30 samples** right now. This has been the fundamental constraint on
classifier performance; no model architecture, oversampling, weighting, or
feature engineering approach tried so far has overcome it. More labeled
ground-truth data, not a better classifier, is the highest-leverage next
step for the modelling side.

---

## Field Site

**Sunset Crater**, Arizona. The scan area covers a cinder cone with
significant terrain variation. The irregular drone flight
path means coverage is not a uniform rectangle — a `coverage_radius`
proximity filter in the label matcher handles the non-rectangular footprint.

Species present: pinyon pine, juniper, ponderosa pine.

<div align="center">
  <img src="assets/root/SSC_google_maps_SC.png" alt="Aerial screen shot of site A in Sunset Crater, AZ. Taken on Google Maps." width="75%">
</div>

>Aerial screen shot of site A in Sunset Crater, AZ. Taken on Google Maps. 

## Debugging
 Individual instances of clustering jobs have `.txt` and `.err` files in `clustering/logs/`, and modelling job logs live in `modelling/logs/`. The `.txt` files have the progress of the job along with it's results and parameters. In `run_clustering.py`, the variable `SILENT` produces even more in-depth outputs when set to `True`. 

`run_full_pipeline.sh` puts its outputs in `trial_data/run_outputs/`, and each log file has clustering and modelling in the same file.

## To Do List 

- **Elevation Gradient/Large Flyover:** A final flyover covering a larger area over a more dramatic elevation gradient is an end goal idea to test final models. This would capture more and less  water stressed areas to possibly reveal patterns pinyon drought tolerance. This flyover could be a thin rectangle, or multiple flyights put together.
- **Seperate Overlapping Trees:** A problem that I have been working on for quite a while, there are a few trees that physically overlap that we gathered labels for. Overlapping trees are common, so having a method of seperation would be ideal.

- **Collect Thermal & Multispectral Data:** On top of the main research goal, looking at relationships between drought tolerance in pinyons and thermal & multipsectral data would be cool to explore. Hopefully multispectral data could be collected by December, but themal data may not be possible to get collected until warmer seasons. 


## See Also

- [`PIPELINE.md`](PIPELINE.md) — detailed walkthrough of every pipeline stage and functions. 