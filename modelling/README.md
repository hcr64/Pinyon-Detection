# Pinyon-Detection — Species Classification

This is the cheap, iterate-quickly half of the pipeline: trains a species
classifier (pinyon / juniper / ponderosa) on GPS-labeled clusters. Reads
straight from the dataframes `run_clustering.py` already saved — no point
cloud, no clustering, no GPS matching. Runs in seconds to low minutes.

Requires `run_clustering.py` (in `../clustering/`) to have completed at least
once for the trial you're pointing at, so `df_clusters` already has its
`"Name"` column populated.

---

## Structure

```
modelling/
├── train_model.py
└── functions/
    ├── features/
    │   ├── get_pointcloud_stats.py        # position/geometry df (used for GPS matching)
    │   ├── get_deep_cluster_features.py   # shape/PCA/colour/structural features + engineer_features()
    │   └── plot_feature_separability.py   # per-feature-pair scatter + boxplot diagnostics
    │
    └── classification/
        ├── train_tree_classifier.py       # baseline models + toggleable enhancements
        ├── advanced_classifiers.py        # XGBoost, LightGBM, soft-voting ensemble, MLP, calibrated SVM
        └── semi_supervised.py             # LabelSpreading over labeled + unlabeled clusters
```

---

## Quick start

```bash
python -u train_model.py --trial_name Sunset_sfm_trial
```

Or via SLURM:

```bash
sbatch shells/train_model.sh
```

Add `--advanced` to swap `train_tree_classifier()` for
`run_advanced_classifiers()` + semi-supervised label spreading:

```bash
python -u train_model.py --trial_name Sunset_sfm_trial --advanced
```

Output: per-model classification reports, macro F1 / Cohen's Kappa, a
confusion matrix PNG for the best model, feature importances (tree models
only), and `predicted_label` / `prob_<species>` columns appended to
`df_deep_clusters` in memory (not currently re-persisted to disk — if you
need the predictions saved, add a `save_dataframes()` call at the end of
`train_model.py`).

---

## The data bottleneck



---

## `train_tree_classifier.py` enhancement toggles

```python
ENHANCEMENTS = {
    "SMOTE":     False,   # synthetic oversampling for juniper/ponderosa
    "WEIGHTING": True,    # manual 2× upweight on ponderosa beyond "balanced"
    "SELECTION": True,    # drop features below mean RF importance
    "TUNING":    False,   # RandomizedSearchCV over RF hyperparameters (slowest)
}
```

Each enabled enhancement adds its own row to the model comparison table, so
you can see exactly what each intervention buys over baseline RF. Toggle
these directly in `train_tree_classifier.py` — they aren't currently exposed
as CLI flags.

---
## Normal Features (Species Classifier)
Without any flags, the a species classifier runs, that classifies trees as pinyons, ponderosas, and junipers, trained on labeled clusters. It runs a hanful of models, and outputs a confusion matrix of the best performer. 

<div align="center">
  <img src="../assets/modelling/confusion_matrix.png" alt="Species classifier confusion matrix of SVM_RBF (see advanced classifiers) model." width="75%">
</div>
Species classifier confusion matrix of SVM_RBF (see advanced classifiers) model.

---
## Advanced classifiers (`--advanced`)

Adds XGBoost, LightGBM, a soft-voting ensemble (RF+GB+SVM), an MLP, and a
calibrated SVM to the comparison. Requires `xgboost` and `lightgbm` in the
venv — **not currently in `setup_venv.sh`**, install manually if you use
`--advanced`:

```bash
pip install xgboost lightgbm
```

`--advanced` also runs semi-supervised `LabelSpreading` afterward, which
propagates species labels from the ~166 labeled clusters to the ~3,400
unlabeled ones via an RBF similarity graph in feature space. Two diagnostic
plots are saved: a confidence histogram and a height/radius scatter with
labeled clusters ringed in black. Treat the printed "labeled-node recovery"
metric as a consistency check, not a real accuracy estimate — the model saw
those labels during fitting.

---
## Drought Tolerance Prediction (`--drought`)
The pinyon drought tolerance prediction model takes only S for drought susceptible or T for drought tolerant, and predicts those. In the GPS labels, there is also a number 1-10 along with the letter as to how tolerant or susceptible it is. the lower the number the more drought tolerant, higher is more drought susceptible.

<div align="center">
  <img src="../assets/modelling/drought_confusion_matrix.png" alt="Combined species classifier and pinyon drought tolerance confusion matrix of SVM_RBF (see advanced classifiers) model." width="75%">
</div>
Above: pinyon drought tolerance confusion matrix. 


<div align="center">
  <img src="../assets/modelling/combined_confusion_matrix.png" alt="Combined species classifier and pinyon drought tolerance confusion matrix of SVM_RBF (see advanced classifiers) model." width="75%">
</div>
Combined species classifier and pinyon drought tolerance confusion matrix of SVM_RBF (see advanced classifiers) model.

---
## Known Quirks

- **`FEATURES` is duplicated, not shared.** `train_tree_classifier.py`,
  `advanced_classifiers.py`, and `semi_supervised.py` each define their own
  `FEATURES` list. They currently agree, but there's no single source of
  truth — if you add an engineered feature in
  `get_deep_cluster_features.py`, you need to add it to all three by hand.
- **`train_model.py` doesn't re-save predictions.** `predicted_label` /
  `prob_<species>` columns are added to `df_deep_clusters` in memory only.
  If you want them available on a later run without recomputing, add a
  `save_dataframes()` call before the script exits.
- **`--advanced` requires packages not in the shared venv setup.** See above.