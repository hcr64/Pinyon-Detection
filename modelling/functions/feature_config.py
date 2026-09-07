"""
feature_config.py
────────────────────────────────────────────────────────────────────────────────
Single source of truth for the feature columns used by every classifier in
modelling/functions/classification/.

Previously FEATURES was defined separately (and identically, by luck) in
train_tree_classifier.py, advanced_classifiers.py, and semi_supervised.py.
Adding an engineered feature meant updating three files by hand with no
enforcement that they stayed in sync. Import FEATURES from here instead.

NOTE: advanced_classifiers.py and semi_supervised.py still hardcode their
own FEATURES lists (see modelling/README.md "Known Quirks") — if you add a
feature here, add it there too until those are pointed at this file.

Requirements: none
"""

# baseline shape/PCA/colour features from get_deep_cluster_features()
# + engineered ratio features from engineer_features()
# + structural features from get_deep_cluster_features() (verticality etc.)
# + chromaticity features from get_deep_cluster_features() (shadow-robust,
#   per-point RGB normalized by per-point brightness before averaging —
#   see clustering/functions/preprocessing/clean_up_pointcloud.py docstring
#   and modelling notes on shadow/illumination robustness)
FEATURES = [
    "height", "radius", "n_points",
    "obb_extent_x", "obb_extent_y", "obb_extent_z",
    "eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
    "linearity", "planarity", "sphericity",
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "height_to_radius",
    "green_dominance",
    "crown_volume",
    "color_saturation",
    "verticality",
    "flatness_ratio",
    "crown_base_ratio",
    "chroma_r",
    "chroma_g",
    "chroma_b",
    "chroma_g_std",
    "exg_chroma",
    "top_red_exr",
    "top_blue_exb",
    "top_yellow_exy",
    "top_grey",

    # footprint / irregularity
    # "hull_area_ratio",
    # "hull_volume_ratio",
    "obb_aspect_ratio",
    "n_density_peaks",

    # vertical/radial profile
    # "z_skewness",
    # "z_kurtosis",
    "trunk_gap_score",
    "radial_density_gradient",

    # colour, reframed
    "hue_mean",
    "hue_std",
    "saturation_mean",
    "bark_fraction", 
]