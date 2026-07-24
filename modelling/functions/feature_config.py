"""
feature_config.py
────────────────────────────────────────────────────────────────────────────────
Single source of truth for the feature columns used by every classifier in
modelling/functions/classification/.

Previously FEATURES was defined separately (and identically, by luck) in
train_tree_classifier.py, advanced_classifiers.py, and semi_supervised.py.
Adding an engineered feature meant updating three files by hand with no
enforcement that they stayed in sync. Import FEATURES from here instead.

Requirements: none
"""

# baseline shape/PCA/colour features from get_deep_cluster_features()
# + engineered ratio features from engineer_features()
# + structural features from get_deep_cluster_features() (verticality etc.)
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
]