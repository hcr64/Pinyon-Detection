from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd

def train_tree_classifier(df_deep, df_labels_matched):

    # merge features with labels
    df = df_deep.merge(df_labels_matched[["file", "Name"]], on="file")

    # drop rows with no label and unknown labels
    df = df.dropna(subset=["Name"])
    df = df[df["Name"] != "unknown"]

    print(f"Training on {len(df)} labeled clusters")
    print(f"Classes: {df['Name'].unique()}")

    # define features
    features = [
        "height", "radius", "n_points",
        "obb_extent_x", "obb_extent_y", "obb_extent_z",
        "eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
        "linearity", "planarity", "sphericity",
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b"
    ]

    X = df[features]
    y = df["Name"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # feature importance
    importances = pd.Series(model.feature_importances_, index=features)
    print("\nFeature Importances:")
    print(importances.sort_values(ascending=False))

    # cross validation score
    scores = cross_val_score(model, X, y, cv=3)  # cv=3 since juniper/ponderosa are small
    print(f"\nCross Validation Accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")

    # predict on ALL clusters including unlabeled ones
    df_deep["predicted_label"] = model.predict(df_deep[features])

    return model, features