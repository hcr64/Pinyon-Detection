import os
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import KDTree

SPECIES = ["pinyon", "juniper", "ponderosa"]


def _parse_one(path):
    """Parse one raw GPS CSV into species / drought tag / lat / lon rows."""
    df = pd.read_csv(path)

    missing = [c for c in ("Name", "Latitude", "Longitude") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing} "
                         f"(has {list(df.columns)})")

    # drop dead trees and survey points
    lowered = df["Name"].astype(str).str.lower()
    df = df[~lowered.str.contains("dead", na=False)
            & ~lowered.str.contains("point", na=False)]

    raw     = df["Name"].astype(str).str.strip()
    tokens  = raw.str.split()
    species = tokens.str[0].str.lower().replace({"junioer": "juniper"})

    # everything after the first word is the drought tag, e.g. "T4", "S 7", "T"
    tag   = tokens.str[1:].str.join(" ").str.strip().str.upper()
    parts = tag.str.extract(r"^([TS])\s*(\d{1,2})?$")

    # a tag that exists but doesn't parse (e.g. "pinyon juniper") is an entry error
    bad = (tag != "") & parts[0].isna()
    if bad.any():
        print(f"  [{os.path.basename(path)}] dropping {bad.sum()} rows with "
              f"unparseable tags: {sorted(raw[bad].unique())}")

    out = pd.DataFrame({
        "species":       species,
        "drought_class": parts[0].map({"T": "tolerant", "S": "susceptible"}),
        "drought_level": pd.to_numeric(parts[1], errors="coerce"),
        "Latitude":      df["Latitude"],
        "Longitude":     df["Longitude"],
        "source":        os.path.basename(path),
    })
    out = out[~bad & out["species"].isin(SPECIES)]
    out = out.dropna(subset=["Latitude", "Longitude"])
    return out.reset_index(drop=True)


def build_label_csv(input_paths, output_path, epsg="EPSG:26912",
                    dedup_distance=1.0):
    """
    Merge any number of raw GPS label CSVs into one processed label CSV.

    Output columns: x, y (UTM metres), species, drought_class,
    drought_level, source.

    Args:
        input_paths (list[str]): Raw CSVs with Name / Latitude / Longitude.
            Name is like "pinyon", "juniper", "ponderosa", "pinyon T4",
            "pinyon S7".
        output_path (str): Where to write the processed CSV.
        epsg (str): Target CRS. Default NAD83 / UTM 12N, matching the
            pipeline's existing conversion.
        dedup_distance (float): GPS points closer than this (metres) are
            treated as the same tree; the earlier file/row wins.

    Returns:
        pd.DataFrame: The processed labels.
    """
    frames = []
    for p in input_paths:
        print(f"Reading {p}")
        df = _parse_one(p)
        print(f"  kept {len(df)} rows")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # lat/lon -> UTM
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x, y = tf.transform(df["Longitude"].values, df["Latitude"].values)
    df["x"], df["y"] = np.round(x, 3), np.round(y, 3)

    # distance-based dedup across all files (earlier row wins)
    pairs = sorted(KDTree(df[["x", "y"]].values).query_pairs(r=dedup_distance))
    drop, conflicts = set(), []
    for i, j in pairs:
        if i in drop:
            continue
        drop.add(j)
        a, b = df.iloc[i], df.iloc[j]
        if (a["species"] != b["species"]
                or (a["drought_class"] != b["drought_class"]
                    and pd.notna(a["drought_class"]) and pd.notna(b["drought_class"]))):
            conflicts.append((i, j))

    print(f"\nDuplicates within {dedup_distance} m dropped: {len(drop)}")
    for i, j in conflicts:
        a, b = df.iloc[i], df.iloc[j]
        print(f"  CONFLICT kept {a['species']} {a['drought_class']} ({a['source']}) "
              f"over {b['species']} {b['drought_class']} ({b['source']}) "
              f"at ({a['x']:.1f}, {a['y']:.1f})")
    df = df.drop(index=list(drop)).reset_index(drop=True)

    df = df[["x", "y", "species", "drought_class", "drought_level", "source"]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} labels -> {output_path}")
    print(df["species"].value_counts().to_string())
    print("Pinyon drought tags:")
    print(df.loc[df["species"] == "pinyon", "drought_class"]
            .value_counts(dropna=False).to_string())
    return df