import open3d as o3d
import os


def save_labeled_clusters(clusters, df_clusters, save_path):
    """
    Save only GPS-labeled clusters to disk, named by species.
 
    Iterates df_clusters and writes each cluster whose "Name" column holds a
    known species (not "unknown") as "<species>_<n>.ply" — e.g. pinyon_0.ply,
    juniper_1.ply. Clusters with label "unknown" are silently skipped. The
    output folder is cleared of existing .ply files before writing.
 
    Args:
        clusters (list of o3d.geometry.PointCloud): Full cluster list from
            cluster_by_chm_peaks() or split_large_clusters(). Indices must
            correspond to the "file" column in df_clusters.
        df_clusters (pd.DataFrame): DataFrame returned by
            match_labels_to_clusters(). Must contain at least the columns
            "file" (int cluster index) and "Name" (species string or "unknown").
        save_path (str): Directory to write .ply files into. Created
            automatically if it does not exist.
 
    Returns:
        list of str: Sorted list of file paths written.
 
    Requirements:
        open3d, os
    """

    # prepare the output directory
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            if f.endswith(".ply"):
                os.remove(os.path.join(save_path, f))
    else:
        os.makedirs(save_path)

    # keep only rows that have a real label
    labeled = df_clusters[
        df_clusters["Name"].notna() &
        (df_clusters["Name"] != "unknown")
    ]

    # track how many of each species we've saved (for the numeric suffix)
    species_counts = {}
    saved = []

    # iterate in file-index order for reproducibility
    for _, row in labeled.sort_values("file").iterrows():
        cluster_idx = int(row["file"])
        species     = str(row["Name"]).lower().strip()

        # bounds-check — cluster list and df may differ after filtering
        if cluster_idx >= len(clusters):
            print(f"Warning: cluster index {cluster_idx} out of range "
                  f"(only {len(clusters)} clusters). Skipping.")
            continue

        # build filename: pinyon_0.ply, pinyon_1.ply, juniper_0.ply, …
        n = species_counts.get(species, 0)
        filename  = f"{species}_{n}.ply"
        full_path = os.path.join(save_path, filename)

        o3d.io.write_point_cloud(full_path, clusters[cluster_idx])
        species_counts[species] = n + 1
        saved.append(full_path)

    # summary
    print(f"Saved {len(saved)} labeled clusters to '{save_path}':")
    for species, count in sorted(species_counts.items()):
        print(f"  {species}: {count}")

    return saved