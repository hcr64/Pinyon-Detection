import os
import re
import open3d as o3d


def save_clusters_descriptive(clusters, filenames, save_path):
    """
    Save a list of point cloud clusters to disk with caller-supplied,
    human-readable filenames instead of the generic cluster0.ply, cluster1.ply
    naming used by save_clusters().

    Intended for diagnostic/export folders (pre_split_clusters/,
    multi_match folders, etc.) that get synced to Google Drive for visual
    inspection and never get read back in by load_clusters() — so filenames
    can encode useful metadata (radius, species, distance) instead of just
    a positional index.

    Clears the destination folder before writing, same as save_clusters().

    Args:
        clusters (list of o3d.geometry.PointCloud): Clusters to save.
        filenames (list of str): One filename per cluster, same length and
            order as `clusters`. ".ply" is appended if not already present.
            Filenames are sanitised (illegal filesystem characters replaced
            with "_") but NOT deduplicated — pass unique names or later
            writes will overwrite earlier ones.
        save_path (str): Directory to write .ply files into. Created
            automatically if it does not exist.

    Returns:
        list of str: Full paths written, in input order.

    Requirements:
        open3d, os, re
    """

    if len(clusters) != len(filenames):
        raise ValueError(
            f"clusters ({len(clusters)}) and filenames ({len(filenames)}) "
            f"must be the same length"
        )

    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))
    else:
        os.makedirs(save_path)

    illegal = re.compile(r'[<>:"/\\|?*]')

    written = []
    seen = set()
    for cluster, name in zip(clusters, filenames):
        clean = illegal.sub("_", name)
        if not clean.endswith(".ply"):
            clean += ".ply"

        if clean in seen:
            print(f"Warning: duplicate filename '{clean}' — this write will "
                  f"overwrite a previous cluster with the same name.")
        seen.add(clean)

        out_path = os.path.join(save_path, clean)
        o3d.io.write_point_cloud(out_path, cluster)
        written.append(out_path)

    print(f"All {len(clusters)} clusters saved to {save_path} with descriptive names")
    return written