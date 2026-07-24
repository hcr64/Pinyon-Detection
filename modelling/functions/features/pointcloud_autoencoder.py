"""
pointcloud_autoencoder.py
────────────────────────────────────────────────────────────────────────────────
Self-supervised point cloud embeddings for Pinyon-Detection, as a drop-in
feature source alongside get_deep_cluster_features.py.

Why this instead of PointMLP / PointNeXt / Point Transformer / KPConv
───────────────────────────────────────────────────────────────────────────────
Those architectures assume either (a) a large labeled dataset to train
end-to-end, or (b) a pretrained checkpoint to transfer from. Neither applies
cleanly here:
    - (a) is a non-starter with ~166 labeled clusters (~20 ponderosa) — see
      the data-bottleneck notes in modelling/README.md. A multi-million
      parameter point transformer will overfit this instantly.
    - (b) pretrained checkpoints for those five architectures are GitHub-only,
      often require custom CUDA extensions to even build (pointnet2_ops,
      KPConv kernels), and are pretrained on CAD models (ModelNet40) or
      indoor scans (ScanNet/S3DIS) — a large domain gap from aerial LiDAR
      tree crowns with no guarantee of useful transfer.

This module instead trains a small PointNet-style autoencoder AS SELF-
SUPERVISED PRETRAINING on your own ~3,600 clusters (labeled + unlabeled) —
same "unlabeled data isn't waste" logic as semi_supervised.py's
LabelSpreading, just applied to representation learning instead of label
propagation. No labels are used during autoencoder training. The encoder's
bottleneck vector becomes a fixed-length embedding per cluster, which you
concatenate onto the existing engineered features and feed into your
existing train_tree_classifier.py / RF pipeline — the classical classifier
still only has to learn on 166 samples, just with (optionally) richer
input features.

Architecture
────────────
Encoder: per-point shared MLP (1x1 conv equivalent) → global max-pool
         → bottleneck vector. This is exactly PointNet's classification
         backbone minus the classification head — no custom ops, pure
         torch.nn.Conv1d + torch.max.
Decoder: MLP that maps the bottleneck vector back to a fixed-size point
         set (FoldingNet-lite — a simple "decode to N*3 coordinates and
         reshape" head, not full folding-based decoding, to keep this
         dependency-free and fast to train on CPU or a single GPU).
Loss:    Chamfer distance between reconstructed and input point sets
         (implemented here in plain torch — no pytorch3d dependency).

Usage
─────
    from pointcloud_autoencoder import (
        train_autoencoder, extract_embeddings, PointCloudDataset
    )

    # clusters = list of o3d.geometry.PointCloud, ALL of them (labeled +
    # unlabeled) — same clusters list used elsewhere in the pipeline
    encoder, decoder = train_autoencoder(
        clusters,
        n_points=256,
        embedding_dim=64,
        epochs=100,
        save_path=PATHS['Images'] + '../models/pointnet_ae.pt'
    )

    df_embeddings = extract_embeddings(clusters, encoder, n_points=256)
    # df_embeddings has columns: file, emb_0 ... emb_63

    # merge onto your existing deep feature dataframe before training:
    df_deep_clusters = df_deep_clusters.merge(df_embeddings, on="file")

Requirements
────────────
    torch (CPU-only is fine for ~3,600 clusters at n_points=256; a GPU
    node on Monsoon will just be faster — check `nvidia-smi` / partition
    docs before assuming one is available)
    numpy, open3d, pandas

    NOT in setup_venv.sh yet:
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        (or the appropriate CUDA wheel if you confirm a GPU partition)
"""

import numpy as np
import pandas as pd
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── point sampling helper ─────────────────────────────────────────────────────

def _sample_points(points, n_points, rng):
    """
    Resample a variable-length point array to a fixed size n_points.

    Clusters range from a few dozen to several thousand points, but the
    autoencoder needs a fixed-size input/output. Uses random sampling with
    replacement when the cluster has fewer than n_points, and random
    sampling without replacement otherwise. Center + scale-normalizes so
    the network sees shape, not absolute position/size — absolute
    height/radius are already captured by your existing engineered
    features, so the embedding should focus on shape detail those miss.

    Args:
        points (np.ndarray): (M, 3) XYZ array.
        n_points (int): Target point count.
        rng (np.random.Generator): Shared RNG for reproducibility.

    Returns:
        np.ndarray: (n_points, 3) resampled, centered, unit-scaled points.

    Requirements:
        numpy
    """
    m = len(points)
    if m == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    replace = m < n_points
    idx = rng.choice(m, size=n_points, replace=replace)
    sampled = points[idx].astype(np.float32)

    # center on centroid, scale so max radial distance = 1
    centroid = sampled.mean(axis=0, keepdims=True)
    sampled = sampled - centroid
    scale = np.max(np.linalg.norm(sampled, axis=1)) + 1e-6
    sampled = sampled / scale

    return sampled


class PointCloudDataset(Dataset):
    """
    Wraps a list of Open3D PointClouds as fixed-size, centered, unit-scale
    numpy arrays ready for the autoencoder.

    Args:
        clusters (list of o3d.geometry.PointCloud): ALL clusters — labeled
            and unlabeled. No labels are read or required.
        n_points (int): Fixed point count per cluster. 256 is a reasonable
            default; raise if clusters routinely have thousands of points
            and fine detail matters, at the cost of slower training.
        seed (int): RNG seed for resampling reproducibility. Default 42.

    Requirements:
        numpy, open3d, torch
    """

    def __init__(self, clusters, n_points=256, seed=42):
        self.n_points = n_points
        self.rng = np.random.default_rng(seed)
        self.arrays = []
        for c in clusters:
            pts = np.asarray(c.points)
            self.arrays.append(_sample_points(pts, n_points, self.rng))

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        return torch.from_numpy(self.arrays[idx])   # (n_points, 3)


# ── model ─────────────────────────────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """
    Minimal PointNet-style encoder: per-point shared MLP + global max-pool.

    No T-Net (spatial transformer) — clusters are already roughly aligned
    (Z is up, from LiDAR/SfM), so the extra alignment-invariance T-Net
    provides in the original PointNet paper isn't needed here and would
    just add more parameters to overfit with limited unlabeled data.

    Args:
        embedding_dim (int): Size of the output bottleneck vector.
            Default 64.

    Requirements:
        torch
    """

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, embedding_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N) for Conv1d
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))          # (B, embedding_dim, N)
        x = torch.max(x, dim=2)[0]            # global max-pool -> (B, embedding_dim)
        return x


class PointNetDecoder(nn.Module):
    """
    Maps a bottleneck embedding back to a fixed-size point set.

    A plain MLP head (not full FoldingNet-style 2D-grid folding) that
    outputs n_points * 3 values directly. Simpler and dependency-free;
    reconstruction quality is "good enough to force the embedding to
    encode real shape information," which is all this needs to be useful
    as an auxiliary feature — it doesn't need to be a state-of-the-art
    generative decoder.

    Args:
        embedding_dim (int): Must match PointNetEncoder's embedding_dim.
        n_points (int): Must match the dataset's n_points.

    Requirements:
        torch
    """

    def __init__(self, embedding_dim=64, n_points=256):
        super().__init__()
        self.n_points = n_points
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, n_points * 3)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_points, 3)


def chamfer_distance(pred, target):
    """
    Symmetric Chamfer distance between two point sets, plain torch
    implementation (no pytorch3d dependency).

    For each point in pred, find its nearest neighbour in target and vice
    versa; average both directions. O(N^2) per sample via a full pairwise
    distance matrix — fine at n_points=256, would need a KDTree-based
    approximation above a few thousand points per cluster.

    Args:
        pred (torch.Tensor): (B, N, 3) reconstructed points.
        target (torch.Tensor): (B, N, 3) ground-truth (input) points.

    Returns:
        torch.Tensor: scalar loss, mean over the batch.

    Requirements:
        torch
    """
    # pairwise squared distances: (B, N, N)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)     # (B, N, N, 3)
    dist2 = (diff ** 2).sum(-1)                         # (B, N, N)

    pred_to_target = dist2.min(dim=2)[0].mean(dim=1)    # (B,)
    target_to_pred = dist2.min(dim=1)[0].mean(dim=1)    # (B,)

    return (pred_to_target + target_to_pred).mean()


# ── training ──────────────────────────────────────────────────────────────────

def train_autoencoder(clusters, n_points=256, embedding_dim=64,
                      epochs=100, batch_size=32, lr=1e-3,
                      device=None, save_path=None, seed=42):
    """
    Train the PointNet autoencoder on a list of clusters (self-supervised —
    no labels used or needed).

    Args:
        clusters (list of o3d.geometry.PointCloud): ALL clusters, labeled
            and unlabeled alike.
        n_points (int): Fixed point count per cluster, see
            PointCloudDataset. Default 256.
        embedding_dim (int): Bottleneck embedding size. Default 64. Larger
            captures more shape detail but risks overfitting the ~3,600
            samples available; 64 is a conservative starting point — try
            32 if reconstruction loss looks noisy/unstable, or 128 if loss
            plateaus early and doesn't seem to have enough capacity.
        epochs (int): Training epochs. Default 100.
        batch_size (int): Default 32.
        lr (float): Adam learning rate. Default 1e-3.
        device (str | None): "cuda" or "cpu". Auto-detects if None.
        save_path (str | None): Path to save the trained encoder + decoder
            state dicts (torch.save). Pass None to skip saving. Default None.
        seed (int): Reproducibility seed for both dataset sampling and
            torch. Default 42.

    Returns:
        encoder (PointNetEncoder): Trained encoder — pass to
            extract_embeddings().
        decoder (PointNetDecoder): Trained decoder — only needed if you
            want to inspect reconstructions, not used downstream.

    Requirements:
        torch, numpy, open3d
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PointNet autoencoder on {device} "
          f"({len(clusters)} clusters, n_points={n_points}, "
          f"embedding_dim={embedding_dim})")

    dataset = PointCloudDataset(clusters, n_points=n_points, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    encoder = PointNetEncoder(embedding_dim=embedding_dim).to(device)
    decoder = PointNetDecoder(embedding_dim=embedding_dim, n_points=n_points).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    encoder.train()
    decoder.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)              # (B, n_points, 3)
            optimizer.zero_grad()

            z = encoder(batch)
            recon = decoder(z)
            loss = chamfer_distance(recon, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:>4}/{epochs}   chamfer loss = {avg_loss:.5f}")

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        torch.save({
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "embedding_dim": embedding_dim,
            "n_points": n_points,
        }, save_path)
        print(f"Saved autoencoder checkpoint -> {save_path}")

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def load_autoencoder(save_path, device=None):
    """
    Reload a previously trained encoder from train_autoencoder()'s
    save_path, so you don't need to retrain on every run.

    Args:
        save_path (str): Path from train_autoencoder()'s save_path arg.
        device (str | None): Auto-detects if None.

    Returns:
        encoder (PointNetEncoder): Loaded, in eval() mode.
        n_points (int): n_points the encoder was trained with — pass this
            to extract_embeddings() so resampling matches training.

    Requirements:
        torch
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(save_path, map_location=device)

    encoder = PointNetEncoder(embedding_dim=ckpt["embedding_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    print(f"Loaded autoencoder encoder <- {save_path} "
          f"(embedding_dim={ckpt['embedding_dim']}, n_points={ckpt['n_points']})")

    return encoder, ckpt["n_points"]


# ── embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings(clusters, encoder, n_points=256, device=None,
                       batch_size=64, seed=42):
    """
    Run every cluster through the trained encoder and return a DataFrame
    of embedding columns, indexed the same way as clusters_to_dataframe()
    (a "file" column = position in the clusters list) so it merges
    directly onto df_clusters / df_deep_clusters.

    Args:
        clusters (list of o3d.geometry.PointCloud): Same clusters list
            (and same order) as used elsewhere in the pipeline.
        encoder (PointNetEncoder): Trained encoder from train_autoencoder()
            or load_autoencoder().
        n_points (int): Must match the value used during training.
        device (str | None): Auto-detects if None.
        batch_size (int): Inference batch size. Default 64.
        seed (int): Must match the seed used for the resampling during
            training for consistency (not strictly required for
            correctness, but keeps embeddings reproducible run-to-run).
            Default 42.

    Returns:
        pd.DataFrame: One row per cluster, columns: "file", "emb_0" ...
            "emb_<embedding_dim - 1>".

    Requirements:
        torch, numpy, open3d, pandas
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()

    dataset = PointCloudDataset(clusters, n_points=n_points, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = encoder(batch)
            all_embeddings.append(z.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)   # (n_clusters, embedding_dim)

    cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
    df = pd.DataFrame(cols)
    df["file"] = np.arange(len(clusters))

    print(f"Extracted embeddings for {len(clusters)} clusters "
          f"(embedding_dim={embeddings.shape[1]})")

    return df