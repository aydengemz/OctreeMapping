import os
import time

os.environ["OPEN3D_DISABLE_VISUALIZATION"] = "1"

import open3d as o3d
import numpy as np

from mapping import load_point_cloud, build_octree
from compare import dense_voxel_grid, visualize_voxel_grid
from visualize import draw_octree
from metrics import (
    count_octree_nodes,
    count_octree_occupied_leaves,
    bbox,
    print_metrics,
    estimate_octree_memory,
)
from plots import plot_runtime_comparison


def generate_sparse_world(
    n_objects: int = 200,
    points_per_object: int = 300,
    world_size: float = 100.0,
    object_radius: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a large, mostly empty 3D world with small occupied regions.

    - world_size: length of the cube (e.g., 100m -> [-50, 50] in each axis)
    - n_objects: number of small local clusters (e.g., obstacles)
    - points_per_object: number of points per cluster
    """
    half = world_size / 2.0

    centers = np.random.uniform(-half, half, size=(n_objects, 3))

    pts_list = []
    for c in centers:
        sphere = c + np.random.normal(scale=object_radius, size=(points_per_object, 3))
        pts_list.append(sphere)

    pts = np.vstack(pts_list)
    bbox_center = np.array([0.0, 0.0, 0.0])
    bbox_size = world_size

    return pts, bbox_center, bbox_size


def run_experiment(name, pts, bbox_center, bbox_size, voxel_size=0.1, depth=6):
    """
    Run dense voxel grid + octree on a given point cloud and bounding box.
    Prints metrics + comparative analysis.
    """
    print(f"\n================== {name} ==================\n")
    print("Point cloud has", len(pts), "points.")
    print("Bounding box center:", bbox_center)
    print("Bounding box size:", bbox_size)
    print("Voxel size:", voxel_size)

    t0 = time.time()
    grid = dense_voxel_grid(
        pts, voxel_size=voxel_size, bbox_center=bbox_center, bbox_size=bbox_size
    )
    t1 = time.time()

    voxel_count = grid.get_voxel_count()
    dense_memory_sparse = grid.estimate_memory_bytes()
    dense_runtime = t1 - t0

    n_side = int(np.ceil(bbox_size / voxel_size))
    theoretical_dense_cells = n_side**3
    bytes_per_voxel = 50
    theoretical_dense_memory = theoretical_dense_cells * bytes_per_voxel

    print_metrics(
        "Dense Voxel Grid (Python, sparse dict)",
        t0,
        t1,
        voxel_count=voxel_count,
        voxel_size=voxel_size,
        estimated_sparse_memory_bytes=dense_memory_sparse,
        estimated_sparse_memory_mb=f"{dense_memory_sparse / (1024*1024):.2f} MB (sparse, occupied only)",
        theoretical_dense_cells=theoretical_dense_cells,
        theoretical_dense_memory_bytes=theoretical_dense_memory,
        theoretical_dense_memory_mb=f"{theoretical_dense_memory / (1024*1024):.2f} MB (theoretical full grid)",
    )

    t2 = time.time()
    octree_root = build_octree(
        pts, depth=depth, bbox_center=bbox_center, bbox_size=bbox_size
    )
    t3 = time.time()

    n_nodes = count_octree_nodes(octree_root)
    n_occ = count_octree_occupied_leaves(octree_root)
    octree_memory = estimate_octree_memory(octree_root)
    octree_runtime = t3 - t2

    print_metrics(
        "Octree (Python)",
        t2,
        t3,
        total_nodes=n_nodes,
        occupied_leaves=n_occ,
        depth_limit=depth,
        estimated_memory_bytes=octree_memory,
        estimated_memory_mb=f"{octree_memory / (1024*1024):.2f} MB (estimated)",
    )

    print("\n" + "=" * 60)
    print(f"COMPARATIVE ANALYSIS — {name}")
    print("=" * 60)

    print("\nRuntime Comparison:")
    print(f"  Dense Grid:  {dense_runtime:.6f} sec")
    print(f"  Octree:      {octree_runtime:.6f} sec")
    speed_ratio = dense_runtime / octree_runtime if octree_runtime > 0 else float("inf")
    print(
        f"  Speedup:     {speed_ratio:.2f}x "
        f"{'(Dense faster)' if dense_runtime < octree_runtime else '(Octree faster)'}"
    )

    print("\nEstimated Memory Usage (Sparse Structures):")
    print(
        f"  Dense Grid (sparse dict): {voxel_count:,} voxels "
        f"({dense_memory_sparse / (1024*1024):.2f} MB estimated)"
    )
    print(
        f"  Octree:                    {n_nodes:,} total nodes, {n_occ:,} occupied leaves "
        f"({octree_memory / (1024*1024):.2f} MB estimated)"
    )
    sparse_memory_ratio = (
        dense_memory_sparse / octree_memory if octree_memory > 0 else float("inf")
    )
    print(
        f"  Sparse Memory Ratio: {sparse_memory_ratio:.2f}x "
        f"{'(Octree more efficient)' if octree_memory < dense_memory_sparse else '(Dense more efficient)'}"
    )

    print("\nTheoretical FULL Dense Grid (for this bbox & voxel size):")
    print(f"  Total cells:      {theoretical_dense_cells:,}")
    print(
        f"  Theoretical mem:  {theoretical_dense_memory / (1024*1024):.2f} MB "
        "(if we actually allocated full 3D grid)"
    )
    occ_fraction = voxel_count / theoretical_dense_cells
    print(f"  Occupancy fraction: {occ_fraction * 100:.6f}% of cells actually used")
    print(
        f"  Octree nodes:       {n_nodes:,} (stores only explored/occupied areas instead of all {theoretical_dense_cells:,} cells)"
    )
    compression = theoretical_dense_cells / max(n_nodes, 1)
    print(
        f"  Structural compression: Octree stores ~{compression:.1f}x fewer cells than a full dense grid would."
    )

    print("\nBuild Time per Node/Cell (Sparse Representations):")
    print(f"  Dense Grid:  {dense_runtime / voxel_count * 1e6:.2f} microseconds/voxel")
    print(
        f"  Octree:      {octree_runtime / n_nodes * 1e6:.2f} microseconds/node "
        "(includes internal + leaf nodes)"
    )

    print("=" * 60 + "\n")
    
    plot_runtime_comparison(
        dense_runtime=dense_runtime,
        octree_runtime=octree_runtime,
        dataset_name=name,
        filename=f"results/runtime_comparison_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    )


def main():
    dataset = o3d.data.PCDPointCloud()
    pcd_path = dataset.path
    print("Loading real point cloud from:", pcd_path)
    real_pts = load_point_cloud(pcd_path)

    mins, maxs, extent = bbox(real_pts)
    bbox_center_real = (mins + maxs) / 2.0
    bbox_size_real = float(np.max(extent))

    run_experiment(
        name="Real Dataset (fragment.pcd)",
        pts=real_pts,
        bbox_center=bbox_center_real,
        bbox_size=bbox_size_real,
        voxel_size=0.1,
        depth=6,
    )

    sparse_pts, bbox_center_sparse, bbox_size_sparse = generate_sparse_world(
        n_objects=200,
        points_per_object=300,
        world_size=100.0,
        object_radius=0.3,
    )

    run_experiment(
        name="Synthetic Large Sparse World",
        pts=sparse_pts,
        bbox_center=bbox_center_sparse,
        bbox_size=bbox_size_sparse,
        voxel_size=0.1,
        depth=6,
    )


if __name__ == "__main__":
    main()
