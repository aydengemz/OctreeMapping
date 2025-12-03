import os
import time
import numpy as np

os.environ["OPEN3D_DISABLE_VISUALIZATION"] = "1"

import open3d as o3d
from mapping import load_point_cloud, build_octree
from metrics import bbox
from compare import dense_voxel_grid
from plots import plot_runtime_comparison, plot_runtime


def generate_runtime_comparison_graphs():
    """Generate runtime comparison graphs for both datasets."""
    
    dataset = o3d.data.PCDPointCloud()
    real_pts = load_point_cloud(dataset.path)
    mins, maxs, extent = bbox(real_pts)
    bbox_center_real = (mins + maxs) / 2.0
    bbox_size_real = float(np.max(extent))
    
    print("="*70)
    print("GENERATING RUNTIME COMPARISON GRAPHS")
    print("="*70)
    print(f"\nReal Dataset: {len(real_pts):,} points")
    print(f"Bounding box size: {bbox_size_real:.2f}")
    
    depths = [3, 4, 5, 6, 7]
    real_octree_runtimes = []
    real_dense_runtimes = []
    
    print("\nBuilding dense voxel grid...")
    t0 = time.time()
    real_grid = dense_voxel_grid(real_pts, voxel_size=0.1, 
                                  bbox_center=bbox_center_real, 
                                  bbox_size=bbox_size_real)
    t1 = time.time()
    real_dense_runtime = t1 - t0
    print(f"  Dense grid runtime: {real_dense_runtime:.4f} sec")
    
    print("\nBuilding octrees at different depths...")
    for d in depths:
        t0 = time.time()
        octree_root = build_octree(real_pts, depth=d, 
                                   bbox_center=bbox_center_real, 
                                   bbox_size=bbox_size_real)
        t1 = time.time()
        runtime = t1 - t0
        real_octree_runtimes.append(runtime)
        real_dense_runtimes.append(real_dense_runtime)
        print(f"  Depth {d}: {runtime:.4f} sec")
    
    print("\nGenerating runtime comparison graphs...")
    
    plot_runtime_comparison(
        dense_runtime=real_dense_runtime,
        octree_runtime=real_octree_runtimes[depths.index(6)],
        dataset_name="Real Dataset (Depth 6)",
        filename="results/runtime_comparison_real_depth6.png"
    )
    
    plot_runtime(
        depths, real_octree_runtimes, real_dense_runtimes,
        filename="results/runtime_vs_depth_real.png"
    )
    
    print("\n✓ Runtime comparison graphs generated!")
    print("  - results/runtime_comparison_real_depth6.png")
    print("  - results/runtime_vs_depth_real.png")
    print("="*70)


if __name__ == "__main__":
    generate_runtime_comparison_graphs()

