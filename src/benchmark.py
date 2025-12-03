import time
import os
import numpy as np
os.environ['OPEN3D_DISABLE_VISUALIZATION'] = '1'
import open3d as o3d
from mapping import load_point_cloud, build_octree
from compare import dense_voxel_grid
from metrics import count_octree_nodes, count_octree_occupied_leaves, estimate_octree_memory, bbox


def generate_sparse_world(
    n_objects: int = 200,
    points_per_object: int = 300,
    world_size: float = 100.0,
    object_radius: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a large, mostly empty 3D world with small occupied regions.
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

def run_single_benchmark(pts, bbox_center, bbox_size, dataset_name, voxel_size=0.1):
    """Run benchmark for a single dataset across multiple depths."""
    depths = [3, 4, 5, 6, 7]
    runtimes_octree = []
    runtimes_dense = []
    octree_nodes = []
    octree_leaves = []
    dense_vox = []
    octree_memory = []
    dense_memory = []
    theoretical_dense_cells = []
    build_time_per_cell_octree = []
    build_time_per_cell_dense = []
    memory_per_cell_octree = []
    memory_per_cell_dense = []

    print("="*70)
    print(f"BENCHMARKING: {dataset_name}")
    print("="*70)
    print(f"Point cloud: {len(pts):,} points")
    print(f"Bounding box size: {bbox_size:.2f}")
    print(f"Voxel size: {voxel_size}")
    print()

    print("Building dense voxel grid...")
    t0 = time.time()
    grid = dense_voxel_grid(pts, voxel_size=voxel_size, bbox_center=bbox_center, bbox_size=bbox_size)
    t1 = time.time()
    dense_runtime = t1 - t0
    dense_voxel_count = grid.get_voxel_count()
    dense_mem = grid.estimate_memory_bytes()
    
    n_side = int(np.ceil(bbox_size / voxel_size))
    theoretical_cells = n_side**3
    
    print(f"  Dense grid (sparse): {dense_voxel_count:,} occupied voxels in {dense_runtime:.4f} sec")
    print(f"  Theoretical full grid: {theoretical_cells:,} cells")
    print(f"  Occupancy: {dense_voxel_count / theoretical_cells * 100:.6f}%")
    print()

    print("Building octrees at different depths...")
    for d in depths:
        t2 = time.time()
        octree_root = build_octree(pts, depth=d, bbox_center=bbox_center, bbox_size=bbox_size)
        t3 = time.time()

        n_nodes = count_octree_nodes(octree_root)
        n_occ = count_octree_occupied_leaves(octree_root)
        octree_mem = estimate_octree_memory(octree_root)
        octree_rt = t3 - t2

        octree_nodes.append(n_nodes)
        octree_leaves.append(n_occ)
        runtimes_octree.append(octree_rt)
        octree_memory.append(octree_mem)
        
        runtimes_dense.append(dense_runtime)
        dense_vox.append(dense_voxel_count)
        dense_memory.append(dense_mem)
        theoretical_dense_cells.append(theoretical_cells)
        
        build_time_per_cell_octree.append(octree_rt / n_nodes * 1e6 if n_nodes > 0 else 0)
        build_time_per_cell_dense.append(dense_runtime / dense_voxel_count * 1e6 if dense_voxel_count > 0 else 0)
        memory_per_cell_octree.append(octree_mem / n_nodes if n_nodes > 0 else 0)
        memory_per_cell_dense.append(dense_mem / dense_voxel_count if dense_voxel_count > 0 else 0)
        
        print(f"  Depth {d}: {n_nodes:,} nodes, {n_occ:,} occupied leaves, {octree_rt:.4f} sec")

    return {
        'depths': depths,
        'runtimes_octree': runtimes_octree,
        'runtimes_dense': runtimes_dense,
        'octree_nodes': octree_nodes,
        'octree_leaves': octree_leaves,
        'dense_vox': dense_vox,
        'octree_memory': octree_memory,
        'dense_memory': dense_memory,
        'theoretical_dense_cells': theoretical_dense_cells,
        'build_time_per_cell_octree': build_time_per_cell_octree,
        'build_time_per_cell_dense': build_time_per_cell_dense,
        'memory_per_cell_octree': memory_per_cell_octree,
        'memory_per_cell_dense': memory_per_cell_dense,
        'dataset_name': dataset_name,
        'dense_voxel_count': dense_voxel_count,
        'dense_mem': dense_mem,
        'dense_runtime': dense_runtime,
    }


def run_benchmarks():
    """Run benchmarks on both real and synthetic sparse datasets."""
    dataset = o3d.data.PCDPointCloud()
    real_pts = load_point_cloud(dataset.path)
    mins, maxs, extent = bbox(real_pts)
    bbox_center_real = (mins + maxs) / 2.0
    bbox_size_real = float(np.max(extent))
    
    sparse_pts, bbox_center_sparse, bbox_size_sparse = generate_sparse_world(
        n_objects=200,
        points_per_object=300,
        world_size=100.0,
        object_radius=0.3,
    )
    
    real_data = run_single_benchmark(
        real_pts, bbox_center_real, bbox_size_real, 
        "Real Dataset (fragment.pcd)", voxel_size=0.1
    )
    
    print("\n" + "="*70 + "\n")
    
    sparse_data = run_single_benchmark(
        sparse_pts, bbox_center_sparse, bbox_size_sparse,
        "Synthetic Sparse World", voxel_size=0.1
    )
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for data in [real_data, sparse_data]:
        name = data['dataset_name']
        best_idx = np.argmin(data['octree_memory'])
        best_depth = data['depths'][best_idx]
        
        print(f"\n{name} - Best Octree Depth (by memory): {best_depth}")
        print(f"  Total nodes: {data['octree_nodes'][best_idx]:,}")
        print(f"  Occupied leaves: {data['octree_leaves'][best_idx]:,}")
        print(f"  Estimated memory: {data['octree_memory'][best_idx] / (1024*1024):.2f} MB")
        print(f"  Runtime: {data['runtimes_octree'][best_idx]:.4f} sec")
        print(f"  Dense voxels: {data['dense_voxel_count']:,}")
        print(f"  Theoretical dense cells: {data['theoretical_dense_cells'][0]:,}")
        compression = data['theoretical_dense_cells'][0] / max(data['octree_nodes'][best_idx], 1)
        print(f"  Compression vs full grid: {compression:.1f}x")
    
    print("="*70)
    print()

    from plots import (
        plot_runtime, plot_memory, plot_resolution,
        plot_build_time_per_cell, plot_memory_per_cell, plot_efficiency_comparison,
        plot_dataset_comparison, plot_theoretical_dense_comparison
    )
    
    print("Generating plots for Real Dataset...")
    plot_runtime(real_data['depths'], real_data['runtimes_octree'], real_data['runtimes_dense'], 
                filename="results/runtime_vs_depth_real.png")
    plot_memory(real_data['depths'], real_data['octree_leaves'], real_data['dense_vox'],
               filename="results/memory_vs_depth_real.png")
    plot_resolution(real_data['depths'], real_data['octree_nodes'],
                   filename="results/octree_resolution_real.png")
    plot_efficiency_comparison(real_data['depths'], real_data['octree_memory'], 
                              real_data['dense_memory'], real_data['octree_leaves'], 
                              real_data['dense_vox'],
                              filename="results/efficiency_comparison_real.png")
    
    print("Generating plots for Sparse World...")
    plot_runtime(sparse_data['depths'], sparse_data['runtimes_octree'], sparse_data['runtimes_dense'],
                filename="results/runtime_vs_depth_sparse.png")
    plot_memory(sparse_data['depths'], sparse_data['octree_leaves'], sparse_data['dense_vox'],
               filename="results/memory_vs_depth_sparse.png")
    plot_resolution(sparse_data['depths'], sparse_data['octree_nodes'],
                   filename="results/octree_resolution_sparse.png")
    plot_efficiency_comparison(sparse_data['depths'], sparse_data['octree_memory'],
                              sparse_data['dense_memory'], sparse_data['octree_leaves'],
                              sparse_data['dense_vox'],
                              filename="results/efficiency_comparison_sparse.png")
    
    print("Generating comparison plots...")
    plot_dataset_comparison(real_data, sparse_data)
    plot_theoretical_dense_comparison(real_data, sparse_data)
    
    print("All plots saved under results/")

if __name__ == "__main__":
    run_benchmarks()
