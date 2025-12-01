import time
import open3d as o3d
import numpy as np
from mapping import load_point_cloud, build_octree
from compare import dense_voxel_grid, visualize_voxel_grid
from visualize import draw_octree
from metrics import (
    count_octree_nodes,
    count_octree_occupied_leaves,
    bbox,
    print_metrics
)

def main():
    dataset = o3d.data.PCDPointCloud()
    pcd_path = dataset.path
    print("Loading point cloud from:", pcd_path)
    pts = load_point_cloud(pcd_path)

    print("Point cloud has", len(pts), "points.")

    mins, maxs, extent = bbox(pts)
    print("Point cloud bbox:")
    print("  Min:", mins)
    print("  Max:", maxs)
    print("  Size:", extent)

    # -------------------------------
    # DENSE VOXEL GRID
    # -------------------------------
    t0 = time.time()
    grid = dense_voxel_grid(pts, voxel_size=0.1)
    t1 = time.time()

    voxel_count = len(grid.get_voxels())

    print_metrics(
        "Dense Voxel Grid",
        t0, t1,
        voxel_count=voxel_count,
        voxel_size=0.1
    )

    visualize_voxel_grid(grid)

    # -------------------------------
    # OCTREE
    # -------------------------------
    t2 = time.time()
    octree_root = build_octree(pts, depth=6, bbox_center=[0, 0, 0], bbox_size=10.0)
    t3 = time.time()

    n_nodes = count_octree_nodes(octree_root)
    n_occ = count_octree_occupied_leaves(octree_root)

    print_metrics(
        "Octree",
        t2, t3,
        total_nodes=n_nodes,
        occupied_leaves=n_occ,
        depth_limit=6
    )

    draw_octree(octree_root)


if __name__ == "__main__":
    main()
