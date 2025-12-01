import time
import open3d as o3d
from mapping import load_point_cloud, build_octree
from compare import dense_voxel_grid
from metrics import count_octree_nodes, count_octree_occupied_leaves
from plots import plot_runtime, plot_memory, plot_resolution

def run_benchmarks():
    dataset = o3d.data.PCDPointCloud()
    pts = load_point_cloud(dataset.path)

    depths = [3, 4, 5, 6, 7]
    runtimes_octree = []
    runtimes_dense = []
    octree_nodes = []
    octree_leaves = []
    dense_vox = []

    # Dense voxel grid once (same for all depths)
    t0 = time.time()
    grid = dense_voxel_grid(pts, voxel_size=0.1)
    t1 = time.time()
    dense_runtime = t1 - t0
    dense_voxel_count = len(grid.get_voxels())

    # Fill arrays for plotting
    for d in depths:
        # Octree
        t2 = time.time()
        octree_root = build_octree(pts, depth=d, bbox_center=[0,0,0], bbox_size=10.0)
        t3 = time.time()

        octree_nodes.append(count_octree_nodes(octree_root))
        octree_leaves.append(count_octree_occupied_leaves(octree_root))
        runtimes_octree.append(t3 - t2)

        # Dense (same each time)
        runtimes_dense.append(dense_runtime)
        dense_vox.append(dense_voxel_count)

    # Make plots
    plot_runtime(depths, runtimes_octree, runtimes_dense)
    plot_memory(depths, octree_leaves, dense_vox)
    plot_resolution(depths, octree_nodes)

    print("All plots saved under results/")

if __name__ == "__main__":
    run_benchmarks()
