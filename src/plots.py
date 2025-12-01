import matplotlib.pyplot as plt

def plot_runtime(depths, runtimes_octree, runtimes_dense):
    plt.figure()
    plt.plot(depths, runtimes_octree, marker='o', label="Octree Runtime")
    plt.plot(depths, runtimes_dense, marker='o', label="Dense Voxel Grid Runtime")
    plt.xlabel("Octree Depth")
    plt.ylabel("Runtime (sec)")
    plt.title("Runtime vs Depth")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/runtime_vs_depth.png")
    plt.close()

def plot_memory(depths, octree_leaves, dense_voxels):
    plt.figure()
    plt.plot(depths, octree_leaves, marker='o', label="Octree Occupied Leaves")
    plt.plot(depths, dense_voxels, marker='o', label="Dense Voxel Count")
    plt.xlabel("Depth")
    plt.ylabel("Nodes / Voxels")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/memory_vs_depth.png")
    plt.close()

def plot_resolution(depths, octree_nodes):
    plt.figure()
    plt.plot(depths, octree_nodes, marker='o', label="Octree Total Nodes")
    plt.xlabel("Depth")
    plt.ylabel("Total Nodes Allocated")
    plt.title("Resolution Adaptability (Octree)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/octree_resolution.png")
    plt.close()
