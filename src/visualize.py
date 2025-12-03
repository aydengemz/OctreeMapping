import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def collect_voxels(node, voxels):
    if node.is_leaf:
        if node.log_odds > 0:
            voxels.append((node.center, node.size))
        return
    for c in node.children:
        if c is not None:
            collect_voxels(c, voxels)


def draw_octree(root):
    """
    MAC-SAFE visualization of the octree using matplotlib 3D scatter.
    Each occupied leaf is plotted at its center; point size ~ node size.
    """
    voxels = []
    collect_voxels(root, voxels)

    if not voxels:
        print("No occupied voxels to visualize.")
        return

    xs = []
    ys = []
    zs = []
    sizes = []

    for center, size in voxels:
        xs.append(center[0])
        ys.append(center[1])
        zs.append(center[2])
        sizes.append(max(size, 1e-3) * 10.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=sizes, c="red")

    ax.set_title("Octree Occupancy (leaf centers)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()
