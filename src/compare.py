# compare.py
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def dense_voxel_grid(points, voxel_size=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid


def visualize_voxel_grid(voxel_grid):
    """
    MAC-SAFE visualization using matplotlib only.
    Plots voxel centers as a 3D scatter instead of opening an Open3D window.
    """
    voxels = voxel_grid.get_voxels()
    if not voxels:
        print("No voxels to visualize.")
        return

    xs = []
    ys = []
    zs = []

    for v in voxels:
        # v.grid_index is integer index; convert to float for plotting
        gx, gy, gz = v.grid_index
        xs.append(gx)
        ys.append(gy)
        zs.append(gz)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=2)

    ax.set_title("Dense Voxel Grid (centers)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()
