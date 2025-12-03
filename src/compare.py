import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DenseVoxelGrid:
    """
    Pure Python implementation of a dense voxel grid.
    Uses a dictionary to store occupied voxels (voxel_index -> count).
    """
    def __init__(self, points, voxel_size=0.1, bbox_center=None, bbox_size=None):
        self.voxel_size = voxel_size
        self.voxels = {}
        
        if bbox_center is None or bbox_size is None:
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            bbox_center = (mins + maxs) / 2.0
            bbox_size = np.max(maxs - mins)
        
        self.bbox_center = np.array(bbox_center)
        self.bbox_size = bbox_size
        
        half_size = bbox_size / 2.0
        self.grid_min = self.bbox_center - half_size
        self.grid_max = self.bbox_center + half_size
        
        self._build_grid(points)
    
    def _point_to_voxel_index(self, point):
        """Convert a 3D point to voxel grid indices (i, j, k)."""
        offset = point - self.grid_min
        indices = np.floor(offset / self.voxel_size).astype(int)
        return tuple(indices)
    
    def _voxel_index_to_center(self, voxel_idx):
        """Convert voxel indices back to world coordinates (center of voxel)."""
        i, j, k = voxel_idx
        center = self.grid_min + (np.array([i, j, k]) + 0.5) * self.voxel_size
        return center
    
    def _build_grid(self, points):
        """Build the voxel grid by inserting all points."""
        for point in points:
            voxel_idx = self._point_to_voxel_index(point)
            self.voxels[voxel_idx] = self.voxels.get(voxel_idx, 0) + 1
    
    def get_voxels(self):
        """Return list of voxel indices that are occupied."""
        return list(self.voxels.keys())
    
    def get_voxel_count(self):
        """Return the number of occupied voxels."""
        return len(self.voxels)
    
    def get_total_points(self):
        """Return total number of points stored in voxels."""
        return sum(self.voxels.values())
    
    def get_voxel_centers(self):
        """Return list of world-space centers of occupied voxels."""
        centers = []
        for voxel_idx in self.voxels.keys():
            center = self._voxel_index_to_center(voxel_idx)
            centers.append(center)
        return np.array(centers)
    
    def estimate_memory_bytes(self):
        """
        Estimate memory usage in bytes.
        Each voxel entry: 3 ints (24 bytes) + 1 int for count (8 bytes) + dict overhead
        Rough estimate: ~50 bytes per voxel entry
        
        Note: This only counts OCCUPIED voxels (sparse storage), not the entire grid.
        For a truly dense grid, memory would be: (grid_size/voxel_size)^3 * bytes_per_voxel
        """
        return len(self.voxels) * 50


def dense_voxel_grid(points, voxel_size=0.1, bbox_center=None, bbox_size=None):
    """
    Create a dense voxel grid from points (pure Python implementation).
    
    Args:
        points: numpy array of shape (N, 3)
        voxel_size: size of each voxel
        bbox_center: optional center of bounding box
        bbox_size: optional size of bounding box
    
    Returns:
        DenseVoxelGrid object
    """
    return DenseVoxelGrid(points, voxel_size, bbox_center, bbox_size)


def visualize_voxel_grid(voxel_grid):
    """
    MAC-SAFE visualization using matplotlib only.
    Plots voxel centers as a 3D scatter instead of opening an Open3D window.
    """
    if isinstance(voxel_grid, DenseVoxelGrid):
        centers = voxel_grid.get_voxel_centers()
        if len(centers) == 0:
            print("No voxels to visualize.")
            return
        
        xs = centers[:, 0]
        ys = centers[:, 1]
        zs = centers[:, 2]
    else:
        voxels = voxel_grid.get_voxels()
        if not voxels:
            print("No voxels to visualize.")
            return
        
        xs = []
        ys = []
        zs = []
        
        for v in voxels:
            gx, gy, gz = v.grid_index
            xs.append(gx)
            ys.append(gy)
            zs.append(gz)
        
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=2, c='blue', alpha=0.6)

    ax.set_title("Dense Voxel Grid (centers)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()
