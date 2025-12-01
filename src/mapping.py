import open3d as o3d
import numpy as np
from octree import OctreeNode, insert_point

def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    return pts

def build_octree(points, depth=6, bbox_center=[0,0,0], bbox_size=10.0):
    root = OctreeNode(center=bbox_center, size=bbox_size)
    for p in points:
        insert_point(root, p, depth_limit=depth)
    return root
