import numpy as np
import time


def count_octree_nodes(root):
    """Count ALL nodes (internal + leaves)."""
    stack = [root]
    count = 0
    while stack:
        node = stack.pop()
        count += 1
        if not node.is_leaf:
            for c in node.children:
                if c is not None:
                    stack.append(c)
    return count


def count_octree_occupied_leaves(root):
    """Count only leaf nodes whose log-odds > 0 (occupied)."""
    stack = [root]
    count = 0
    while stack:
        node = stack.pop()
        if node.is_leaf:
            if node.log_odds > 0:
                count += 1
        else:
            for c in node.children:
                if c is not None:
                    stack.append(c)
    return count


def bbox(points):
    """Compute bounding box of point cloud."""
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    size = maxs - mins
    return mins, maxs, size


def estimate_octree_memory(root):
    """
    Estimate memory usage of octree in bytes.
    Each node has:
    - center: 3 floats (24 bytes)
    - size: 1 float (8 bytes)
    - log_odds: 1 float (8 bytes)
    - is_leaf: 1 bool (1 byte, but Python objects have overhead)
    - children: 8 pointers (64 bytes on 64-bit system)
    - Python object overhead: ~56 bytes
    
    Rough estimate: ~200 bytes per node
    """
    n_nodes = count_octree_nodes(root)
    return n_nodes * 200


def print_metrics(name, t_start, t_end, **kwargs):
    print("\n========== {} METRICS ==========".format(name))
    print("Runtime: {:.6f} sec".format(t_end - t_start))
    for key, val in kwargs.items():
        print("{}: {}".format(key, val))
    print("================================\n")
