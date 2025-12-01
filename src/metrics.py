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


def print_metrics(name, t_start, t_end, **kwargs):
    print("\n========== {} METRICS ==========".format(name))
    print("Runtime: {:.6f} sec".format(t_end - t_start))
    for key, val in kwargs.items():
        print("{}: {}".format(key, val))
    print("================================\n")
