import numpy as np
import math

L_OCC = math.log(0.7 / (1 - 0.7))
L_FREE = math.log(0.4 / (1 - 0.4))
CLAMP_MAX = 3.5
CLAMP_MIN = -3.5


class OctreeNode:
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        self.children = [None] * 8
        self.log_odds = 0.0
        self.is_leaf = True

    def subdivide(self):
        half = self.size / 2
        quarter = self.size / 4

        offsets = np.array([
            [x, y, z]
            for x in [-quarter, quarter]
            for y in [-quarter, quarter]
            for z in [-quarter, quarter]
        ])

        self.children = [
            OctreeNode(self.center + off, half) for off in offsets
        ]
        self.is_leaf = False


def update_log_odds(node, is_occupied):
    if is_occupied:
        node.log_odds += L_OCC
    else:
        node.log_odds += L_FREE

    node.log_odds = min(max(node.log_odds, CLAMP_MIN), CLAMP_MAX)


def insert_point(root, point, depth_limit=6):
    node = root

    for _ in range(depth_limit):
        if node.is_leaf:
            node.subdivide()

        idx = 0
        idx |= (point[0] > node.center[0])
        idx |= (point[1] > node.center[1]) << 1
        idx |= (point[2] > node.center[2]) << 2

        node = node.children[idx]

    update_log_odds(node, True)
