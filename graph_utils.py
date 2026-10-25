"""Pure NumPy graph helpers (no torch needed). Used for early-stage truncation and drawing."""
from collections import deque
import numpy as np


def bfs_depth(edge_index, n_nodes, root):
    """Hop distance of every node from the root (-1 = unreachable). edge_index: array shape (2, E)."""
    adj = [[] for _ in range(n_nodes)]
    for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[s].append(t)
    depth = np.full(n_nodes, -1, dtype=np.int64)
    depth[root] = 0
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


def truncate(x, edge_index, depth, k):
    """Keep only nodes within k hops of the root (the 'early' part of the cascade).
    k=None keeps the full graph (all nodes reachable from the root).
    Returns (x_new, edge_index_new, keep_mask)."""
    keep = depth >= 0 if k is None else (depth >= 0) & (depth <= k)
    new_id = np.cumsum(keep) - 1
    edge_keep = keep[edge_index[0]] & keep[edge_index[1]]
    new_edges = new_id[edge_index[:, edge_keep]]
    return x[keep], new_edges, keep


def radial_layout(depth):
    """Root in the centre, one ring per hop level. Returns (n, 2) positions."""
    n = len(depth)
    pos = np.zeros((n, 2))
    max_d = int(depth.max()) if n else 0
    for d in range(1, max_d + 1):
        idx = np.where(depth == d)[0]
        if len(idx) == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, len(idx), endpoint=False) + d * 0.35
        pos[idx, 0] = d * np.cos(angles)
        pos[idx, 1] = d * np.sin(angles)
    return pos
