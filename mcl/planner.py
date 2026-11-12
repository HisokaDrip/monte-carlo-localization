"""A* on the distance field so the simulated robot has somewhere sensible to drive."""
import heapq
import math

import numpy as np

_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def astar(grid, start, goal, clearance=0.35):
    free = grid.dist > clearance
    (sr, sc), (gr, gc) = [tuple(int(v) for v in grid.to_cell(*p)) for p in (start, goal)]
    if not (grid.inside(gr, gc) and free[gr, gc]):
        return None
    free = free.copy()
    free[sr, sc] = True  # the robot may have bumped closer to a wall than we'd like

    h, w = free.shape
    cost = np.full((h, w), np.inf)
    parent = np.full((h, w), -1, dtype=np.int64)
    cost[sr, sc] = 0.0
    heap = [(0.0, sr, sc)]
    while heap:
        _, r, c = heapq.heappop(heap)
        if (r, c) == (gr, gc):
            break
        base = cost[r, c]
        for dr, dc, step in _MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc]:
                nc_cost = base + step
                if nc_cost < cost[nr, nc]:
                    cost[nr, nc] = nc_cost
                    parent[nr, nc] = r * w + c
                    # octile distance, admissible for 8-connected moves
                    dy, dx = abs(gr - nr), abs(gc - nc)
                    hcost = max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
                    heapq.heappush(heap, (nc_cost + hcost, nr, nc))
    if not np.isfinite(cost[gr, gc]):
        return None

    cells = []
    idx = gr * w + gc
    while idx != -1:
        cells.append(divmod(int(idx), w))
        idx = parent[cells[-1]]
    cells.reverse()
    return [((c + 0.5) * grid.res, (r + 0.5) * grid.res) for r, c in cells]


def segment_clear(grid, a, b, clearance):
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (grid.res * 0.5)) + 1)
    xs = np.linspace(a[0], b[0], n)
    ys = np.linspace(a[1], b[1], n)
    return bool(np.all(grid.lookup_dist(xs, ys) > clearance))


def shortcut(grid, path, clearance=0.3):
    """Greedy line-of-sight smoothing: jump to the furthest point we can see."""
    if not path:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_clear(grid, path[i], path[j], clearance):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan(grid, start, goal, clearance=0.35):
    path = astar(grid, start, goal, clearance)
    if path is None:
        return None
    return shortcut(grid, path, clearance * 0.8)
