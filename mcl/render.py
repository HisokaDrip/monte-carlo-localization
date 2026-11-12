import numpy as np


def render(grid, particles, truth=None, estimate=None, cell=0.5):
    """ASCII frame: walls '#', particle density . : *, truth R, estimate E (@ if they overlap)."""
    rows = int(np.ceil(grid.height_m / cell))
    cols = int(np.ceil(grid.width_m / cell))
    k = int(round(cell / grid.res))

    occ = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            occ[r, c] = grid.occ[r * k:(r + 1) * k, c * k:(c + 1) * k].any()

    pr = np.clip((particles[:, 1] / cell).astype(int), 0, rows - 1)
    pc = np.clip((particles[:, 0] / cell).astype(int), 0, cols - 1)
    counts = np.zeros((rows, cols), dtype=int)
    np.add.at(counts, (pr, pc), 1)
    frac = counts / max(len(particles), 1)

    canvas = np.full((rows, cols), " ", dtype="<U1")
    canvas[frac > 0] = "."
    canvas[frac > 0.002] = ":"
    canvas[frac > 0.01] = "*"
    canvas[occ] = "#"

    def put(pose, ch):
        r, c = int(pose[1] / cell), int(pose[0] / cell)
        if 0 <= r < rows and 0 <= c < cols:
            canvas[r, c] = "@" if canvas[r, c] in "RE" and canvas[r, c] != ch else ch

    if truth is not None:
        put(truth, "R")
    if estimate is not None:
        put((estimate.x, estimate.y), "E")
    return "\n".join("".join(row) for row in canvas)
