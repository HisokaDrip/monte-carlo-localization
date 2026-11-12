"""Occupancy grid + Euclidean distance field.

The distance field does double duty: it *is* the likelihood field for the
sensor model, and it lets the simulator sphere-trace lidar rays instead of
marching them one cell at a time.
"""
from pathlib import Path

import numpy as np

_BIG = 1e12  # stand-in for infinity; real inf turns the parabola maths into nan


def _edt_1d(f):
    # Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions".
    # Lower envelope of parabolas rooted at each sample, O(n).
    n = len(f)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1)
    d = np.empty(n)
    k = 0
    z[0], z[1] = -np.inf, np.inf
    for q in range(1, n):
        fq = f[q] + q * q
        s = (fq - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = (fq - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def distance_transform(occ):
    """Exact Euclidean distance (in cells) from every cell to the nearest occupied one."""
    f = np.where(occ, 0.0, _BIG)
    cols = np.empty_like(f)
    for c in range(f.shape[1]):
        cols[:, c] = _edt_1d(f[:, c])
    for r in range(f.shape[0]):
        f[r, :] = _edt_1d(cols[r, :])
    return np.sqrt(f)


class GridMap:
    """Rows grow with y, columns with x. (0, 0) is the top-left corner of the map."""

    def __init__(self, occ, resolution):
        self.occ = np.asarray(occ, dtype=bool)
        self.res = float(resolution)
        self.h, self.w = self.occ.shape
        self.width_m = self.w * self.res
        self.height_m = self.h * self.res
        self.dist = distance_transform(self.occ) * self.res

    @classmethod
    def from_ascii(cls, text, cell_size=0.5, upsample=5):
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        width = max(len(ln) for ln in lines)
        coarse = np.array([[ch == "#" for ch in ln.ljust(width)] for ln in lines])
        occ = np.kron(coarse, np.ones((upsample, upsample), dtype=bool))
        return cls(occ, cell_size / upsample)

    @classmethod
    def load(cls, path, **kw):
        return cls.from_ascii(Path(path).read_text(), **kw)

    def to_cell(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.floor(y / self.res).astype(int), np.floor(x / self.res).astype(int)

    def inside(self, r, c):
        return (r >= 0) & (r < self.h) & (c >= 0) & (c < self.w)

    def lookup_dist(self, x, y, outside=0.0):
        r, c = self.to_cell(np.atleast_1d(x), np.atleast_1d(y))
        ok = self.inside(r, c)
        out = np.full(r.shape, outside, dtype=float)
        out[ok] = self.dist[r[ok], c[ok]]
        return out

    def is_free(self, x, y):
        r, c = self.to_cell(np.atleast_1d(x), np.atleast_1d(y))
        ok = self.inside(r, c)
        out = np.zeros(r.shape, dtype=bool)
        out[ok] = ~self.occ[r[ok], c[ok]]
        return out

    def sample_free(self, n, rng, clearance=0.0):
        cand = np.flatnonzero(self.dist.ravel() > clearance)
        idx = rng.choice(cand, n)
        r, c = np.divmod(idx, self.w)
        return (c + rng.random(n)) * self.res, (r + rng.random(n)) * self.res

    def raycast(self, x, y, angles, max_range):
        """Range to the first occupied cell along each angle, capped at max_range."""
        angles = np.asarray(angles, dtype=float)
        dx, dy = np.cos(angles), np.sin(angles)
        t = np.zeros_like(angles)
        done = np.zeros(angles.shape, dtype=bool)
        # dist is measured between cell centres, so a point can be up to ~1.2 cells
        # closer to a wall than the field says. Step a bit short of it, and fall back
        # to half-cell marching once we're hugging the wall.
        slack = 1.5 * self.res
        for _ in range(400):
            d = self.lookup_dist(x + t * dx, y + t * dy, outside=0.0)
            done |= (d <= 0.0) | (t >= max_range)
            if done.all():
                break
            step = np.maximum(d - slack, 0.5 * self.res)
            t = np.where(done, t, t + step)
        return np.minimum(t, max_range)
