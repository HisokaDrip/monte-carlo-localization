from dataclasses import dataclass

import numpy as np

from .models import LidarModel, MotionNoise, likelihood_field, sample_motion, wrap


def low_variance_resample(weights, m, rng):
    """Systematic resampling: one random offset, m evenly spaced pointers.

    Lower variance than m independent draws, and O(m) after the cumsum.
    """
    positions = (rng.random() + np.arange(m)) / m
    cum = np.cumsum(weights)
    cum[-1] = 1.0  # float drift would otherwise occasionally index past the end
    return np.searchsorted(cum, positions, side="right")


@dataclass
class Estimate:
    x: float
    y: float
    theta: float
    confidence: float  # weight mass inside the winning cluster


class MonteCarloLocalizer:
    """Augmented MCL (Thrun et al. table 8.3) with a likelihood-field sensor model.

    alpha_slow / alpha_fast track short and long term averages of how well the
    belief explains each scan. When the short term one falls below the long
    term one, the robot has probably been picked up and moved, so a slice of
    the particles gets thrown back out uniformly over the map.
    """

    def __init__(self, grid, n_particles=3000, motion=None, lidar=None,
                 alpha_slow=0.02, alpha_fast=0.3, max_inject=0.2, seed=None):
        self.grid = grid
        self.n = n_particles
        self.motion = motion or MotionNoise()
        self.lidar = lidar or LidarModel()
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.max_inject = max_inject
        self.rng = np.random.default_rng(seed)

        self.particles = self._random_particles(self.n)
        self.weights = np.full(self.n, 1.0 / self.n)
        self.w_slow = None
        self.w_fast = None
        self.injected = 0
        self.fit = 0.0

    def _random_particles(self, n):
        x, y = self.grid.sample_free(n, self.rng, clearance=0.1)
        th = self.rng.uniform(-np.pi, np.pi, n)
        return np.column_stack([x, y, th])

    def init_at(self, pose, sd=(0.3, 0.3, 0.2)):
        """Tracking mode: start from a known-ish pose instead of the whole map."""
        self.particles = np.asarray(pose, float) + self.rng.normal(0, sd, (self.n, 3))
        self.particles[:, 2] = wrap(self.particles[:, 2])
        self.weights[:] = 1.0 / self.n

    def predict(self, odom):
        self.particles = sample_motion(self.particles, odom, self.motion, self.rng)

    def update(self, ranges):
        log_w, lik = likelihood_field(self.grid, self.particles, ranges, self.lidar)
        log_w = log_w + np.log(self.weights + 1e-300)
        log_w -= log_w.max()
        w = np.exp(log_w)
        self.weights = w / w.sum()

        # The book averages raw particle weights here. I weight the average by the
        # posterior instead: otherwise the uniform particles we just injected drag
        # the average down and trigger more injection, and it never settles.
        fit = float(np.dot(self.weights, lik))
        self.fit = fit
        if self.w_slow is None:
            self.w_slow = self.w_fast = fit
        else:
            self.w_slow += self.alpha_slow * (fit - self.w_slow)
            self.w_fast += self.alpha_fast * (fit - self.w_fast)

        inject = min(self.max_inject, max(0.0, 1.0 - self.w_fast / self.w_slow))
        if inject > 0.01 or self.n_eff() < 0.5 * self.n:
            self.resample(inject)
        else:
            self.injected = 0

    def resample(self, inject_frac=0.0):
        m = int(round(inject_frac * self.n))
        idx = low_variance_resample(self.weights, self.n - m, self.rng)
        fresh = self.particles[idx]
        if m:
            fresh = np.vstack([fresh, self._random_particles(m)])
        self.particles = fresh
        self.weights = np.full(self.n, 1.0 / self.n)
        self.injected = m

    def n_eff(self):
        return 1.0 / np.sum(self.weights ** 2)

    def spread(self):
        """Weighted std-dev of particle positions, in metres."""
        p, w = self.particles, self.weights
        mx, my = np.dot(w, p[:, 0]), np.dot(w, p[:, 1])
        return float(np.sqrt(np.dot(w, (p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2)))

    def estimate(self, bin_size=1.0):
        # A plain weighted mean is garbage while the belief is multimodal (it lands
        # in the wall between two candidate rooms). Find the heaviest 1 m bin and
        # average only the particles around it.
        p, w = self.particles, self.weights
        bx = np.floor(p[:, 0] / bin_size).astype(np.int64)
        by = np.floor(p[:, 1] / bin_size).astype(np.int64)
        keys, inv = np.unique(by * 100_003 + bx, return_inverse=True)
        mass = np.bincount(inv, weights=w)
        sel = inv == mass.argmax()
        ws = w[sel] + 1e-15
        cx, cy = np.average(p[sel, 0], weights=ws), np.average(p[sel, 1], weights=ws)

        near = np.hypot(p[:, 0] - cx, p[:, 1] - cy) < bin_size
        wn = w[near] + 1e-15
        x = np.average(p[near, 0], weights=wn)
        y = np.average(p[near, 1], weights=wn)
        th = np.arctan2(np.dot(wn, np.sin(p[near, 2])), np.dot(wn, np.cos(p[near, 2])))
        return Estimate(float(x), float(y), float(th), float(w[near].sum()))
