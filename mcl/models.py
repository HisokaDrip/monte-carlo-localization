"""Motion and sensor models (Probabilistic Robotics ch. 5 & 6, more or less)."""
from dataclasses import dataclass

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class MotionNoise:
    # std-dev form rather than the book's variance form; easier to reason about
    rot_per_rot: float = 0.08     # rad of heading error per rad turned
    rot_per_m: float = 0.03       # rad of heading error per metre driven
    trans_per_m: float = 0.08     # metres of slip per metre driven
    trans_per_rot: float = 0.02   # metres of slip per rad turned
    floor_rot: float = 0.01       # keep some spread even when standing still,
    floor_trans: float = 0.01     # otherwise the cloud collapses to a point


def sample_motion(particles, odom, noise, rng):
    """Odometry motion model: odom = (rot1, trans, rot2)."""
    rot1, trans, rot2 = odom
    n = len(particles)
    sd_r1 = noise.floor_rot + noise.rot_per_rot * abs(rot1) + noise.rot_per_m * trans
    sd_tr = noise.floor_trans + noise.trans_per_m * trans + noise.trans_per_rot * (abs(rot1) + abs(rot2))
    sd_r2 = noise.floor_rot + noise.rot_per_rot * abs(rot2) + noise.rot_per_m * trans

    r1 = rot1 + rng.normal(0.0, sd_r1, n)
    tr = trans + rng.normal(0.0, sd_tr, n)
    r2 = rot2 + rng.normal(0.0, sd_r2, n)

    x, y, th = particles[:, 0], particles[:, 1], particles[:, 2]
    out = np.empty_like(particles)
    out[:, 0] = x + tr * np.cos(th + r1)
    out[:, 1] = y + tr * np.sin(th + r1)
    out[:, 2] = wrap(th + r1 + r2)
    return out


@dataclass
class LidarModel:
    n_beams: int = 30
    fov: float = 2 * np.pi
    max_range: float = 8.0
    sigma_hit: float = 0.2
    z_hit: float = 0.9
    z_rand: float = 0.1
    # Beams from one scan are nowhere near independent (they all share the same
    # pose error), so multiplying 30 likelihoods makes the filter wildly
    # overconfident. Raising the product to a power < 1 is the usual hack.
    temper: float = 0.3
    off_map_dist: float = 2.0

    def angles(self):
        if self.fov >= 2 * np.pi - 1e-9:
            return np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False)
        return np.linspace(-self.fov / 2, self.fov / 2, self.n_beams)


def likelihood_field(grid, particles, ranges, model):
    """Score each particle against a scan.

    Returns (log_weight, per_beam_likelihood). The second one is the geometric
    mean beam likelihood normalised to (0, 1]; it doesn't depend on how many
    beams came back, which makes it a decent "how lost am I" signal.
    """
    n = len(particles)
    angles = model.angles()
    valid = ranges < model.max_range - 1e-6  # max-range returns carry no endpoint
    if not valid.any():
        return np.zeros(n), np.ones(n)
    z, a = ranges[valid], angles[valid]

    x, y, th = particles[:, 0], particles[:, 1], particles[:, 2]
    ex = x[:, None] + z[None, :] * np.cos(th[:, None] + a[None, :])
    ey = y[:, None] + z[None, :] * np.sin(th[:, None] + a[None, :])
    d = grid.lookup_dist(ex, ey, outside=model.off_map_dist)

    p = model.z_hit * np.exp(-0.5 * (d / model.sigma_hit) ** 2) + model.z_rand / model.max_range
    p_max = model.z_hit + model.z_rand / model.max_range
    logp = np.log(p / p_max)

    log_w = model.temper * logp.sum(axis=1)
    lik = np.exp(logp.mean(axis=1))

    # a particle sitting inside a wall is wrong no matter what the scan says
    in_wall = ~grid.is_free(x, y)
    log_w[in_wall] = -1e9
    lik[in_wall] = 0.0
    return log_w, lik
