import numpy as np

from .models import wrap
from .planner import plan


class Robot:
    """Ground truth robot: drives between random goals and fakes a noisy lidar.

    The controller steers on the *true* pose. That's cheating as far as a real
    robot goes, but it keeps the simulation from driving into walls while the
    filter is still confused. The filter itself only ever sees odometry + scans.
    """

    def __init__(self, grid, lidar, rng, step_len=0.3, max_turn=0.6,
                 range_noise=0.03, outlier_rate=0.03):
        self.grid = grid
        self.lidar = lidar
        self.rng = rng
        self.step_len = step_len
        self.max_turn = max_turn
        self.range_noise = range_noise
        self.outlier_rate = outlier_rate
        self.waypoints = []
        self.pose = self._random_pose()

    def _random_pose(self):
        x, y = self.grid.sample_free(1, self.rng, clearance=0.6)
        return np.array([x[0], y[0], self.rng.uniform(-np.pi, np.pi)])

    def kidnap(self):
        self.pose = self._random_pose()
        self.waypoints = []

    def scan(self):
        x, y, th = self.pose
        angles = self.lidar.angles() + th
        clean = self.grid.raycast(x, y, angles, self.lidar.max_range)
        no_hit = clean >= self.lidar.max_range
        r = clean + self.rng.normal(0.0, self.range_noise, clean.shape)
        # people walking past, chair legs, etc: random short returns
        junk = self.rng.random(clean.shape) < self.outlier_rate
        r[junk] = self.rng.uniform(0.1, np.maximum(clean[junk], 0.2))
        r = np.clip(r, 0.05, self.lidar.max_range - 1e-3)
        r[no_hit & ~junk] = self.lidar.max_range
        return r

    def _next_waypoint(self):
        while not self.waypoints:
            gx, gy = self.grid.sample_free(1, self.rng, clearance=0.5)
            goal = (gx[0], gy[0])
            for clearance in (0.35, 0.2):
                path = plan(self.grid, self.pose[:2], goal, clearance)
                if path:
                    self.waypoints = path[1:]
                    break
        return self.waypoints[0]

    def step(self):
        """Move one control tick. Returns the odometry reading (rot1, trans, rot2)."""
        wx, wy = self._next_waypoint()
        x, y, th = self.pose
        dist = np.hypot(wx - x, wy - y)
        if dist < 0.2:
            self.waypoints.pop(0)
            if not self.waypoints:
                return self.step()
            wx, wy = self.waypoints[0]
            dist = np.hypot(wx - x, wy - y)

        err = wrap(np.arctan2(wy - y, wx - x) - th)
        rot1 = float(np.clip(err, -self.max_turn, self.max_turn))
        trans = float(min(self.step_len, dist)) if abs(err) <= self.max_turn else 0.0

        # what actually happens vs what the wheel encoders think happened
        true_r1 = rot1 + self.rng.normal(0, 0.01 + 0.03 * abs(rot1))
        true_tr = trans * (1 + self.rng.normal(0, 0.03))
        nth = wrap(th + true_r1)
        nx, ny = x + true_tr * np.cos(nth), y + true_tr * np.sin(nth)
        if self.grid.lookup_dist(nx, ny)[0] > 0.15:
            self.pose = np.array([nx, ny, nth])
        else:
            self.pose = np.array([x, y, nth])  # bumper hit, wheels spin
        return rot1, trans, 0.0
