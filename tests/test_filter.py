import numpy as np
import pytest

from mcl.__main__ import DEFAULT_MAP, first_sustained, main
from mcl.filter import MonteCarloLocalizer, low_variance_resample
from mcl.grid import GridMap
from mcl.models import LidarModel, MotionNoise, likelihood_field, sample_motion
from mcl.planner import plan, segment_clear
from mcl.sim import Robot


@pytest.fixture(scope="module")
def office():
    return GridMap.load(DEFAULT_MAP)


def test_low_variance_resample_is_proportional():
    w = np.array([0.5, 0.25, 0.125, 0.125])
    idx = low_variance_resample(w, 8000, np.random.default_rng(3))
    counts = np.bincount(idx, minlength=4)
    # systematic resampling is exact up to +-1 per bucket
    assert np.all(np.abs(counts - w * 8000) <= 1)


def test_resample_never_picks_zero_weight():
    w = np.array([0.0, 1.0, 0.0])
    idx = low_variance_resample(w, 100, np.random.default_rng(0))
    assert set(idx) == {1}


def test_noiseless_motion_is_exact():
    quiet = MotionNoise(0, 0, 0, 0, 0, 0)
    p = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, np.pi / 2]])
    out = sample_motion(p, (np.pi / 2, 2.0, 0.0), quiet, np.random.default_rng(0))
    assert np.allclose(out, [[1.0, 3.0, np.pi / 2], [-2.0, 0.0, -np.pi]], atol=1e-12) or \
        np.allclose(out, [[1.0, 3.0, np.pi / 2], [-2.0, 0.0, np.pi]], atol=1e-12)


def test_motion_noise_mean_is_unbiased():
    p = np.zeros((20000, 3))
    out = sample_motion(p, (0.0, 1.0, 0.0), MotionNoise(), np.random.default_rng(0))
    assert abs(out[:, 0].mean() - 1.0) < 0.01
    assert abs(out[:, 1].mean()) < 0.01


def test_sensor_model_prefers_true_pose(office):
    rng = np.random.default_rng(5)
    lidar = LidarModel()
    bot = Robot(office, lidar, rng)
    scan = bot.scan()
    wrong = bot.pose + np.array([0.6, -0.4, 0.3])
    log_w, lik = likelihood_field(office, np.vstack([bot.pose, wrong]), scan, lidar)
    assert log_w[0] > log_w[1]
    assert lik[0] > 0.5


def test_planned_path_is_collision_free(office):
    rng = np.random.default_rng(2)
    for _ in range(3):
        (sx,), (sy,) = office.sample_free(1, rng, clearance=0.5)
        (gx,), (gy,) = office.sample_free(1, rng, clearance=0.5)
        path = plan(office, (sx, sy), (gx, gy))
        assert path is not None
        assert np.hypot(path[-1][0] - gx, path[-1][1] - gy) < 0.2
        for a, b in zip(path, path[1:]):
            assert segment_clear(office, a, b, 0.2)


def test_first_sustained():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0], bool)
    assert first_sustained(m, run=5) == 4
    assert first_sustained(m, run=6) is None


def _run(office, seed, steps, kidnap_at=None, inject=0.2):
    rng = np.random.default_rng(seed)
    lidar = LidarModel()
    bot = Robot(office, lidar, rng)
    pf = MonteCarloLocalizer(office, 3000, lidar=lidar, seed=seed + 1, max_inject=inject)
    errs = []
    for k in range(steps):
        if k == kidnap_at:
            bot.kidnap()
        pf.predict(bot.step())
        pf.update(bot.scan())
        est = pf.estimate()
        errs.append(np.hypot(est.x - bot.pose[0], est.y - bot.pose[1]))
    return np.array(errs)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_global_localization_converges(office, seed):
    errs = _run(office, seed, 100)
    conv = first_sustained(errs < 0.3)
    assert conv is not None and conv < 70
    assert errs[conv:].mean() < 0.15


def test_kidnapped_robot_recovers(office):
    errs = _run(office, 4, 140, kidnap_at=80)
    assert errs[80] > 1.0  # it really did get moved
    rec = first_sustained(errs < 0.3, start=80)
    assert rec is not None and rec - 80 < 50


def test_plain_mcl_stays_lost_after_kidnap(office):
    # the whole point of the augmentation, so worth pinning down
    errs = _run(office, 4, 140, kidnap_at=80, inject=0.0)
    assert errs[-20:].mean() > 1.0


def test_cli_smoke(capsys):
    assert main(["--steps", "60", "--kidnap-at", "-1", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "global localization: converged" in out
