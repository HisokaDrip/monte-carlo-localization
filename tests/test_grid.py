import numpy as np

from mcl.grid import GridMap, distance_transform

BOX = """
##########
#        #
#        #
#        #
##########
"""


def brute_edt(occ):
    rs, cs = np.nonzero(occ)
    out = np.zeros(occ.shape)
    for r in range(occ.shape[0]):
        for c in range(occ.shape[1]):
            out[r, c] = np.sqrt(((rs - r) ** 2 + (cs - c) ** 2).min())
    return out


def test_edt_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        occ = rng.random((23, 31)) < 0.05
        occ[0, 0] = True
        assert np.allclose(distance_transform(occ), brute_edt(occ))


def test_edt_single_obstacle():
    occ = np.zeros((7, 7), bool)
    occ[3, 3] = True
    d = distance_transform(occ)
    assert d[3, 3] == 0
    assert np.isclose(d[0, 0], np.sqrt(18))


def test_raycast_hits_box_walls():
    g = GridMap.from_ascii(BOX, cell_size=0.5, upsample=5)
    # interior spans x in [0.5, 4.5], y in [0.5, 2.0]
    r = g.raycast(2.0, 1.25, np.array([0.0, np.pi / 2, np.pi, -np.pi / 2]), max_range=10)
    assert np.allclose(r, [2.5, 0.75, 1.5, 0.75], atol=0.06)


def test_raycast_caps_at_max_range():
    g = GridMap.from_ascii(BOX)
    assert g.raycast(1.0, 1.25, np.array([0.0]), max_range=1.0)[0] == 1.0


def test_free_space_queries():
    g = GridMap.from_ascii(BOX)
    assert g.is_free(2.0, 1.0)[0]
    assert not g.is_free(0.2, 1.0)[0]
    assert not g.is_free(-5, 1.0)[0]
    xs, ys = g.sample_free(500, np.random.default_rng(1), clearance=0.2)
    assert g.is_free(xs, ys).all()
