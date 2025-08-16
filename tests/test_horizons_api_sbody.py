import sys

import numpy as np
import numpy.testing as npt
import pytest

from sbdynt import horizons_api

sys.path.insert(0, "../src")


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            ["248835", 2455000],
            [
                1,
                [9.35161838],
                [-4.27779184],
                [6.21377256],
                [0.4208934],
                [2.07323381],
                [0.56858258],
            ],
            None,
        ),
    ],
)
def test_query_horizons_sb(test, expected, expect_raises):
    xe = np.zeros(1)
    ye = np.zeros(1)
    ze = np.zeros(1)
    vxe = np.zeros(1)
    vye = np.zeros(1)
    vze = np.zeros(1)
    x = np.zeros(1)
    y = np.zeros(1)
    z = np.zeros(1)
    vx = np.zeros(1)
    vy = np.zeros(1)
    vz = np.zeros(1)
    flag, xe, ye, ze, vxe, vye, vze = expected
    sbody, epoch = test

    ft, x, y, z, vx, vy, vz = horizons_api.query_sb_from_horizons(
        des=sbody, epoch=epoch
    )
    npt.assert_equal(int(ft), int(flag))
    npt.assert_almost_equal(x, xe, decimal=2)
    npt.assert_almost_equal(y, ye, decimal=2)
    npt.assert_almost_equal(z, ze, decimal=2)
    npt.assert_almost_equal(vx, vxe, decimal=2)
    npt.assert_almost_equal(vy, vye, decimal=2)
    npt.assert_almost_equal(vz, vze, decimal=2)

    return
