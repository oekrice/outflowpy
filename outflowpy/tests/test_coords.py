import numpy as np

import outflowpy.coords


def test_forward_backwards():
    x, y, z = 1, 0, 0
    rho, s, phi = outflowpy.coords.cart2strum(x, y, z)
    x1, y1, z1 = outflowpy.coords.strum2cart(rho, s, phi)

    assert np.allclose(np.array([x, y, z]), np.array([x1, y1, z1]))
