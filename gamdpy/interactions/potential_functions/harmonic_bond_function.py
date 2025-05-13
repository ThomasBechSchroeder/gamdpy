import numpy as np
import numba
import math
from numba import cuda

def harmonic_bond_function(dist: float, params: np.ndarray) -> tuple:
    """ Harmonic bond potential

    .. math::

        u(r) = \\frac{1}{2} k (r - r_0)^2

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        r₀, k

    Returns
    -------

    u : float
        Potential energy
    s : float
        Force multiplier, -u'(r)/r
    umm : float
        Second derivative of potential energy

    See Also
    --------

    gamdpy.Bonds

    """
    length = params[0]
    strength = params[1]

    u = numba.float32(0.5) * strength * (dist - length) ** 2
    s = -strength * (dist - length) / dist
    umm = strength
    return u, s, umm  # U(r), s == -U'(r)/r, U''(r)

