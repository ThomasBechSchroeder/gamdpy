import numpy as np
import numba
import math
from numba import cuda

def gaussian_screened_coulomb(dist, params):
    """ Point-like charges screened by gaussian distributions centered on the particles.

    .. math::

        u(r) = Q erfc(\\kappa r)/r

    Parameters
    ----------

    dist : float
        Distance between particles
        
    params : array-like
        Q (product of charges), kappa (screening decay rate)
    """
    Q = params[0]
    kappa = params[1]

    # Helper variables
    two = numba.float32(2.0)
    inv_dist = numba.float32(1.0) / dist
    kappa_dist = kappa * dist
    erfc_kappa_dist = math.erfc(kappa_dist)
    exp_kappa2_dist2 = math.exp(-kappa_dist * kappa_dist)
    inv_dist2 = inv_dist * inv_dist
    inv_dist3 = inv_dist2 * inv_dist
    prefactor = two * kappa / math.sqrt(math.pi)

    # u(r)
    u = Q * erfc_kappa_dist * inv_dist

    # -u(r)/r
    s = Q * (erfc_kappa_dist * inv_dist3 + prefactor * exp_kappa2_dist2 * inv_dist2)

    # u''(r)
    upp = two * Q * (erfc_kappa_dist * inv_dist3 +
                     prefactor * kappa * kappa * exp_kappa2_dist2 +
                     prefactor * exp_kappa2_dist2 * inv_dist2)
    
    return u, s, upp

