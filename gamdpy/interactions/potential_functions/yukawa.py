import numba
from math import exp

def yukawa(dist, params):
    r""" The Yukawa potential (simple screened Coulomb potential)

    .. math::

        u(r) = \varepsilon \frac{\sigma}{r} \exp(-r/\sigma)

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        σ, ε

    """

    # Extract parameters
    sigma = numba.float32(params[0])  # σ
    epsilon = numba.float32(params[1])  # ɛ

    # Helper variables
    one = numba.float32(1.0)  # 1
    two = numba.float32(2.0)  # 2
    prefactor = epsilon*sigma  # A = ɛ·σ
    kappa = one/sigma  # κ = 1/σ
    kappa_dist = kappa * dist  # κ·r
    inv_dist = one / dist  # 1/r
    inv_dist3 = inv_dist*inv_dist*inv_dist  # 1/r³
    exp_kappa_dist = prefactor * exp(-kappa_dist)  # A·exp(-κ·r)

    # u(r) = A·exp(-κ·r)/r
    u = exp_kappa_dist * inv_dist

    # -u'(r)/r = A·exp(-κ·r)·(κ·r + 1)/r³
    s = exp_kappa_dist * (kappa_dist + one) * inv_dist3

    # u''(r) = A·exp(-κ·r)*([κ·r]² + 2κ·r + 2)/r³
    d2u_dr2 = exp_kappa_dist * (kappa_dist*kappa_dist + two * kappa_dist + two) * inv_dist3

    return u, s, d2u_dr2
