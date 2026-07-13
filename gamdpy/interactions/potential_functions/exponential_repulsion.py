import numba
from math import exp

def exponential_repulsion(dist, params):
    r""" The exponential repulsion pair potential (EXP)

    .. math::

        u(r) = \varepsilon \exp(-r/\sigma)

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
    kappa = one/sigma  # κ = 1/σ
    kappa_dist = kappa * dist  # κ·r
    inv_dist = one / dist  # 1/r
    exp_kappa_dist = epsilon * exp(-kappa_dist)  # ɛ·exp(-κ·r)

    # u(r) = ɛ·exp(-κ·r)
    u = exp_kappa_dist

    # -u'(r)/r = ɛ·exp(-κ·r)·κ/r
    s = exp_kappa_dist * kappa * inv_dist

    # u''(r) = ɛ·exp(-κ·r)·κ²
    d2u_dr2 = exp_kappa_dist * kappa * kappa

    return u, s, d2u_dr2
