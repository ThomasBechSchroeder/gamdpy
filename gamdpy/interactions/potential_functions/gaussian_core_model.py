import numba
from math import exp

def gaussian_core_model(dist, params):
    r""" Gaussian-core model (GCM) [Stillinger1976]_

    .. math::

        u(r) = \varepsilon \exp(-r^2/\sigma^2)

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        [sigma, epsilon]

    References
    ----------
    .. [Stillinger1976] Frank H. Stillinger, "Phase transitions in the Gaussian core system", J. Chem. Phys. 65, 3968–3974 (1976), https://doi.org/10.1063/1.432891

    """

    # Extract parameters
    sigma = numba.float32(params[0])  # σ
    epsilon = numba.float32(params[1])  # ɛ

    # Helper variables
    one = numba.float32(1.0)  # 1
    two = numba.float32(2.0)  # 2
    four = numba.float32(4.0) # 4
    kappa = one/sigma/sigma  # κ = 1/σ²
    kappa_dist2 = kappa * dist * dist
    gcm = epsilon * exp(-kappa_dist2)  # ɛ·exp(-κ·r²)

    # u(r) = ɛ·exp(-κ·r²)
    u = gcm

    # -u'(r)/r = ɛ·exp(-κ·r²)·2κ
    s = gcm * two * kappa

    # u''(r) = ɛ·exp(-κ·r²)·(4κ²r²-2κ)
    d2u_dr2 = gcm * (four * kappa * kappa_dist2 - two * kappa)

    return u, s, d2u_dr2
