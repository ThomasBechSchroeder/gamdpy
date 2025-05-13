import numba

def LJ_12_6(dist, params):  
    """ The 12-6 Lennard-Jones potential

    See :func:`gamdpy.apply_shifted_potential_cutoff` for a usage example.

    .. math::

        u(r) = A_{12} r^{-12} + A_6 r^{-6}
        
        u'(r) = -12*A_{12} r^{-13} - 6*A_6 r^{-7}

    Parameters
    ----------

    dist : float
        Distance between particles

    params : array-like
        A₁₂, A₆

    Returns
    -------

    u : float
        Potential energy
    s : float
        Force multiplier, -u'(r)/r
    umm : float
        Second derivative of potential energy

    """
    A12 = params[0]  #     Um(r) =    -12*A12*r**-13 -   6*A6*r**-7
    A6 = params[1]  #     Umm(r) = 13*12*A12*r**-14 + 7*6*A6*r**-8
    invDist = numba.float32(1.0) / dist  # s = -Um/r =     12*A12*r**-14 +   6*A6*r**-8, Fx = s*dx

    u = A12 * invDist ** 12 + A6 * invDist ** 6
    s = numba.float32(12.0) * A12 * invDist ** 14 + numba.float32(6.0) * A6 * invDist ** 8
    umm = numba.float32(156.0) * A12 * invDist ** 14 + numba.float32(42.0) * A6 * invDist ** 8
    return u, s, umm  # U(r), s == -U'(r)/r, U''(r)

