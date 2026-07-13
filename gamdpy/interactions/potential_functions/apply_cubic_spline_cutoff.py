import numba

def apply_cubic_spline_cutoff(pair_potential):
    r""" Apply cubic spline cutoff to a pair-potential function

    Actually the cubic spline is applied to the pair force as done by LAMMPS
    (see https://docs.lammps.org/pair_lj_smooth.html).
    The potential energy is given by a quartic function in the transition region.
    Specifically, if :math:`u(r)` is the original potential, then the (modified) truncated potential is

    .. math::
       u_m(r) = u(r) + K \quad (r < r_i),

    .. math::
       u_m(r) = C_0 + C_1 [r-r_i] + \frac{C_2}{2} [r-r_i]^2 + \frac{C_3}{3} [r-r_i]^3 + \frac{C_4}{4} [r-r_i]^4 \quad (r_i \le r \le r_c),

    and

    .. math::
       u_m(r) = 0 \quad (r > r_c),

    The parameters (the :math:`C`'s and the :math:`K`) for a smooth truncation are computed automatically:
    At the inner cutoff radius, the force and its first derivative match the unsmoothed potential (:math:`u(r)`).
    At the outer cutoff radius, both the force and the first derivative are zero.
    See SM of Ref. [Leoni2025]_ for more details.

    The last two values of the parameter array (`params`) are the inner and outer cutoffs: `[..., r_i, r_c]`.

    Parameters
    ----------
        pair_potential: callable
            a function that calculates a pair-potential:
            u, s, umm =  pair_potential(dist, params)

    Returns
    -------

        potential: callable
            a function where a cubic spline cutoff is applied to original function

    References
    ----------

    .. [Leoni2025] Fabio Leoni, John Russo, Francesco Sciortino, and Taiki Yanagishima.
       "Generating Ultrastable Glasses by Homogenizing the Local Virial Stress".
       Phys. Rev. Lett. 134, 128201 (2025)
       https://doi.org/10.1103/PhysRevLett.134.128201

    """

    # Cut-off by computing potential twice, avoiding changes to params
    # Note: calls original potential function  twice each time, avoiding changes to params.
    # The four coefficients of the spline are also computed each time, along with the two energy
    # shift constants, which results in a small but noticeable performance cost.

    pair_pot = numba.njit(pair_potential)

    # @numba.njit # Jit moved to 'last minut', i.e. PairPotential.get_kernel 
    def potential(dist, params): # pragma: no cover
        cut_outer = params[-1]
        if dist>cut_outer:
            return numba.float32(0.0), numba.float32(0.0), numba.float32(0.0)

        cut_inner = params[-2]
        Delta_r = cut_outer - cut_inner

        u_bare, s_bare, umm_bare = pair_pot(dist, params)
        u_cut_inner, s_cut_inner, umm_cut_inner = pair_pot(cut_inner, params)

        two = numba.float32(2.0)
        three = numba.float32(3.0)
        four = numba.float32(4.0)

        C1, C2, C3, C4 = (s_cut_inner*cut_inner,
                          -umm_cut_inner,
                             (-three*s_cut_inner*cut_inner + two*umm_cut_inner * Delta_r) / Delta_r**2,
                             (two*s_cut_inner*cut_inner - umm_cut_inner*Delta_r)/Delta_r**3)
        # for the potential, we integrate, and include a constant C0 whose value is such that the potential is zero at the outer cutoff
        C0 = - (C1*Delta_r + C2*Delta_r**2/two + C3*Delta_r**3/three + C4*Delta_r**4/four)
        # And now we have to shift the LJ potential inside the inner cutoff to match the potential at that point
        K = -C0 - u_cut_inner


        if dist < cut_inner:
            s = s_bare
            u = u_bare + K
            umm = umm_bare
        else:
            delta_r = dist - cut_inner
            u = - (C0 + C1*delta_r + C2*delta_r**2/two + C3*delta_r**3/three + C4*delta_r**4/four)
            s = (C1 + C2*delta_r + C3*delta_r**2 + C4*delta_r**3)/dist
            umm = - (C2 + two*C3*delta_r + three*C4*delta_r**2)


        return u, s, umm

    return potential

