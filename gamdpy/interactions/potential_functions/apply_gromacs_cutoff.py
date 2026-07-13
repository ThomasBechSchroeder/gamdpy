import numba

def apply_gromacs_cutoff(pair_potential: callable) -> callable:
    r""" Apply a smooth cutoff to a pair-potential function using GROMACS style.

    The potential is smoothly truncated at :math:`r_c` by applying a switching function.
    Specifically, if :math:`u(r)` is the original potential, then the (modified) truncated potential is

    .. math::
       u_m(r) = u(r) + S(r)

    Outside the cut-off, the switching function is :math:`S(r) = -u(r)` (:math:`r > r_c`)
    truncating the potential at :math:`r_c` :math:`u_m(r) = 0 \quad (r > r_c)`.
    In the switching region

    .. math::
       S(r) = \frac{A}{3}[r-r_1]^3 + \frac{B}{4}[r-r_1]^4 + C \quad (r_1 < r < r_c)

    At :math:`r<r_1`, the potential is shifted by a constant

    .. math::
       S(r) = C \quad (r < r_1)

    The parameters ensuring smooth truncation are

    .. math::
       A = \frac{- 3 u'(r_c) + [r_c - r_1] u''(r_c) }{[ r_c - r_1 ]^2}

    .. math::
       B = [  2 u'(r_c) - [r_c - r_1] u''(r_c) ]/[ r_c - r_1 ]^3

    .. math::
       C = - u(r_c) + \frac{1}{2} [r_c - r_1] u'(r_c) - \frac{1}{12} [r_c - r_1]^2 u''(r_c)

    The parameters are computed automatically.
    The last two values of the parameter array (`params`) are the inner and outer cutoffs: `[..., r_1, r_c]`.

    For more, see GROMACS force-switch function (3rd degree polynomial) in manual at https://www.gromacs.org/,
    or the LAMMPS documentation, https://docs.lammps.org/pair_gromacs.html.

    Parameters
    ----------

        pair_potential: callable
            a function that calculates a pair-potential:
            u, s, umm = pair_potential(dist, params)

    Returns
    -------

        potential: callable
            a function where a smooth cutoff is applied to the original function.

    Example
    -------

    >>> import gamdpy as gp
    >>> params = sig, eps, cut_inner, cut_outer = 1.0, 1.0, 3.0, 4.0
    >>> pair_func = gp.apply_gromacs_cutoff(gp.LJ_12_6)

    """
    pair_pot = numba.njit(pair_potential)

    #@numba.njit # Jit moved to 'last minut', i.e. PairPotential.get_kernel 
    def potential(dist, params): # pragma: no cover
        flt = numba.float32
        r_c = flt(params[-1])
        r_1 = flt(params[-2])

        dr = dist - r_1  # r - r_1
        dr2 = dr*dr

        Dr = r_c - r_1  # r_c - r_1
        Dr2 = Dr*Dr

        one_twelve = flt(flt(1.0)/flt(12.0))

        # u    s=-[du/dr]/r     umm=[d2u/dr2]
        u_bare, s_bare, umm_bare = pair_pot(dist, params)
        u_c, s_c, umm_c = pair_pot(r_c, params)
        um_c = -s_c*r_c

        C = - u_c + flt(0.5) * Dr * um_c - one_twelve * Dr2 * umm_c

        if dist <= r_1:  # inside the inner cutoff (r<r_1)
            u = u_bare + C
            s = s_bare
            umm = umm_bare
        elif dist < r_c:  # in the switching region (r_1<r<r_c)
            dr3 = dr2 * dr
            dr4 = dr2 * dr2
            one_third = flt(flt(1.0) / flt(3.0))

            Dr3 = Dr2 * Dr

            A = - flt(3.0) * um_c + Dr * umm_c
            A /= Dr2
            B = flt(2.0) * um_c - Dr * umm_c
            B /= Dr3

            S = one_third * A * dr3 + flt(0.25)*B*dr4 + C
            u = u_bare + S
            S_m = A*dr2 + B*dr3  # derivative of S with respect to r
            s = s_bare - S_m/dist
            S_mm = 2*A*dr + 3*B*dr2
            umm = umm_bare + S_mm

        else:  # outside the cutoff (r>r_c)
            u, s, umm = flt(0.0), flt(0.0), flt(0.0)

        return u, s, umm

    return potential
