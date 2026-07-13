import numba

def apply_shifted_force_cutoff(pair_potential):  
    # Cut-off by computing potential twice, avoiding changes to params
    r""" Apply shifted force cutoff to a pair-potential function

    If the input pair potential is :math:`u(r)`, then the shifted force potential is

    .. math::
        u_m(r) = u(r) - u(r_{c}) + s(r_{c})(r - r_{c}) \quad (r<r_c)

    and :math:`u_m(r)=0` for :math:`r>r_c`,
    where :math:`r_c` is the cutoff distance, and  :math:`s(r) = -u'(r)/r`.
    The last entry the parameter array (`params`) is the cutoff: `[..., r_c]`.

    Parameters
    ----------
        pair_potential: callable
            a function that calculates a pair-potential:
            u, s, umm =  pair_potential(dist, params)

    Returns
    -------

        potential: callable
            a function where shifted force cutoff is applied to original function

    """
    pair_pot = numba.njit(pair_potential)

    # @numba.njit # Jit moved to 'last minut', i.e. PairPotential.get_kernel 
    def potential(dist, params): # pragma: no cover
        cut = params[-1]
        if dist>cut:
            return numba.float32(0.0), numba.float32(0.0), numba.float32(0.0)

        u, s, umm = pair_pot(dist, params)
        u_cut, s_cut, umm_cut = pair_pot(cut, params)
        u -= u_cut - s_cut * cut * (dist - cut)
        s -= s_cut * cut/dist
        #u -= u_cut - s_cut*dist*(dist-cut)
        #s -= s_cut
        return u, s, umm

    return potential

