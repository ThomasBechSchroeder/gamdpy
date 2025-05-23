import numba
import numpy as np

def add_potential_functions(potential_1, potential_2):  
    """ Add two potential functions into a single potential function

    Note that the two potential functions will have the same cut-off, by convention always stored as the last entry in params.
    The **potential_1** cannot explicitly depend on cut-off (last entry in params).
    The **potential_2** should know where to *look* for its first parameter in the list of parmeters (**params**),
    see e.g. **first_parameter** in :func:`gamdpy.make_IPL_n`.


    Parameters
    ----------
        potential_1: callable
            a function that calculates a pair-potential:
            u, s, umm = potential_1(dist, params)
        potential_2: callable
            a function that calculates a pair-potential:
            u, s, umm = potential_2(dist, params)

    Returns
    -------
        potential: callable
            a function implementing the sum of potential_1 and potential_2

    Example
    -------
    Below we make the 12-6 Lennard-Jones potential by adding two inverse power-law potentials.

    >>> import gamdpy as gp
    >>> LJ = gp.add_potential_functions(gp.make_IPL_n(12), gp.make_IPL_n(6, first_parameter=1))
    >>> params = A12, A6, cut = 4.0, -4.0, 2.5
    >>> dist = 2**(1/6) # Minima of LJ potential
    >>> LJ(dist,params)[0] # Pair energy in minima
    -1.0
    """
    pair_pot1 = numba.njit(potential_1)
    pair_pot2 = numba.njit(potential_2)

    #@numba.njit
    def potential(dist, params): # pragma: no cover
        u1, s1, umm1 = pair_pot1(dist, params)
        u2, s2, umm2 = pair_pot2(dist, params)
        return u1+u2, s1+s2, umm1+umm2

    return potential

