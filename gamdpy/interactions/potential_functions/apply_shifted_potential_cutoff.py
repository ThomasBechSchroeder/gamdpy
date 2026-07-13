import numpy as np
import numba
import math
from numba import cuda

def apply_shifted_potential_cutoff(pair_potential: callable) -> callable:
    r""" Apply shifted potential cutoff to a pair-potential function

        If the input pair potential is :math:`u(r)`,
        then the shifted potential is

        .. math::

           u_m(r) = u(r) - u(r_{c}) \quad (r<r_c)

        and :math:`u_m(r)=0` for :math:`r>r_c` where :math:`r_c` is the cutoff distance.
        Calls the original potential function twice, avoiding changes to params.
        The last entry the parameter array (`params`) is the cutoff: `[..., r_c]`.

        Parameters
        ----------

        pair_potential : callable
            A function that calculates a pair-potential:
            `u, s, umm =  pair_potential(dist, params)`

        Returns
        -------

        pair_potential : callable
            A function where shifted potentia cutoff is applied to the original function.

        Example
        -------

        Example demonstrating how to set up the Lennard-Jones 12-6 potential, :func:`~gamdpy.LJ_12_6`,
        truncated and shifted to zero at the cutoff distance of 2.5.

        >>> import gamdpy as gp
        >>> pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6)
        >>> A12, A6, cut = 1.0, 1.0, 2.5
        >>> pair_pot = gp.PairPotential(pair_func, params=[A12, A6, cut], max_num_nbs=1000)
        >>> interactions = [pair_pot, ]  # List of interactions passed to an instance of the Simulation class

    """
    pair_pot = numba.njit(pair_potential)

    #@numba.njit # Jit moved to 'last minut', i.e. PairPotential.get_kernel 
    def potential(dist, params): # pragma: no cover
        cut = params[-1]
        if dist>cut:
            return numba.float32(0.0), numba.float32(0.0), numba.float32(0.0)
        
        u, s, umm = pair_pot(dist, params)
        u_cut, s_cut, umm_cut = pair_pot(cut, params)
        u -= u_cut
        return u, s, umm

    return potential

