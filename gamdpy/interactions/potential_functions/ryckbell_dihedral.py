import numpy as np 
import numba 
import math 
from numba import cuda


def ryckbell_dihedral(dihedral, params):
    """ Ryckert-Bellemans potential

    .. math::

        u(\\phi) = \\sum_{n=0}^5 p_n \\cos^n(phi) 

    Parameters
    ----------
    dihedral: Current dihedral
    params: Parameter array 
    
    Returns
    -------
    u: Potential energy
    f: Force multiplier
    """
    
    cos_dihedral = math.cos(math.pi - dihedral)
   
    u = f = numba.float32(0.0)
    for n in range(6):
        u = u + params[n]*cos_dihedral**n
        if n > 0:
            f = f - n*params[n]*cos_dihedral**(n-1) 

    return  u, f


