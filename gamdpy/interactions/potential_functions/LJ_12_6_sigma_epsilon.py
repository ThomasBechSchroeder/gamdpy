import numpy as np
import numba
import math
from numba import cuda

def LJ_12_6_sigma_epsilon(dist, params):
    """ The 12-6 Lennard-Jones potential
    
    .. math::
    
        u(r) = 4\\epsilon(   (r/\\sigma)^{-12} -   (r/\\sigma)^{-6} )
    
    This is the same as the :func:`gamdpy.LJ_12_6` potential, 
    but with :math:`\\sigma` (sigma) and :math:`\\epsilon` (epsilon) as parameters.
    
    Parameters
    ----------
    
    dist : float
        Distance between particles
        
    params : array-like
        σ, ε

    """  # LJ:  U(r)  =     4*epsilon(   (r/sigma)**-12 +   (r/sigma)**-6 )
    sigma = params[0]  #      Um(r) =   -24*epsilon( 2*(r/sigma)**-13 +   (r/sigma)**-7 )/sigma
    epsilon = params[1]  #      Umm(r) =   24*epsilon(26*(r/sigma)**-14 + 7*(r/sigma)**-8 )/sigma**2
    OneOdist = numba.float32(
        1.0) / dist  # s = -Um/r =     24*epsilon( 2*(r/sigma)**-14 +   (r/sigma)**-8 )/sigma**2,  Fx = s*dx
    sigmaOdist = sigma * OneOdist

    u = numba.float32(4.0) * epsilon * (sigmaOdist ** 12 - sigmaOdist ** 6)
    s = numba.float32(24.0) * epsilon * (numba.float32(2.0) * sigmaOdist ** 12 - sigmaOdist ** 6) * OneOdist ** 2
    umm = numba.float32(24.0) * epsilon * (
                numba.float32(26.0) * sigmaOdist ** 12 - numba.float32(7.0) * sigmaOdist ** 6) * OneOdist ** 2
    return u, s, umm  # U(r), s == -U'(r)/r, U''(r)

