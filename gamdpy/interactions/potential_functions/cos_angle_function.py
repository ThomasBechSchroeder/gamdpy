import numpy as np 
import numba 
import math 
from numba import cuda

   


def cos_angle_function(angle, params):
    r"""Cosine angle potential,

    .. math::

        u(\theta) = \frac{k}{2} (\cos(\theta) - \cos(\theta_0))^2

    Parameters
    ----------
    theta : float
        Angle (radians).
    params : array-like
        [theta_0, k] Angle of minimum energy (zero force)) and spring constant.

    See Also
    --------

    gamdpy.harmonic_angle_function

    """
    
    angle0, kspring = params[0], params[1]

    cos_angle_0 = math.cos(angle0)
    cos_angle = math.cos(angle) 
    dcos_angle = cos_angle - cos_angle_0

    f = -kspring*dcos_angle
    u = numba.float32(0.5)*kspring*dcos_angle**2

    return  u, f
