#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 08:55:28 2025

@author: nbailey
"""

import numpy as np
import numba
import math
from numba import cuda




def make_harmonic_angle_function(SMALL=1.e-6):
    """
    Create a version of harmonic_angle_function with a specified value of the regularization
    parameter SMALL. This prevents overflow when dividing by small values of sin(theta).

    Parameters
    ----------

    SMALL : float
        Value to be added to sin(theta) before dividing by the latter to get the gradient

    Returns
    -------

    angle potential function based on a harmonic potential with a specified SMALL parameter to prevent overflow
    """
    def harmonic_angle_function(theta: float, params: np.ndarray) -> tuple:
        theta_0 = params[0]
        kspring = params[1]
        s = math.sin(theta)
        u = numba.float32(0.5) * kspring * (theta - theta_0) ** 2
        d_u_d_cos_theta_neg = kspring * (theta - theta_0) /  (s+SMALL)
        return u, d_u_d_cos_theta_neg
    return harmonic_angle_function


harmonic_angle_function = make_harmonic_angle_function(1.e-6)

harmonic_angle_function.__doc__ = r"""

    Harmonic angle potential,

    With a regularization parameter SMALL = 1.e-6 hard-coded in.
    To change the value of SMALL call make_harmonic_angle_function(SMALL)
    .. math::

        u(\theta) = \frac{k}{2} (\theta - \theta_0)^2

    Parameters
    ----------

    theta : float
        Angle (radians) defined by three neighboring particles in a molecule, more precisely: the angle subtended by atoms 0 and 2 at 1

    params : array-like
         :math:`\theta_0`, the angle of minimum energy, :math:`k_{spring}`, the spring constant.

    Returns
    -------

    u : float
        Potential energy
    d_u_cos_theta_neg: float
        Negative derivative of the potential energy with respect to cos(theta). This involves dividing by sin(theta)+SMALL
    See Also
    --------

    gamdpy.Angles

    """


