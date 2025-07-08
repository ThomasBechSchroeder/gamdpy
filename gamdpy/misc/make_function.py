import numpy as np
import numba
from numba import cuda
import math

def make_function_constant(value: float) -> callable:
    r""" Return a function that returns a constant value.

    .. math::
        f(x) = y_0,

    Parameters
    ----------
    value : float
        The value :math:`y_0`

    Returns
    -------
    callable
        A function that can be compiled to the device.

    """
    value = np.float32(value)

    def function(x):
        return value

    return function


def make_function_ramp(value0: float, x0: float, value1: float, x1: float) -> callable:
    r""" Create a piecewise‐linear “ramp” function.

    .. math::
        f(x) =
        \begin{cases}
          y_0, & x < x_0,\\
          y_0 + \dfrac{y_1 - y_0}{x_1 - x_0}\,(x - x_0), & x_0 \le x \le x_1,\\
          y_1, & x > x_1,
        \end{cases}

    Parameters
    ----------
    value0 : float
        The value :math:`y_0` for :math:`x < x_0`.
    x0 : float
        The start point :math:`x_0` of the linear region.
    value1 : float
        The value :math:`y_1` for :math:`x > x_1`.
    x1 : float
        The end point :math:`x_1` of the linear region.

    Returns
    -------
    callable
        A function implementing the above piecewise behavior.

    """
    value0, x0, value1, x1 = np.float32(value0), np.float32(x0), np.float32(value1), np.float32(x1)
    alpha = (value1 - value0) / (x1 - x0)

    def function(x):
        if x < x0:
            return value0
        if x < x1:
            return value0 + (x - x0) * alpha
        return value1

    return function


def make_function_sin(period: float, amplitude: float, offset: float) -> callable:
    r""" Return a function that returns a sin function,

    .. math::
        f(x) = y_0 + A \sin(2 \pi x / T),

    with given period (:math:`T`), amplitude (:math:`A`) and offset (:math:`y_0`)

    Parameters
    ----------
    T : float
        The period, :math:`T`.
    amplitude : float
        The amplitude, :math:`A`.
    offset : float
        The offset :math:`y_0`.

    Returns
    -------
    callable
        A function that can be compiled to the device.

    """

    from math import sin, pi
    period, amplitude, offset = np.float32(period), np.float32(amplitude), np.float32(offset)

    def function(x):
        return offset + amplitude * sin(2 * pi * x / period)

    return function


