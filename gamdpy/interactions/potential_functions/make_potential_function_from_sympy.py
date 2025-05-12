import numpy as np
import numba
import math
from numba import cuda

def make_potential_function_from_sympy(ufunc, param_names) -> callable:
    """ Make a potential function from a sympy expression

    Known to result in slow code. Use for testing and prototyping.

    Parameters
    ----------

    ufunc : sympy expression
        The potential energy expression in Sympy's symbolic form
        It has to be a radial function and the pair distance symbol shuould be r

    param_names : list
        List of parameters

    Returns
    -------

    potential_function : callable
        A function that calculates the potential energy, force multiplier, and second derivative of the potential energy
        u, s, umm = potential_function(dist, params)

    """
    import sympy
    from sympy.abc import r
    from sympy.utilities.lambdify import lambdify

    # Define functions in Sympy
    dufunc = sympy.simplify(sympy.diff(ufunc, r))
    sfunc = sympy.simplify(-sympy.diff(ufunc, r) / r)
    ummfunc = sympy.simplify(sympy.diff(dufunc, r))

    # LC: There is a problem with lambdify and lists:
    # https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    # Tried to fix it but there is a problem with param_names being a list of sympy Symbols
    u_lam = numba.njit(lambdify([r, param_names], ufunc, 'numpy'))  # Jitted python functions
    s_lam = numba.njit(lambdify([r, param_names], sfunc, 'numpy'))
    umm_lam = numba.njit(lambdify([r, param_names], ummfunc, 'numpy'))

    #@numba.njit
    def potential_function(r, params): # pragma: no cover
        u = np.float32(u_lam(r, params))
        s = np.float32(s_lam(r, params))
        umm = np.float32(umm_lam(r, params))
        return u, s, umm

    return potential_function

