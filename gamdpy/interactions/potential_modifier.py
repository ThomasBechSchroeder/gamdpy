import numba
from numba import cuda
from abc import ABC, abstractmethod
from gamdpy import Configuration
from typing import Callable

class PotentialModifier(ABC):
    """
    Abstract Base Class specifying the requirements for a potential modifier (eg. shifted_potential, shifted_force...)
    """

    @abstractmethod   
    def modified_potential(self, potential_function: Callable) -> Callable:
        """
        Get a python function that calculates u, s, and umm for the potential_functions with the requested modification
        """

    @abstractmethod
    def modified_params(self, potential_function: Callable, params: list[float]) -> list[float]:
        """
        Add the parameters required by the modification to the list of parameters, keeping the cutoff as the last entry
        """

class NoModification(PotentialModifier):
    """
    'Modifier' that doesn't actually modify a pair-potential
    """
    
    def modified_params(self, potential_function: Callable, params: list[float]) -> list[float]:
        return params
    
    def modified_potential(self, potential_function: Callable) -> Callable:
        return potential_function


class ShiftedPotential(PotentialModifier):
    """
    Modifier that applies shifted potential cutoff to a pair-potential
    Original potential:
        u, s, umm = potential(r, params)
    Modified potential (subtracting value at cut-off):
        mu = u + A 
            A = -u(cut) 
        ms = -dmu/dr / r = s
        mumm = umm
    """

    def modified_params(self, potential_function: Callable, params: list[float]) -> list[float]:
        """
        Add the parameters required by the modification to the list of parameters, 
        keeping the cutoff as the last entry.
        """
        cut = params[-1]
        u_cut, s_cut, umm_cut = potential_function(cut, params)
        A = -u_cut

        return [*params[:-1], numba.float32(A), cut]
 
    def modified_potential(self, potential_function: Callable) -> Callable:
        """
        Make a python function that calculates u, s, and umm for the shifted potential_function 
        """

        potential_function = numba.njit(potential_function)
        
        def new_potential(dist, params):
            A = params[-2]
            u, s, umm = potential_function(dist, params)

            return u+A, s, umm
        
        return new_potential


class ShiftedForce(PotentialModifier):
    """
    Modifier that applies shifted force cutoff to a pair-potential
    Original potential:
        u, s, umm = potential(r, params)
    Modified potential (subtracting first order Taylor at cut-off):
        mu = u + A + B(r - cut)
            A = -u(cut) 
            B = -du/dr(cut) = s(cut)*cut
        ms = -dmu/dr / r = -d[u + Br]/dr / r = s - B/r
        mumm = umm
    """

    def modified_params(self, potential_function: Callable, params: list[float]) -> list[float]:
        """
        Add the parameters required by the modification to the list of parameters, 
        keeping the cutoff as the last entry
        """
        cut = params[-1]
        u_cut, s_cut, umm_cut = potential_function(cut, params)
        A = -u_cut
        B = s_cut*cut
        
        return [*params[:-1], numba.float32(A),  numba.float32(B), cut]

    def modified_potential(self, potential_function: Callable) -> Callable:
        """
        Make a python function that calculates u, s, and umm for the shifted force potential
        """
        
        potential_function = numba.njit(potential_function)
        
        def new_potential(dist, params):
            A = params[-3]
            B = params[-2]
            cut = params[-1]

            u, s, umm = potential_function(dist, params)

            return u + A + B*(dist-cut), s - B/dist, umm
        
        return new_potential

