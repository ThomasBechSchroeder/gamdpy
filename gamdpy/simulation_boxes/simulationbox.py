
from abc import ABC, abstractmethod

class SimulationBox(ABC):
    """
    Abstract Base Class specifying the requirements for a SimulationBox
    """


    def make_device_copy(self):
        """ Creates a new device copy of the simbox data and returns it to the caller.
        To be used by neighbor list for recording the box state at time of last rebuild"""
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def copy_to_device(self):
        pass

    @abstractmethod
    def copy_to_host(self):
        pass


    @abstractmethod
    def get_volume_function(self) -> callable:
        """
        Returns the function which calculates the volume of the simulation box
        """
        pass

    @abstractmethod
    def get_volume(self) -> float:
        """
        Calculate and return the volume, on the host
        """
        pass

    @abstractmethod
    def scale(self, scale_factor):
        """
        Rescale the box by a factor scale_factor, in all directions, if scale_factor is a single float
        or by a different factor in each direction if scale_factor is an array of length D
        """
        pass

    @abstractmethod
    def get_dist_sq_dr_function(self) -> callable:
        """Generates function dist_sq_dr which computes displacement and distance for one neighbor """
        pass

    @abstractmethod
    def get_dist_sq_function(self) -> callable:
        """Generates.function dist_sq_function which computes distance squared for one neighbor """
        pass

    @abstractmethod
    def get_apply_PBC(self) -> callable:
        pass

    @abstractmethod
    def get_dist_moved_sq_function(self) -> callable:
        pass

    @abstractmethod
    def get_dist_moved_exceeds_limit_function(self) -> callable:
        pass

    @abstractmethod
    def get_loop_x_addition(self) -> int:
        pass

    @abstractmethod
    def get_loop_x_shift_function(self) -> callable:
        pass
