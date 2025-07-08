
from abc import ABC, abstractmethod

class SimulationBox(ABC):
    """
    Abstract Base Class specifying the requirements for a SimulationBox
    """

    def make_device_copy(self):
        """
        Creates a new device copy of the simbox data and returns it to the caller.
        To be used by neighbor list for recording the box state at time of last rebuild.
        """

    @abstractmethod
    def get_name(self):
        """ Return name of the SimulationBox. """

    @abstractmethod
    def copy_to_device(self):
        """ Copy SimulationBox data from host to device. """

    @abstractmethod
    def copy_to_host(self):
        """ Copy SimulationBox data from device to host. """

    @abstractmethod
    def get_volume_function(self) -> callable:
        """ Returns the function which calculates the volume of the simulation box. """

    @abstractmethod
    def get_volume(self) -> float:
        """ Return the volume of the simulation box. """

    @abstractmethod
    def scale(self, scale_factor):
        """
        Rescale the box by a factor scale_factor, in all directions, if scale_factor is a single float
        or by a different factor in each direction if scale_factor is an array of length D.
        """

    @abstractmethod
    def get_dist_sq_dr_function(self) -> callable:
        """ Generates function dist_sq_dr which computes displacement and distance for one neighbor. """

    @abstractmethod
    def get_dist_sq_function(self) -> callable:
        """ Generates function dist_sq_function which computes distance squared for one neighbor. """

    @abstractmethod
    def get_apply_PBC(self) -> callable:
        """ Apply periodic boundary conditions (PBC). """

    #@abstractmethod
    #def get_dist_moved_sq_function(self) -> callable:
    #    pass

    # LC: The following three methods could be not abstract because they are specific to SLLOD
    @abstractmethod
    def get_dist_moved_exceeds_limit_function(self) -> callable:
        """ Does something SLLODish. """

    @abstractmethod
    def get_loop_x_addition(self) -> int:
        """ Does something SLLODish. """

    @abstractmethod
    def get_loop_x_shift_function(self) -> callable:
        """ Does something SLLODish. """
