from abc import ABC, abstractmethod
from ..configuration import Configuration
import h5py

class Integrator(ABC):
    """
    Abstract Base Class specifying the requirements for an integrator
    """

    @abstractmethod
    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict, interactions_kernel):
        """
        Get a kernel (or python function depending on compute_plan["gridsync"]) that implements performing a number of steps of the integrator
        """

    @abstractmethod
    def get_params(self, configuration: Configuration, interactions_params: dict) -> tuple :
        """
        Get a tuple with the parameters expected by the associated kernel
        """

    @abstractmethod
    def save_internal_state(self, output: h5py.File, group_name: str):
        """
        Write the internal state of the integrator as an attribute called 'integrator_state' to the specified group in the specified HDF file
        """
        # It could be that each integrator uses its own attribute, i.e. we don't need to insist that it's always called the generic name integrator_state

