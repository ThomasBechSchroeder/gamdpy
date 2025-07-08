from abc import ABC, abstractmethod
from gamdpy import Configuration

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

