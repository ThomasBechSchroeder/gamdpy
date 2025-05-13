import numpy as np
import numba
import math
from numba import cuda

from ..simulation.get_default_compute_flags import get_default_compute_flags

# gamdpy
import gamdpy as gp

class Evaluator:
    """ Evaluates interactions between particles in a configuration.
    
    This class can be used to evaluate interactions between particles in a configuration
    that are different from the ones of the Simulation class.

    Parameters
    ----------

    configuration : rumd.Configuration
        The configuration for which the interactions are to be evaluated.

    interactions : an interaction
        Interactions such as pair potentials, bonds, external fields, etc.

    compute_plan : dict, optional
         A dictionary with the compute plan for the simulation. If None, a default compute plan is used.

    verbose : bool, optional
        If True, print information about the interactions.
    
    """
    def __init__(self, configuration, interactions, compute_plan=None, verbose=True):
        
        self.configuration = configuration

        if compute_plan==None:
            self.compute_plan = gp.get_default_compute_plan(self.configuration)
        else:
            self.compute_plan = compute_plan

        compute_flags = get_default_compute_flags()

        # Make sure interactions is a list
        if type(interactions) == list:
            self.interactions = interactions
        else:
            self.interactions = [interactions, ]

        self.interactions_kernel, self.interactions_params = gp.add_interactions_list(
            self.configuration,
            self.interactions,
            compute_plan=self.compute_plan,
            compute_flags=compute_flags,
            verbose=verbose)

        self.evaluater_func = self.make_evaluater_func(self.configuration, self.interactions_kernel, self.compute_plan, verbose)
        
    def make_evaluater_func(self, configuration, compute_interactions, compute_plan, verbose=True):
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1
    
        if gridsync:
            # Return a kernel that evaluates interactions 
            @cuda.jit
            def evaluater(vectors, scalars, ptype, r_im, sim_box, interaction_params):
                grid = cuda.cg.this_grid()
                compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                return
            return evaluater[num_blocks, (pb, tp)]

        else:

            # Return a Python function that evaluates interactions
            def evaluater(vectors, scalars, ptype, r_im, sim_box, interaction_params):
                compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                return
            return evaluater
        return            

    def evaluate(self, configuration=None):
      
        if configuration==None:
            configuration = self.configuration
        configuration.copy_to_device()
        self.evaluater_func(configuration.d_vectors, 
                            configuration.d_scalars, 
                            configuration.d_ptype, 
                            configuration.d_r_im, 
                            configuration.simbox.d_data, 
                            self.interactions_params, 
                            )
        configuration.copy_to_host()
        return 
