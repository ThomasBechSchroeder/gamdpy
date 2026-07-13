import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt
import gamdpy as gp
from .interaction import Interaction

class PairPotentialNsquared(Interaction):
    """  Pairwise interaction potential for a system of particles. 
    
    This interaction uses a brute-force O(N^2) algorithm, mostly
    usefull for very small systems and testing
    
    Does _not_ utilize Newtons 3rd law (Ignores the UtilizeNIII flag in compute_plan)
    
    Does _not do stresses (to simplify the code)_

    Parameters
    ----------
    pairpotential_function : callable
        A JIT compiled function `f(r, params)` that takes a separation distance `r` (float)
        and a list of parameters, and returns a triplet of floats:
            - Potential energy, :math:`u(r)`
            - Force multiplier, :math:`-u'(r)/r`
            - Second derivative of potential energy, :math:`u''(r)`
    params : list of floats or nested list of floats
        Interaction parameters for the pair potential function. Use a nested list for multiple types of particles.
        The last element of each list is the cutoff distance of the pair potential.
    exclusions
        List of particle indices to exclude from interactions for each particle.
    """

    def __init__(self, pairpotential_function, params, exclusions=None):
        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]            # default: read from params array
            return result            
    
        self.pairpotential_function = pairpotential_function
        self.params_function = params_function
        self.params_user = params
        self.exclusions = exclusions

    def convert_user_params(self):
        # Upgrade any scalar parameters to 1x1 numpy array
        num_params = len(self.params_user)
        params_list = []
        for parameter in self.params_user:
            if np.isscalar(parameter):
                params_list.append(np.ones((1,1))*parameter)
            else:
                params_list.append(np.array(parameter, dtype=np.float32))

        # Ensure all parameters are the right format (num_types x num_types) numpy arrays
        num_types = params_list[0].shape[0]
        for parameter in params_list:
            assert len(parameter.shape) == 2
            assert parameter.shape[0] == num_types
            assert parameter.shape[1] == num_types

        # Convert params to the format required by kernels (num_types x num_types) array of tuples (p0, p1, ..., cutoff)
        params = np.zeros((num_types, num_types), dtype="f,"*num_params)
        for i in range(num_types):
            for j in range(num_types):
                plist = []
                for parameter in params_list:
                    plist.append(parameter[i,j])
                params[i,j] = tuple(plist)

        max_cut = np.float32(np.max(params_list[-1]))

        return params, max_cut
               
    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:
        
        self.params, max_cut = self.convert_user_params()
        self.d_params = cuda.to_device(self.params)

        return (self.d_params, )

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        num_cscalars = configuration.num_cscalars

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1  

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        if compute_u:
            u_id = configuration.sid['U']
        if compute_w:
            w_id = configuration.sid['W']
        if compute_lap:
            lap_id = configuration.sid['lapU']

        pairpotential_function = numba.njit(self.pairpotential_function)
 
        virial_factor = numba.float32( 0.5/configuration.D )
        def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
            u, s, umm = pairpotential_function(ij_dist, ij_params)
            half = numba.float32(0.5)
            for k in range(D):
                my_f[k] = my_f[k] - dr[k]*s                         # Force
                if compute_w:
                    cscalars[w_id] += dr[k]*dr[k]*s*virial_factor       # Virial
            if compute_u:
                cscalars[u_id] += half*u                                # Potential energy
            if compute_lap:
                cscalars[lap_id] += numba.float32(1-D)*s + umm          # Laplacian 
                return

        ptype_function = numba.njit(configuration.ptype_function)
        params_function = numba.njit(self.params_function)
        pairpotential_calculator = numba.njit(pairpotential_calculator)
        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
    
        @cuda.jit( device=gridsync )  
        def calc_forces(vectors, cscalars, ptype, sim_box, params):
            """ Calculate forces as given by pairpotential_calculator() (needs to exist in outer-scope) using nblist 
                Kernel configuration: [num_blocks, (pb, tp)]        
            """
            
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y
            
            
            my_f = cuda.local.array(shape=D,dtype=numba.float32)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)
            my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)
            
            if global_id < num_part:
                for k in range(D):
                    #my_r[k] = vectors[r_id][global_id,k]
                    my_f[k] = numba.float32(0.0)
                for k in range(num_cscalars):
                    my_cscalars[k] = numba.float32(0.0)
                my_type = ptype_function(global_id, ptype)
            
            cuda.syncthreads() # Make sure initializing global variables to zero is done

            if global_id < num_part:
                for other_id in range(my_t, num_part, tp):
                    if other_id != global_id:
                        other_type = ptype_function(other_id, ptype)
                        dist_sq = dist_sq_dr_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box, my_dr)
                        ij_params = params_function(my_type, other_type, params)
                        cut = ij_params[-1]
                        if dist_sq < cut*cut:
                            pairpotential_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, 0, vectors[f_id], other_id)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])
                    
                for k in range(num_cscalars):
                    cuda.atomic.add(cscalars, (global_id, k), my_cscalars[k])

            return 

        if gridsync:
            # A device function, 
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, = interaction_parameters
                calc_forces(vectors, scalars, ptype, sim_box, params)
                return
            return compute_interactions
        
        else:
            # A python function, 
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, = interaction_parameters
                calc_forces[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, params)
                return
            return compute_interactions


