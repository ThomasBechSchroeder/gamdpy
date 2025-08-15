import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt
import gamdpy as gp
from .interaction import Interaction

class Electrostatics(Interaction):
    """Electrostatic point-like Coulomb interactions.
    
    This interaction is separated from the normal PairPotential class because 
    it uses a brute-force O(N^2) algorithm as a basis for the long-range part of Ewald sums.
        
    Parameters
    ----------
    params : nested list of floats
        Interaction parameters - charges per type (list), screening decay rate (float) and real-space cutoff (nested list)
    """

    def __init__(self, params):
        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]
            return result            

        self.shifted_damped_coulomb = gp.apply_shifted_potential_cutoff(gp.gaussian_screened_coulomb)
        self.params_function = params_function
        self.set_coulomb_params(params)
        self.ewald = False

    def set_coulomb_params(self, params):
        self.charges_per_type = np.array(params[0], dtype=np.float32) # format [q_type0, q_type1, ...]
        charges_product = np.outer(params[0], params[0]).astype(np.float32)
        # Need to change this: decay rate is not a type-of-pairs quantity
        # Keeping it like that atm because it fits the current params format
        decay_rate = np.full_like(charges_product, params[1], dtype=np.float32)
        cutoff = np.array(params[2], dtype=np.float32)

        self.coulomb_params = [charges_product, decay_rate, cutoff]

    def set_ewald(self, nk):
        self.nk = nk # [nkx, nky, nkz] number of wavevectors in each direction
        self.ewald = True

    def prepare_coulomb_params(self):
        num_types = self.coulomb_params[0].shape[0]
        num_params = len(self.coulomb_params)

        # Convert params to the format required by kernels (num_types x num_types) array of tuples (p0, p1, ..., cutoff)
        params = np.zeros((num_types, num_types), dtype="f,"*num_params)
        for i in range(num_types):
            for j in range(num_types):
                plist = []
                for parameter in self.coulomb_params:
                    plist.append(parameter[i,j])
                params[i,j] = tuple(plist)

        max_cut = np.float32(np.max(self.coulomb_params[-1]))

        return params, max_cut

    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:
        
        self.params, max_cut = self.prepare_coulomb_params()
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

        shifted_damped_coulomb = self.shifted_damped_coulomb
        charges_per_type = self.charges_per_type

        virial_factor = numba.float32( 0.5/configuration.D )
        def coulomb_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
            u, s, umm = shifted_damped_coulomb(ij_dist, ij_params)
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
        coulomb_calculator = numba.njit(coulomb_calculator)
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
                my_type = ptype_function(global_id, ptype)
                global_has_charge = charges_per_type[my_type] != 0
                for k in range(D):
                    #my_r[k] = vectors[r_id][global_id,k]
                    my_f[k] = numba.float32(0.0)
                for k in range(num_cscalars):
                    my_cscalars[k] = numba.float32(0.0)
            
            cuda.syncthreads() # Make sure initializing global variables to zero is done

            if global_id < num_part and global_has_charge:
                for other_id in range(my_t, num_part, tp):
                    if other_id != global_id:
                        other_type = ptype_function(other_id, ptype)
                        if charges_per_type[other_type] != 0:
                            dist_sq = dist_sq_dr_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box, my_dr)
                            ij_params = params_function(my_type, other_type, params)
                            cut = ij_params[-1]
                            if dist_sq < cut*cut:
                                coulomb_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, 0, vectors[f_id], other_id)
                        else:
                            continue
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


