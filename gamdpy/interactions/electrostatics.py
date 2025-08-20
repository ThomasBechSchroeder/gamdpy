import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt
import gamdpy as gp
from .interaction import Interaction

pi = numba.float32(math.pi)

class Electrostatics(Interaction):
    """Electrostatic point-like Coulomb interactions.
    
    This interaction is separated from the normal PairPotential class because 
    it uses a brute-force O(N^2) algorithm as a basis for the long-range part of Ewald sums.
        
    Parameters
    ----------
    damping : float
        Decay rate of the electrostatic gaussian screening. 
        If 0, the normal Coulomb potential is used.

    cutoff : nested list of floats
        Cutoff associated to pair-wise interactions between charges classes.
        Rows and columns MUST be sorted in ascending charges.
        :TODO: Add a check of this.
    """

    def __init__(self, damping, cutoff):
        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]
            return result
        self.params_function = params_function
        self.damping = numba.float32(damping)
        self.cutoff = cutoff

        if self.damping != 0.0:
            self.real_space_pot = gp.apply_shifted_potential_cutoff(gp.gaussian_screened_coulomb)
        else:
            self.real_space_pot = gp.apply_shifted_force_cutoff(gp.make_IPL_n(n=1))

        self.ewald = False

    def set_ewald(self, nk):
        self.nk = np.array(nk, dtype=np.float32) # [nkx, nky, nkz] number of wavevectors in each direction
        self.ewald = True

    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:
        # Gathering charges properties
        charges, self.charged_idx = configuration.get_charged_particles()
        coulomb_matrix, unique_charges, self.charges_types = self.build_pair_coulomb_matrix(charges)
        if self.damping != 0.0:
            params = [coulomb_matrix, self.cutoff]
        else:
            # Need to change this: decay rate is not a type-of-pairs quantity
            # Keeping it like that atm because it fits the current params format
            decay_rate = np.full_like(coulomb_matrix, self.damping, dtype=np.float32)
            params = [coulomb_matrix, decay_rate, self.cutoff]
    
        # Formatting params for kernels
        self.params, max_cut = self.format_pot_params(params)

        # Building reciprocal space constant attributes
        if self.ewald:
            self.kpoints = self.gen_k_grid(self.nk, configuration.simbox.get_lengths())
            self.poisson = self.compute_poisson_grid(self.kpoints, self.damping, configuration.get_volume())
            self.num_kpoints = self.kpoints.size()

        self.copy_to_device()
        if self.ewald:
            return (self.d_params, self.d_charged_idx, self.d_charges_types, \
                    self.self.d_kpoints, self.d_poisson, )
        else:
            return (self.d_params, self.d_charged_idx, self.d_charges_types, )

    def copy_to_device(self):
        self.d_params = cuda.to_device(self.params)
        self.d_charged_idx = cuda.to_device(self.charged_idx)
        self.d_charges_types = cuda.to_device(self.charges_types)
        if self.ewald:
            self.d_kpoints = cuda.to_device(self.kpoints)
            self.d_poisson = cuda.to_device(self.poisson)

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        num_cscalars = configuration.num_cscalars

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']

        # Unpack parameters
        D, N = configuration.D, configuration.N
        num_kpoints = self.num_kpoints
        num_charged = self.charged_idx.size()

        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        if gridsync:
            num_part = N
        else:
            num_part = num_charged
        num_blocks = (num_part - 1) // pb + 1 

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        if compute_u:
            u_id = configuration.sid['U']
        if compute_w:
            w_id = configuration.sid['W']
        if compute_lap:
            lap_id = configuration.sid['lapU']

        real_space_pot = self.real_space_pot

        virial_factor = numba.float32( 0.5/configuration.D )
        def real_space_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
            u, s, umm = real_space_pot(ij_dist, ij_params)
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

        def fourier_space_calculator(dr, qiqj, kpoint, poisson_k, my_f):
            dot_rk = numba.float(0.0)
            two = numba.float(0.0) 
            for d in range(D):
                dot_rk = dot_rk + dr[d] * kpoint[d]
            for d in range(D):
                my_f[d] = my_f[d] + two * qiqj * kpoint[d] * poisson_k * math.cos(dot_rk)
            return

        params_function = numba.njit(self.params_function)
        real_space_calculator = numba.njit(real_space_calculator)
        fourier_space_calculator = numba.njit(fourier_space_calculator)
        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())

        @cuda.jit( device=gridsync )  
        def calc_real_space(vectors, cscalars, sim_box, charges_idx, charges_types, params):
            """ 
            Calculate real space Ewald term.
            """
            
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y
            
            
            my_f = cuda.local.array(shape=D,dtype=numba.float32)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)
            my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)

            if global_id < num_charged:
                for k in range(D):
                    #my_r[k] = vectors[r_id][global_id,k]
                    my_f[k] = numba.float32(0.0)
                for k in range(num_cscalars):
                    my_cscalars[k] = numba.float32(0.0)
            
            cuda.syncthreads() # Make sure initializing global variables to zero is done

            if global_id < num_charged:
                part_id = charges_idx[global_id]
                my_charge_type = charges_types[global_id]
                for other_id in range(my_t, num_charged, tp):
                    other_part_id = charges_idx[other_id]
                    other_charge_type = charges_types[other_id]
                    if part_id != other_part_id:
                        dist_sq = dist_sq_dr_function(vectors[r_id][other_part_id], vectors[r_id][part_id], sim_box, my_dr)
                        ij_params = params_function(my_charge_type, other_charge_type, params)
                        cut = ij_params[-1]
                        if dist_sq < cut*cut:
                            real_space_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, 0, vectors[f_id], other_part_id)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (part_id, k), my_f[k])
                    
                for k in range(num_cscalars):
                    cuda.atomic.add(cscalars, (part_id, k), my_cscalars[k])

            return 

        @cuda.jit( device=gridsync )  
        def calc_fourier_space(vectors, sim_box, charges_idx, charges_types, params, kpoints, poisson_grid):
            """ 
            Calculate reciprocal space Ewald term.
            """
            
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y
            
            
            my_f = cuda.local.array(shape=D,dtype=numba.float32)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)

            if global_id < num_charged:
                for k in range(D):
                    my_f[k] = numba.float32(0.0)

            cuda.syncthreads() # Make sure initializing global variables to zero is done

            if global_id < num_charged:
                part_id = charges_idx[global_id]
                my_charge_type = charges_types[global_id]
                for other_id in range(my_t, num_charged, tp):
                    other_part_id = charges_idx[other_id]
                    other_charge_type = charges_types[other_id]
                    if part_id != other_part_id:
                        qiqj = params_function(my_charge_type, other_charge_type, params)[0]
                        for k_idx in range(num_kpoints):
                            kpoint = kpoints[k_idx]
                            poisson_k = poisson_grid[k_idx]
                            dist_sq = dist_sq_dr_function(vectors[r_id][other_part_id], vectors[r_id][part_id], sim_box, my_dr)
                            fourier_space_calculator(my_dr, qiqj, kpoint, poisson_k, my_f)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (part_id, k), my_f[k])

            return 

        if gridsync:
            # A device function, 
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, charged_idx, charges_types, kpoints, poisson_grid, = interaction_parameters
                calc_real_space(vectors, scalars, ptype, sim_box, charged_idx, charges_types, params)
                calc_fourier_space(vectors, sim_box, charged_idx, charges_types, params, kpoints, poisson_grid)
                return
            return compute_interactions
        
        else:
            # A python function, 
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, charged_idx, charges_types, kpoints, poisson_grid, = interaction_parameters
                calc_real_space[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, \
                                                      charged_idx, charges_types, params)
                calc_fourier_space[num_blocks, (pb, tp)](vectors, sim_box, charged_idx, \
                                                         charges_types, params, kpoints, poisson_grid)
                return
            return compute_interactions

    @staticmethod
    def build_pair_coulomb_matrix(charges):
        """
        Build the matrix of all unique charges-pair products.

        Parameters
        ----------
        charges : numpy array
            Assigning a charge to each particle
        
        Returns
        -------
        coulomb_matrix : numpy array
            Matrix of all unique q_i*q_j products
        
        unique_charges : numpy array
            Sorted array of unique charges composing the coulomb_matrix

        charges_types : numpy array
            Charge class associated to each particle.
            A particle with class i and another with class j refer to the (ij) entry
            of the coulomb_matrix.
        """
        unique_charges, charges_types = np.unique(charges, return_inverse=True)
        coulomb_matrix = np.outer(unique_charges, unique_charges)
        return coulomb_matrix, unique_charges, charges_types

    @staticmethod
    def gen_k_grid(nk, box_size):
        grid_coords = np.meshgrid(*(np.arange(0, n) for n in nk), indexing='ij')
        k_points = 2 * pi * np.stack(grid_coords, axis=-1).reshape(-1, len(box_size))
        k_points = np.delete(k_points, 0, axis=0) # remove k = [0, 0, 0] term
        return k_points / box_size

    @staticmethod
    def compute_poisson_grid(k_points, kappa, volume):
        # Helper variables
        four = numba.float32(4.0)
        kappa2 = kappa * kappa
        k2 = np.linalg.norm(k_points, axis=-1)
        return four * pi * math.exp(-k2 / (four * kappa2)) / (volume * k2)
    
    @staticmethod
    def format_pot_params(params_):
        num_classes = params_[0].shape[0]
        num_params = len(params_)

        params = np.zeros((num_classes, num_classes), dtype="f,"*num_params)
        for i in range(num_classes):
            for j in range(num_classes):
                plist = []
                for parameter in params_:
                    plist.append(parameter[i,j])
                params[i,j] = tuple(plist)

        max_cut = np.float32(np.max(params_[-1]))

        return params, max_cut