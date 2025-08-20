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
    params : nested list of floats
        Interaction parameters - charges per type (list) and real-space cutoff (nested list)
    damping : float
        Decay rate of the electrostatic gaussian screening. 
        If 0, the normal Coulomb potential is used.
    """

    def __init__(self, params, damping):
        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]
            return result
        self.params_function = params_function
        self.damping = numba.float32(damping)
        self.set_pot_params(params)
        if self.damping != 0.0:
            self.real_space_pot = gp.apply_shifted_potential_cutoff(gp.gaussian_screened_coulomb)
        else:
            self.real_space_pot = gp.apply_shifted_force_cutoff(gp.make_IPL_n(n=1))
        self.ewald = False

    def set_pot_params(self, params):
        self.charges_per_type = np.array(params[0], dtype=np.float32) # format [q_type0, q_type1, ...]
        charges_product = np.outer(params[0], params[0]).astype(np.float32)
        cutoff = np.array(params[-1], dtype=np.float32)
        if self.damping != 0.0:
            self.pot_params = [charges_product, cutoff]
        else:
            # Need to change this: decay rate is not a type-of-pairs quantity
            # Keeping it like that atm because it fits the current params format
            decay_rate = np.full_like(charges_product, self.damping, dtype=np.float32)
            self.pot_params = [charges_product, decay_rate, cutoff]

    def set_ewald(self, nk):
        self.nk = np.array(nk, dtype=np.float32) # [nkx, nky, nkz] number of wavevectors in each direction
        self.ewald = True

    def format_pot_params(self):
        num_types = self.pot_params[0].shape[0]
        num_params = len(self.pot_params)

        # Convert params to the format required by kernels (num_types x num_types) array of tuples (p0, p1, ..., cutoff)
        params = np.zeros((num_types, num_types), dtype="f,"*num_params)
        for i in range(num_types):
            for j in range(num_types):
                plist = []
                for parameter in self.pot_params:
                    plist.append(parameter[i,j])
                params[i,j] = tuple(plist)

        max_cut = np.float32(np.max(self.pot_params[-1]))

        return params, max_cut

    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:
        self.params, max_cut = self.format_pot_params()
        if self.ewald:
            self.kpoints = self.gen_k_grid(self.nk, configuration.simbox.get_lengths())
            self.poisson = self.compute_poisson_grid(self.kpoints, self.damping, configuration.get_volume())
            self.num_kpoints = self.kpoints.size()
        self.copy_to_device()
        if self.ewald:
            return (self.d_params, self.d_kpoints, self.d_poisson, )
        else:
            return (self.d_params, )

    def copy_to_device(self):
        self.d_params = cuda.to_device(self.params)
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

        # Reorder charged particles to only loop over them
        new_order, num_charged = configuration.order_charged_system(self.charges_per_type, reorder=True)

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

        def fourier_space_calculator(dr, kpoint, poisson_k, my_f):
            dot_rk = numba.float(0.0)
            two = numba.float(0.0)
            for d in range(D):
                dot_rk = dot_rk + dr[d] * kpoint[d]
            for d in range(D):
                my_f[d] = my_f[d] + two * kpoint[d] * poisson_k * math.cos(dot_rk)
            return

        ptype_function = numba.njit(configuration.ptype_function)
        params_function = numba.njit(self.params_function)
        real_space_calculator = numba.njit(real_space_calculator)
        fourier_space_calculator = numba.njit(fourier_space_calculator)
        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())

        @cuda.jit( device=gridsync )  
        def calc_real_space(vectors, cscalars, ptype, sim_box, params):
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
                my_type = ptype_function(global_id, ptype)
                for k in range(D):
                    #my_r[k] = vectors[r_id][global_id,k]
                    my_f[k] = numba.float32(0.0)
                for k in range(num_cscalars):
                    my_cscalars[k] = numba.float32(0.0)
            
            cuda.syncthreads() # Make sure initializing global variables to zero is done

            if global_id < num_charged:
                for other_id in range(my_t, num_charged, tp):
                    if other_id != global_id:
                        other_type = ptype_function(other_id, ptype)
                        dist_sq = dist_sq_dr_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box, my_dr)
                        ij_params = params_function(my_type, other_type, params)
                        cut = ij_params[-1]
                        if dist_sq < cut*cut:
                            real_space_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, 0, vectors[f_id], other_id)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])
                    
                for k in range(num_cscalars):
                    cuda.atomic.add(cscalars, (global_id, k), my_cscalars[k])

            return 

        @cuda.jit( device=gridsync )  
        def calc_fourier_space(vectors, sim_box, kpoints, poisson_grid):
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
                for other_id in range(my_t, num_charged, tp):
                    if other_id != global_id:
                        for k_idx in range(num_kpoints):
                            kpoint = kpoints[k_idx]
                            poisson_k = poisson_grid[k_idx]
                            dist_sq = dist_sq_dr_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box, my_dr)
                            fourier_space_calculator(my_dr, kpoint, poisson_k, my_f)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])

            return 

        if gridsync:
            # A device function, 
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, kpoints, poisson_grid, = interaction_parameters
                calc_real_space(vectors, scalars, ptype, sim_box, params)
                calc_fourier_space(vectors, sim_box, kpoints, poisson_grid)
                return
            return compute_interactions
        
        else:
            # A python function, 
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, kpoints, poisson_grid, = interaction_parameters
                calc_real_space[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, params)
                calc_fourier_space[num_blocks, (pb, tp)](vectors, sim_box, kpoints, poisson_grid)
                return
            return compute_interactions

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