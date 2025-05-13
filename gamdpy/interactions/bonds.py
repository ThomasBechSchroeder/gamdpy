import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # bonds is an example of 'fixed' interactions

# Abstract Base Class and type annotation
from .interaction import Interaction
from gamdpy import Configuration


class Bonds(Interaction):
    """ Fixed bond interactions between particles, such as harmonic bonds or FENE bonds.

    Parameters
    ----------

    bond_potential : function
        A function that takes the distance between two particles and the bond type as arguments and returns the potential energy, force and laplacian.
        See :func:gamdpy.potential_functions.harmonic_bond_function for an example.

    potential_params : list
        A list of parameters for each bond type. Each entry is a list of parameters for a specific bond type.

    indices : list
        A list of lists, each containing the indices of the two particles involved in a bond and the bond

    See Also
    --------

    gamdpy.harmonic_bond_function : Harmonic bond potential

    """
    def __init__(self, bond_potential, potential_params, indices):
        self.bond_potential = bond_potential
        self.potential_params = potential_params
        self.indices = indices

    def get_params(self, configuration: Configuration, compute_plan: dict, verbose=False) -> tuple:
        self.N = configuration.N
        self.potential_params_array = np.array(self.potential_params, dtype=np.float32)
        assert len(self.potential_params_array.shape)==2 # for now...
        self.num_types = self.potential_params_array.shape[0]
        self.indices_array = np.array(self.indices, dtype=np.int32)
        assert self.indices_array.shape[1] == 3 # i, j, bond_type
        assert max(self.indices_array[:,-1]) <= self.num_types 

        if verbose:
            print(f'Setting up bond interactions: {self.N} particles,')
            print(f'{self.num_types} bond types, {self.indices_array.shape[0]} bonds in total.')

        self.d_indices = cuda.to_device(self.indices_array)
        self.d_params = cuda.to_device(self.potential_params_array)
        return (self.d_indices, self.d_params)


    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']
        compute_stresses = compute_flags['stresses']

        if verbose:
            print('get_kernel: Bonds:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks*pb}')
            print(f'\tNumber of threads {num_blocks*pb*tp}')
            if compute_stresses:
                print('\tIncluding computation of stress tensor')
    
        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        if compute_u:
            u_id = configuration.sid['U']
        if compute_w:
            w_id = configuration.sid['W']
        if compute_lap:
            lap_id = configuration.sid['lapU']

        if compute_stresses:
            sx_id = configuration.vectors.indices['sx']
            if D > 1:
                sy_id = configuration.vectors.indices['sy']
                if D > 2:
                    sz_id = configuration.vectors.indices['sz']
                    if D > 3:
                        sw_id = configuration.vectors.indices['sw']


        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
        bondpotential_function = numba.njit(self.bond_potential)

        virial_factor = numba.float32( 0.5/configuration.D)
        half = numba.float32(0.5)

        def bond_calculator(vectors, scalars, ptype, sim_box, indices, values):

            dr = cuda.local.array(shape=D,dtype=numba.float32)
            dist_sq = dist_sq_dr_function(vectors[r_id][indices[1]], vectors[r_id][indices[0]], sim_box, dr)
            u, s, umm = bondpotential_function(math.sqrt(dist_sq), values[indices[2]])

            for k in range(D):
                cuda.atomic.add(vectors, (f_id, indices[0], k), -dr[k]*s)      # Force
                cuda.atomic.add(vectors, (f_id, indices[1], k), +dr[k]*s)
                if compute_w:
                    cuda.atomic.add(scalars, (indices[0], w_id), dr[k]*dr[k]*s*virial_factor)    # Virial
                    cuda.atomic.add(scalars, (indices[1], w_id), dr[k]*dr[k]*s*virial_factor)
                if compute_stresses:
                    cuda.atomic.add(vectors[sx_id], (indices[0], k), - half * s * dr[0] * dr[k])
                    cuda.atomic.add(vectors[sx_id], (indices[1], k), - half * s * dr[0] * dr[k])
                    if D > 1:
                        cuda.atomic.add(vectors[sy_id], (indices[0], k), - half * s * dr[1] * dr[k])
                        cuda.atomic.add(vectors[sy_id], (indices[1], k), - half * s * dr[1] * dr[k])
                        if D > 2:
                            cuda.atomic.add(vectors[sz_id], (indices[0], k), - half * s * dr[2] * dr[k])
                            cuda.atomic.add(vectors[sz_id], (indices[1], k), - half * s * dr[2] * dr[k])
                            if D > 3:
                                cuda.atomic.add(vectors[sw_id], (indices[0], k), - half * s * dr[3] * dr[k])
                                cuda.atomic.add(vectors[sw_id], (indices[1], k), - half * s * dr[3] * dr[k])


            if compute_u:
                cuda.atomic.add(scalars, (indices[0], u_id), u*numba.float32(0.5)) # Potential enerrgy
                cuda.atomic.add(scalars, (indices[1], u_id), u*numba.float32(0.5))
            if compute_lap:
                lap = numba.float32(1-D)*s + umm                                   # Laplacian
                cuda.atomic.add(scalars, (indices[0], lap_id), lap)
                cuda.atomic.add(scalars, (indices[1], lap_id), lap)


            return

        return make_fixed_interactions(configuration, bond_calculator, compute_plan, verbose=False)

    def get_exclusions(self, configuration, max_number_exclusions=20):
        exclusions = np.zeros((configuration.N, max_number_exclusions+1), dtype=np.int32) # last entry: number of actual exclusions
        for i, j, _ in self.indices: # bond involving particles 'i' and 'j'. We dont care about the bond type (maybe later)
            if exclusions[i,-1] < max_number_exclusions:
                exclusions[i,exclusions[i,-1]] = j
            exclusions[i,-1] += 1
            
            if exclusions[j,-1] < max_number_exclusions:
                exclusions[j,exclusions[j,-1]] = i
            exclusions[j,-1] += 1
    
        assert np.max(exclusions[:,-1]) <= max_number_exclusions
        return exclusions

