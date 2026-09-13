import numpy as np
import numba
import math
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


# Abstract Base Class and type annotation
from gamdpy.runtime_actions import RuntimeAction
from gamdpy import Configuration

from numba.cuda import libdevice

# Could include flags of dimensions to work on
class ParticleSwapper(RuntimeAction):
    """
    Perform Metropolis Monte Carlo particle-type swaps on the GPU.

    Multiple swap attempts are evaluated in parallel on the GPU. The
    first accepted swap in each round is selected and applied to the
    configuration. The rest are discarded.
    """

    dtype = numba.float32 


    def __init__(self, steps_between_swaps: int, swap_attempts: int, pair_potential, swap_temperature, seed) -> None:
        if type(steps_between_swaps) != int or steps_between_swaps < 0:
            raise ValueError(f'steps_between_swaps ({steps_between_swaps}) should be non-negative integer.')
        self.steps_between_swap = steps_between_swaps

        if type(swap_attempts) != int or swap_attempts < 0:
            raise ValueError(f'swaps attempts per swap session({swap_attempts}) should be non-negative integer.')
        self.swap_attempts = swap_attempts

        self.pair_potential = pair_potential
        self.swap_temperature = swap_temperature
        self.seed = seed

    def reset_success_counter(self):
        """
        Reset the success counter for the next run.
        """
        self.success_counter[0] = 0


    def setup(self, simulation, num_timeblocks: int, steps_per_timeblock: int, output, verbose=False) -> None:
        pass

    def get_params(self, configuration: Configuration, compute_plan: dict) -> tuple:
        self.configuration = configuration
        number_of_parallel_attempts = configuration.N*compute_plan['tp']
        self.rng_states = create_xoroshiro128p_states(number_of_parallel_attempts, seed=self.seed) 
        self.first_success_idx = cuda.device_array(1, dtype=np.int32) 
        self.attempts_counter = cuda.device_array(1, dtype=np.int32)
        self.success_counter = cuda.device_array(1, dtype=np.int32)
        self.success_counter[0] = 0
        
        self.params = (self.swap_attempts,
            self.swap_temperature, 
            self.rng_states,
            self.pair_potential.nblist.d_nblist,
            self.pair_potential.d_params,
            self.first_success_idx, 
            self.attempts_counter, 
            self.success_counter
            )

        return  self.params # return parameters as a tuple

    def get_prestep_kernel(self, configuration: Configuration, compute_plan: dict):

        # Reason for prestep kernel: we are guarenteed that NBlist is ok
        # - but then it uses wrong forces for subsequent move...
        # maybe do it post-step, and call check NBlist first?

        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        if gridsync:
            def kernel(grid, vectors, scalars, ptype, r_im, sim_box, step, momentum_reset_params):
                pass
                return
            return cuda.jit(device=gridsync)(kernel)
        else:
            def kernel(grid, vectors, scalars, ptype, r_im, sim_box, step, momentum_reset_params):
                pass
            return kernel

    def get_poststep_kernel(self, configuration: Configuration, compute_plan: dict):

        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        r_id = configuration.vectors.indices['r']
        D, num_particles = configuration.D, configuration.N
        steps_between_swap = self.steps_between_swap

        pair_pot = self.pair_potential 
        dtype = self.dtype 
        pairpotential_function = numba.njit(pair_pot.pairpotential_function) 
        dist_sq_function = numba.njit(configuration.simbox.get_dist_sq_function())

        @cuda.jit(device=True) 
        def energy_change_for_particle(particle_index, new_type, old_type, swap_index, 
                                       positions, ptype, sim_box, 
                                       neighbour_list, params): 
            """
            Calculate the change in interaction energy when the type of a
            particle is changed.

            Only interactions between the particle and its neighbours are
            considered. The interaction with the other particle involved
            in the proposed swap is excluded, since for these particular models 
            it cancels out in delta E.
            """

            num_neighbours = neighbour_list[particle_index, -1] 
            delta = dtype(0.0) 
            my_r = positions[particle_index] 
            my_global_id, my_t = cuda.grid(2)
 
            for n in range(my_t, num_neighbours, tp): 
                neighbour_index = neighbour_list[particle_index, n] 
                if neighbour_index == swap_index: 
                    continue 
 
                dist_sq = dist_sq_function(positions[neighbour_index], my_r, sim_box) 
                neighbour_type = ptype[neighbour_index] 
 
                new_params = params[new_type, neighbour_type] 
                old_params = params[old_type, neighbour_type] 
                new_cutoff = new_params[-1] 
                old_cutoff = old_params[-1] 
                max_cutoff = new_cutoff if new_cutoff > old_cutoff else old_cutoff 
 
                if dist_sq < max_cutoff * max_cutoff: 
                    distance = libdevice.sqrtf(dist_sq) 
                    if dist_sq < new_cutoff * new_cutoff: 
                        delta += pairpotential_function(distance, new_params)[0] # pick potential energy from returned tuple
                    if dist_sq < old_cutoff * old_cutoff:  
                        delta -= pairpotential_function(distance, old_params)[0] 
 
            return delta 

        @cuda.jit(device=True) 
        def save_random_int(upper_limit_exclusive, rng_states, thread_id):
            """
            Generate a random integer in the range [0, upper_limit_exclusive) using the
            xoroshiro128p random number generator.

            The generated integer is guaranteed to be less than upper_limit_exclusive.
            """
            random_int = upper_limit_exclusive
            while random_int >= upper_limit_exclusive:
                random_int = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * upper_limit_exclusive)
            return random_int

        @cuda.jit(device=gridsync) 
        def thread_swap_attempts(positions, ptype, sim_box, neighbour_list, params,
                                 rng_states, temperature, first_success_idx, attempts_counter, 
                                 number_of_swap_attempts, success_counter): 
                                 
                                             
            """
            Perform parallel Metropolis swap attempts.

            Each active CUDA thread proposes a swap between two particles of
            different types and evaluates its Metropolis acceptance
            criterion. The lowest-index accepted proposal is selected using
            an atomic minimum and is then applied to the configuration.

            The kernel continues until the requested number of sequential
            swap attempts has been performed.
            """
            
            grid = cuda.cg.this_grid() 
            my_global_id, my_t = cuda.grid(2)
            my_local_id = cuda.threadIdx.x
            thread_id = my_global_id

            s_i = cuda.shared.array(pb, dtype=np.int32)
            s_j = cuda.shared.array(pb, dtype=np.int32)
            s_delta = cuda.shared.array(pb, dtype=np.float32)
            
            # Determine the number of parallel swap attempts that can be performed in this kernel launch. 
            # The number of parallel attempts is limited by the total number of threads available and the total number of swap attempts requested.  
            number_of_parallel_attempts = min(cuda.blockDim.x*cuda.gridDim.x, number_of_swap_attempts) 
            #number_of_parallel_attempts = min(numba.cuda.blockDim.x*numba.cuda.blockDim.y*numba.cuda.gridDim.x *numba.cuda.gridDim.y, number_of_swap_attempts) 

            if thread_id == 0 and my_t == 0: 
                attempts_counter[0] = 0 
            grid.sync() 
            
            while attempts_counter[0] < number_of_swap_attempts:

                # Determine the number of swap attempts to perform in this round, as min(needed, possible). 
                attempts_this_round = min(number_of_swap_attempts - attempts_counter[0], number_of_parallel_attempts)
                if thread_id == 0 and my_t == 0:
                    first_success_idx[0] = attempts_this_round
                grid.sync() 
 
                if thread_id < first_success_idx[0]:
                    if my_t==0: # only one thread per attempt needs to sample the two particles to swap
                        i = save_random_int(num_particles, rng_states, thread_id)
                        j=i
                        while ptype[j] == ptype[i]: #only resample j if ptypes are the same
                            j = save_random_int(num_particles, rng_states, thread_id)
                        s_i[my_local_id] = i
                        s_j[my_local_id] = j
                        s_delta[my_local_id] = dtype(0.0)
                cuda.syncthreads()

                if thread_id < first_success_idx[0]:
                    # Broadcast the selected particle indices
                    i = s_i[my_local_id]
                    j = s_j[my_local_id]
                    type_i = ptype[i]
                    type_j = ptype[j]

                    # change in energy for particle i, then particle j
                    my_delta_energy  = energy_change_for_particle(i, type_j, type_i, j, positions, ptype, sim_box, neighbour_list, params) 
                    my_delta_energy += energy_change_for_particle(j, type_i, type_j, i, positions, ptype, sim_box, neighbour_list, params)
                    cuda.atomic.add(s_delta, my_local_id, my_delta_energy) 
                cuda.syncthreads()
                              
                if thread_id < first_success_idx[0]:
                    if my_t==0: # only one thread does the metropolis criteria check and atomic min for each swap attempt
                        # metropolis criteria 
                        if xoroshiro128p_uniform_float32(rng_states, thread_id) <= libdevice.expf(-s_delta[my_local_id] / temperature): 
                            cuda.atomic.min(first_success_idx, 0, thread_id) 
                grid.sync() 

                #change p types and apply swap
                winner = first_success_idx[0]
                if winner < attempts_this_round and thread_id == winner and my_t==0:
                    success_counter[0] += 1 
                    #print(winner, i, j, delta_energy, ptype[i], ptype[j]) 
                    ptype[i], ptype[j] = ptype[j], ptype[i] # Should also change masses, velocities, etc. if they are type-dependent.
                    #print(winner, i, j, delta_energy, ptype[i], ptype[j]) 
                grid.sync()		

                #increase the number of attempts used by the lowest thread id with an accepted swap
                if thread_id == 0 and my_t == 0: 
                    if winner < attempts_this_round: 
                        attempts_counter[0] += winner + 1 
                    else: 
                        attempts_counter[0] += attempts_this_round 
                    #print(attempts_counter[0])
                grid.sync() 
         
            return

        if gridsync:
            def kernel(grid, vectors, scalars, ptype, r_im, sim_box, step, swapper_params):
                if step%steps_between_swap == 0:
                    swap_attempts, temperature, rng_states, nblist, params, first_success_idx, attempts_counter, success_counter = swapper_params
                                                    
                    thread_swap_attempts(vectors[r_id], ptype, sim_box, nblist, params, 
                                         rng_states, temperature, first_success_idx, attempts_counter, swap_attempts, success_counter)
            
                return
            return cuda.jit(device=gridsync)(kernel)
        else:
            def kernel(grid, vectors, scalars, ptype, r_im, sim_box, step, params):
                pass
            return kernel
