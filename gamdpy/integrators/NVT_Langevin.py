
import numpy as np
import numba
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
import gamdpy as gp
from .integrator import Integrator


class NVT_Langevin(Integrator):
    """ NVT Langevin Leap-frog integrator

    The langevin thermostat is a stochastic thermostat that keeps the system at a constant temperature:

    .. math::
        m a = f - \\alpha v + \\beta

    where :math:`a` is the acceleration, :math:`f` is the force, :math:`v` is the velocity, :math:`m` is the mass,
    :math:`\\alpha` is the friction coefficient, and :math:`\\beta` is a random number drawn from a normal distribution.
    The implementation uses the leap-frog algorithm described in reference https://arxiv.org/pdf/1303.7011.pdf

    Parameters
    ----------

    temperature : float or function
        Temperature of the thermostat. If a function, it must take a single argument, time, and return a float.

    alpha : float
        Friction coefficient of the thermostat.

    dt : float
        Time step for the integration.

    """
  
    def __init__(self, temperature, alpha: float, dt: float, seed: int) -> None:
        self.temperature = temperature
        self.alpha = alpha 
        self.dt = dt
        self.seed = seed

    def get_params(self, configuration: gp.Configuration, interactions_params: tuple, verbose=False) -> tuple:
        dt = np.float32(self.dt)
        alpha = np.float32(self.alpha)
        rng_states = create_xoroshiro128p_states(configuration.N, seed=self.seed)
        old_beta = np.zeros((configuration.N, configuration.D), dtype=np.float32)
        d_old_beta = cuda.to_device(old_beta)
        return (dt, alpha, rng_states, d_old_beta) # Needs to be compatible with unpacking in
                                                   # step() below

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], interactions_kernel, verbose=False):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        # Convert temperature to a function if isn't allready (better be a number then...)
        if callable(self.temperature):
            temperature_function = self.temperature
        else:
            temperature_function = gp.make_function_constant(value=float(self.temperature))

        if verbose:
            print(f'Generating NVT langevin integrator for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')
        
        # Unpack indices for vectors and scalars to be compiled into kernel
        compute_k = compute_flags['K']
        compute_fsq = compute_flags['Fsq']
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        m_id = configuration.sid['m']
        if compute_k:
            k_id = configuration.sid['K']
        if compute_fsq:
            fsq_id = configuration.sid['Fsq']


        # JIT compile functions to be compiled into kernel
        temperature_function = numba.njit(temperature_function)
        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())

    
        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
            """ Make one NVT Langevin timestep using Leap-frog
                Kernel configuration: [num_blocks, (pb, tp)]
                REF: https://arxiv.org/pdf/1303.7011.pdf
            """

            dt, alpha, rng_states, old_beta = integrator_params
            temperature = temperature_function(time)

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]
                if compute_k:
                    my_k = numba.float32(0.0)  # Kinetic energy
                if compute_fsq:
                    my_fsq = numba.float32(0.0)  # force squared energy
                

                for k in range(D):
                    # REF: https://arxiv.org/pdf/1303.7011.pdf sec. 2.C.
                    random_number = xoroshiro128p_normal_float32(rng_states, global_id)
                    beta = math.sqrt(numba.float32(2.0) * alpha * temperature * dt) * random_number
                    # Eq. (16) in https://arxiv.org/pdf/1303.7011.pdf
                    numerator =   numba.float32(2.0)*my_m - alpha * dt
                    denominator = numba.float32(2.0)*my_m + alpha * dt
                    a = numerator / denominator
                    b_over_m = numba.float32(2.0) / denominator
                    if compute_k:
                        my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  # Half step kinetic energy
                    if compute_fsq:
                        my_fsq += my_f[k] * my_f[k]
                    my_v[k] = a * my_v[k] + b_over_m * my_f[k] * dt + b_over_m * np.float32(0.5)*(beta+old_beta[global_id,k])
                    old_beta[global_id,k] = beta # Store beta for next step
                    my_r[k] += my_v[k] * dt

                apply_PBC(my_r, r_im[global_id], sim_box)
                if compute_k:
                    scalars[global_id][k_id] = my_k
                if compute_fsq:
                    scalars[global_id][fsq_id] = my_fsq
            return

        step = cuda.jit(device=gridsync)(step)

        if gridsync:
            return step  # return device function
        else:
            return step[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters 
