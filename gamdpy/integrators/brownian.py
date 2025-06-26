
import numpy as np
import numba
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
import gamdpy as gp
from .integrator import Integrator

class Brownian(Integrator):
    r""" Brownian dynamics.

    Implementation of Brownain dynamics (overdamped Langevin) using the algorithm GJ-0 described in Ref. [Grønbech2019]_ (Eq. (67)).
    This stochastic integrator keeps the system at a constant temperature,
    via the equations of motion

    .. math::
        \alpha \dot x = f + \beta

    where :math:`f` is the force from a conservative field,
    :math:`\alpha` is a friction coefficient, and :math:`\beta` is uncorrelated Gauss distributed noise:
    :math:`\langle \beta(t)\rangle=0` and :math:`\langle \beta(t)\beta(t')\rangle=2\alpha T\delta(t-t')`
    where :math:`T` is the temperature of a solvent.
    In this implimentatio, the friction coefficient :math:`\alpha` is given by

    .. math::
        \alpha = m/\tau

    where :math:`m` is the mass of a given particle, and :math:`\tau` is a characteristic time.
    Thus, the particle mass, :math:`m`, can be used to set individual particle couplings with the solvent.

    The equation of motion is discretized using a time step :math:`dt` by

    .. math::
        x(t+dt) = x(t) + \frac{1}{\alpha}f(t)dt+\frac{1}{2\alpha}[\beta(t)+\beta(t+dt)]

    The "current" velocity stored is defined as

    .. math::
        v(t) = [x(t-dt) - x(t)]/dt

    Parameters
    ----------
    temperature : float or function
        Temperature of the implicit solvent. If a function, it must take a single argument, time, and return a float.

    tau : float
        Collision time of implicit solvent.

    dt : float
        a time step for the integration.

    seed : int, optional
        Seed for the pseudo random :math:`\beta`'s.

    References
    ----------
    .. [Grønbech2019] Niels Grønbech-Jensen,
       "Complete set of stochastic Verlet-type thermostats for correct Langevin simulations",
       Mol. Phys. 118, e1662506 (2019)
       https://doi.org/10.1080/00268976.2019.1662506

    Examples
    --------
    >>> import gamdpy as gp
    >>> integrator = gp.Brownian(temperature=1.0, tau=0.1, dt=0.005)
    """
  
    def __init__(self, temperature, tau: float, dt: float, seed = 0) -> None:
        self.temperature = temperature
        self.tau = tau
        self.dt = dt
        self.seed = seed

    def get_params(self, configuration: gp.Configuration, interactions_params: tuple, verbose=False) -> tuple:
        dt = np.float32(self.dt)
        tau = np.float32(self.tau)
        rng_states = create_xoroshiro128p_states(configuration.N, seed=self.seed)
        old_beta = np.zeros((configuration.N, configuration.D), dtype=np.float32)
        d_old_beta = cuda.to_device(old_beta)
        return (dt, tau, rng_states, d_old_beta) # Needs to be compatible with unpacking in
                                                   # step() below

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], interactions_kernel, verbose=False):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        # Convert temperature to a function if isn't already (better be a number then...)
        if callable(self.temperature):
            temperature_function = self.temperature
        else:
            temperature_function = gp.make_function_constant(value=float(self.temperature))

        if verbose:
            print(f'Generating Brownian integrator for {num_part} particles in {D} dimensions:')
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
            """ Make one timestep using the GJ-0 algorithm, Eq. (67) in Grønbech2019.  """
            dt, tau, rng_states, old_beta = integrator_params
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

                alpha = my_m / tau
                # if global_id == 0 and my_t == 0:
                #     print(alpha)
                for k in range(D):
                    random_number = xoroshiro128p_normal_float32(rng_states, global_id)
                    beta = math.sqrt(numba.float32(2.0) * alpha * temperature * dt) * random_number
                    if compute_k:
                        my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  # Half step kinetic energy
                    if compute_fsq:
                        my_fsq += my_f[k] * my_f[k]
                    my_v[k] = my_f[k] / alpha + np.float32(0.5)*(beta+old_beta[global_id,k]) / (dt*alpha)
                    my_r[k] += (my_f[k]*dt+np.float32(0.5)*(beta+old_beta[global_id,k])) / alpha
                    old_beta[global_id, k] = beta  # Store beta for the next step

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
