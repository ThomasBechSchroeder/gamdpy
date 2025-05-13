import numpy as np
import numba
from numba import cuda
import gamdpy as gp
from .integrator import Integrator

class NVT(Integrator):
    """ The leapfrog algorithm with a Nose-Hoover thermostat

    Integrator keeping N (number of particles), V (volume), and T (temperature) constant,
    using the leapfrog algorithm and a Nose-Hoover thermostat.

    Parameters
    ----------
    temperature : float or function
        The temperature to keep the system at. If a function, it should take time as argument.

    tau : float
        The relaxation time of the thermostat.

    dt : float
        The time step of the integration.

    """

    def __init__(self, temperature, tau: float, dt: float) -> None: 
        self.temperature = temperature
        self.tau = tau 
        self.dt = dt
        self.thermostat_state = np.zeros(2, dtype=np.float32)           # Right time to allocate and copy to device?
        self.d_thermostat_state = cuda.to_device(self.thermostat_state) # - or in get_params

    def get_params(self, configuration: gp.Configuration, interactions_params: tuple, verbose=False) -> tuple:
        dt = np.float32(self.dt)
        omega2 = np.float32(4.0 * np.pi * np.pi / self.tau / self.tau)
        degrees = configuration.N * configuration.D - configuration.D
        return (dt, omega2, degrees, self.d_thermostat_state)   # Needs to be compatible with unpacking in
                                                                # step() and update_thermostat_state() below.

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict, interactions_kernel, verbose=False):

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
            print(f'Generating NVT kernel for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        m_id = configuration.sid['m']
        compute_k = compute_flags['K']
        compute_fsq = compute_flags['Fsq']
        if compute_k:
            k_id = configuration.sid['K']
        if compute_fsq:
            fsq_id = configuration.sid['Fsq']

        # JIT compile functions to be compiled into kernel
        temperature_function = numba.njit(temperature_function)
        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())



        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            """ Make one NVT timestep using Leap-frog
                Kernel configuration: [num_blocks, (pb, tp)]
            """

            # Unpack parameters. MUST be compatible with get_params() above
            dt, omega2, degrees, thermostat_state = integrator_params  

            factor = np.float32(0.5) * thermostat_state[0] * dt
            plus = np.float32(1.) / (np.float32(1.) + factor)  # Possibly change to exp(...)
            minus = np.float32(1.) - factor  # Possibly change to exp(...)

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]
                my_k = numba.float32(0.0)    # Kinetic energy
                if compute_fsq:
                    my_fsq = numba.float32(0.0)  # force squared

                for k in range(D):
                    if compute_fsq:
                        my_fsq += my_f[k] * my_f[k]
                    my_v[k] = plus * (minus * my_v[k] + my_f[k] / my_m * dt)
                    my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]
                    my_r[k] += my_v[k] * dt
                    
                apply_PBC(my_r, r_im[global_id], sim_box)

                cuda.atomic.add(thermostat_state, 1, my_k)  # Probably slow? Not really!
                if compute_k:
                    scalars[global_id][k_id] = my_k
                if compute_fsq:
                    scalars[global_id][fsq_id] = my_fsq
            return

        def update_thermostat_state(integrator_params, time):
            # Unpack parameters. MUST be compatible with get_params() above
            dt, omega2, degrees, thermostat_state = integrator_params 

            global_id, my_t = cuda.grid(2)
            if global_id == 0 and my_t == 0:
                target_temperature = temperature_function(time)

                ke_deviation = np.float32(2.0) * thermostat_state[1] / (degrees * target_temperature) - np.float32(1.0)
                thermostat_state[0] += dt * omega2 * ke_deviation
                thermostat_state[1] = np.float32(0.)
            return

        step = cuda.jit(device=gridsync)(step)
        update_thermostat_state = cuda.jit(device=gridsync)(update_thermostat_state)

        if gridsync: # construct and return device function
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                step(  grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                grid.sync()
                update_thermostat_state(integrator_params, time)
                return
            return cuda.jit(device=gridsync)(kernel)
        else: # return python function, which makes kernel-calls
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                step[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                update_thermostat_state[1, (1, 1)](integrator_params, time)
                return
            return kernel
