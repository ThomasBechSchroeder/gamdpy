import numpy as np
# LC: check https://orbit.dtu.dk/en/publications/cudarray-cuda-based-numpy
import numba
from numba import cuda
from gamdpy import Configuration, Orthorhombic
from gamdpy.misc.make_function import make_function_constant
from .integrator import Integrator

class NPT_Atomic(Integrator):
    """ Constant NPT integrator for atomic systems.

    Integrator that keeps the number of particles (:math:`N`), pressure (:math:`P`),
    and temperature (:math:`T`) constant, using a Leap-Frog implementation
    of the Nosé–Hoover thermostat by Martyna–Tuckerman–Klein presented in Ref. [Martyna1996]_.
    (Note that the thermostat and barostat states in this implementation are defined as
    :math:`p_\\xi/Q` and :math:`p_\\epsilon/W`, see Eq. 2.9 in the reference.).

    The thermal and barometric coupling parameters are defined via two time-scales.

    Parameters
    ----------
    temperature : float or function
        Target temperature

    tau : float
        Thermostat relaxation time

    pressure : float or function
        Target pressure

    tau_p : float
        Barostat relaxation time

    dt : float
        a time step for the integration.

    Raises
    ------
    TypeError
        If the simulation box is not :class:`~gamdpy.Orthorhombic`.

    ValueError
        If the spatial dimension of the simulation box is not :math:`D=3`.

    References
    ----------
    .. [Martyna1996] Glenn J. Martyna, Mark E. Tuckerman, Douglas J. Tobias, and Michael L. Klein,
       "Explicit Reversible Integrators for Extended Systems Dynamics",
       Mol. Phys. 87, 1117–57 (1996)
       https://doi.org/10.1080/00268979600100761
    """

    def __init__(self, temperature, tau: float, pressure, tau_p : float, dt: float) -> None: 
        self.temperature = temperature
        self.tau = tau 
        self.pressure = pressure
        self.tau_p = tau_p 
        self.dt = dt
        self.thermostat_state = np.zeros(2, dtype=np.float32)          
        self.barostat_state = np.zeros(3, dtype=np.float32)                         # NOTE: array is (barostat_state, virial, volume) 

    def get_params(self, configuration: Configuration, interactions_params: tuple, verbose=False) -> tuple:
        dt = np.float32(self.dt)
        degrees  = configuration.N * configuration.D - configuration.D                        # number of degrees of freedom 
        factor = np.float32(1./(4*np.pi*np.pi))
        mass_t = np.float32((degrees-1)*configuration.D * factor * self.tau   * self.tau  )   # This quantity is the thermostat mass expect for a factor of temperature
        mass_p = np.float32((degrees-1)*configuration.D * factor * self.tau_p * self.tau_p)   # the temperature is missing because at this stage could be a function
        self.barostat_state[2] = configuration.get_volume()                                   # Copy starting volume (can be avoided)
        # Copy state variables to a device
        self.d_barostat_state   = numba.cuda.to_device(self.barostat_state)
        self.d_thermostat_state = numba.cuda.to_device(self.thermostat_state)
        return (dt, mass_t, mass_p, degrees, self.d_thermostat_state, self.d_barostat_state)   # Needs to be compatible with unpacking in
                                                                                               # step() and update_thermostat_state() below.

    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict, interactions_kernel, verbose=False):

        # This integrator is designed for an Orthorhombic simulation box
        if not isinstance(configuration.simbox, Orthorhombic):
            raise TypeError(f"The NPT Atomic integrator expected Orthorhombic simulation box but got {type(configuration.simbox)}.")

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        # This implementation assumes D=3
        if not D == 3:
            raise ValueError(f"This integrator expected a simulation box with D=3 but got {D}.")

        # Convert temperature to a function if isn't already (better be a number then...)
        if callable(self.temperature):
            temperature_function = self.temperature
        else:
            temperature_function = make_function_constant(value=float(self.temperature))
        # Convert pressure to a function if isn't already (better be a number then...)
        if callable(self.pressure):
            pressure_function = self.pressure
        else:
            pressure_function = make_function_constant(value=float(self.pressure))
    
        if verbose:
            print(f'Generating NPT kernel for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')

        # Unpack indices for vectors and scalars to be compiled into kernel
        compute_k = compute_flags['K']
        compute_fsq = compute_flags['Fsq']
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        m_id = configuration.sid['m']
        if not compute_flags['W'] or not configuration.compute_flags['W']:
            raise ValueError("The NPT_Atomic requires virial flag to be True in the configuration object.")
        else:
            w_id = configuration.sid['W']

        if compute_k:
            k_id = configuration.sid['K']
        if compute_fsq:
            fsq_id = configuration.sid['Fsq']

        # JIT compile functions to be compiled into kernel
        temperature_function = numba.njit(temperature_function)
        pressure_function = numba.njit(pressure_function)
        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())


        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):       # pragma: no cover
            """ Make one NPT timestep using Leap-frog
                Kernel configuration: [num_blocks, (pb, tp)]
            """
            
            # Unpack parameters. MUST be compatible with get_params() above
            dt, mass_t, mass_p, degrees, thermostat_state, barostat_state = integrator_params  

            # NOTE: thermostat_state and barostat_state have units of inverse time ([t]**-1)
            factor = np.float32(0.5) * dt * ((1+3./degrees)*barostat_state[0] + thermostat_state[0]) # D=3
            plus = np.float32(1.) / np.float32(1. + factor)  
            minus = np.float32(1. - factor)                    
            rfactor = barostat_state[0]*dt

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_w = scalars[global_id][w_id] # Get virial from scalars
                my_m = scalars[global_id][m_id]
                my_k = numba.float32(0.0)       # Kinetic energy
                if compute_fsq:
                    my_fsq = numba.float32(0.0)     # force squared

                for k in range(D):
                    if compute_fsq:
                        my_fsq += my_f[k] * my_f[k]
                    my_v[k] = plus * (minus * my_v[k] + my_f[k] / my_m * dt)
                    my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]       # This is kinetic energy at the half step
                    my_r[k] += my_v[k] * dt + rfactor*my_r[k]
                    
                apply_PBC(my_r, r_im[global_id], sim_box)

                cuda.atomic.add(thermostat_state, 1, my_k)   # Probably slow? Not really!
                cuda.atomic.add(barostat_state  , 1, my_w)
                if compute_k:
                    scalars[global_id][k_id] = my_k
                if compute_fsq:
                    scalars[global_id][fsq_id] = my_fsq
            return

        # This function update barostat (P) and thermostat (T) states
        def update_thermostat_barostat_state(vectors, sim_box, integrator_params, time):        # pragma: no cover
            # Unpack parameters. MUST be compatible with get_params() above
            dt, mass_t, mass_p, degrees, thermostat_state, barostat_state = integrator_params 

            global_id, my_t = cuda.grid(2)
            if global_id == 0 and my_t == 0:
                temperature  = 2*thermostat_state[1]/degrees
                scale_factor_3 = 1 + dt*D*barostat_state[0]     # assumes D=3
                mass_t = mass_t * temperature                   # fix masses including temperature factor
                mass_p = mass_p * temperature                   # thermostat/barostat masses are extensive quantities
                # Scale volume
                barostat_state[2] *= scale_factor_3
                # Scale simbox lenghts
                scale_factor  = scale_factor_3**(1./3)
                sim_box[0]   *= scale_factor
                sim_box[1]   *= scale_factor
                sim_box[2]   *= scale_factor
                # Update states
                target_temperature = temperature_function(time)
                target_pressure    = pressure_function(time)
                instant_pressure   = (temperature*num_part + barostat_state[1])/barostat_state[2] # PV = N*T + W
                # Note that thermostat_state[0] and barostat_state[0] are intensive quantities
                ke_deviation = (np.float32(2.0) * thermostat_state[1] + barostat_state[0]*barostat_state[0]*mass_p) / ((degrees+1)*target_temperature) - np.float32(1.0)
                p_deviation  = D * (barostat_state[2] * (instant_pressure - target_pressure) + np.float32(0.5)*thermostat_state[1]/degrees)  # D=3
                # Update states
                thermostat_state[0] += dt * ke_deviation * (degrees+1)*target_temperature / mass_t
                barostat_state[0]   += dt * (p_deviation / mass_p + thermostat_state[0]*barostat_state[0])
                # Reset tmp variables, these variables are used to pass values between step() and here
                thermostat_state[1] = np.float32(0.)
                barostat_state[1]   = np.float32(0.)
            return

        # Scale the simulation box to the new density
        def scale_box(vectors, sim_box, integrator_params):     # pragma: no cover
            # Unpack parameters. MUST be compatible with get_params() above
            dt, mass_t, mass_p, degrees, thermostat_state, barostat_state = integrator_params 

            scale_factor_3 = 1 + dt*D*barostat_state[0]     # assumes D=3
            scale_factor  = scale_factor_3**(1./3)

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                for k in range(D): 
                    vectors[r_id][global_id][k] = scale_factor*vectors[r_id][global_id][k] 
            return

        step = numba.cuda.jit(device=gridsync)(step)
        update_thermostat_barostat_state = numba.cuda.jit(device=gridsync)(update_thermostat_barostat_state)
        scale_box = numba.cuda.jit(device=gridsync)(scale_box)

        if gridsync:        # pragma: no cover
            # construct and return device function
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                step(  grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                grid.sync()
                scale_box(vectors, sim_box, integrator_params)
                grid.sync()
                update_thermostat_barostat_state(vectors, sim_box, integrator_params, time)
                return
            return numba.cuda.jit(device=gridsync)(kernel)
        else:       # pragma: no cover
            # return python function, which makes kernel-calls
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                step[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                scale_box[num_blocks, (pb, 1)](vectors, sim_box, integrator_params)
                update_thermostat_barostat_state[1, (1, 1)](vectors, sim_box, integrator_params, time)
                return
            return kernel
