import numpy as np
import numba
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
import gamdpy as gp
from .integrator import Integrator

class NPT_Langevin(Integrator):
    r""" Constant NPT Langevin integrator with isotropic volume fluctuations.

    This integrator is a Leap-Frog implementation of the GJ-F Langevin equations of motion present
    in Ref. [Grønbech2014b]_. It is intended for an orthorhombic simulation box in any spatial dimention.

    Let :math:`f` be the conservative force field, :math:`\alpha` is a friction parameter, and
    :math:`\beta` be uncorrelated Gauss distributed noise:
    :math:`\langle \beta(t)\rangle=0` and :math:`\langle \beta(t)\beta(t')\rangle=2\alpha k_B T\delta(t-t')`
    where :math:`T` is the target temperature.
    The Langevin equation of motion for a given particle is

    .. math:: m\ddot x = f - \alpha \dot x + \beta

    For choosing the :math:`\alpha` parameters, it is instructive to note that a characteristic timescale is given by

    .. math:: \tau_T = m/\alpha

    The (isotropic) equations of motions for the volume of the simulation box (:math:`V`) can similarly be written as

    .. math:: Q\ddot V = f_P - \alpha_V \dot V + \beta_V

    Here, :math:`Q` is an inertial coefficient (a barostat mass), :math:`\alpha_V` is another friction parameter,
    :math:`\beta_V` is uncorrelated Gauss distributed noise, and :math:`f_P` is a "barostat force".
    Specifically, if :math:`W` is the (instantaneous) virial, then

    .. math:: f_P = W + \frac{Nk_B T}{V} - P

    where :math:`P` is the target pressure. To set the barstat parameters :math:`Q` and :math:`\alpha_V` it can be
    instructive to note that equations of motions for the volume resemble that of a damped harmonic oscillator.
    If :math:`K=-VdP/dV` is the bulk modulus of the system,
    then the "spring constant" of the oscillator is :math:`k=K/V`, then the natural frequency
    is :math:`\omega_0=\sqrt{k/Q}` and damping ratio is :math:`\zeta=\alpha_V/2\sqrt{Qk}`
    (:math:`\zeta<1` for underdamped oscillation).
    If a characteristic timescale is defined as :math:`\tau_V\equiv 1/\omega_0`, then

    .. math:: \alpha_V = 2 K \tau_v/V

    .. math:: Q = K (\zeta \tau_v)^2 / V

    Parameters
    ----------
    temperature : float or function
        Target temperature, :math:`T`

    pressure : float or function
        Target pressure, :math:`P`

    alpha : float
        Friction coefficient of the thermostat, :math:`\alpha`

    alpha_barostat : float
        Friction coefficient of the barostat, :math:`\alpha_V`

    mass_barostat : float
        Inertial coefficient (barostat mass), :math:`Q`

    dt : float
        a Time step for the Leap-Frog discretization

    volume_velocity : float
        Initial velocity of volume

    seed : int
        seed for the (pseudo) random noise

    Examples
    --------

    Example of setting parameters for the NPT langevin barostat (see above) in "Lennard-Jones like" reduced units.

    >>> tau_T = 2.0
    >>> tau_V = 8.0
    >>> zeta = 0.2
    >>> K = 100   # Replace with an estimate of the bulk modulus
    >>> V = 1000  # Replace with an estimate of the average volume
    >>> integrator = gp.NPT_Langevin(
    ...    temperature=2.0,
    ...    pressure=1.0,
    ...    alpha=1/tau_T,
    ...    alpha_barostat=2*K/tau_V/V,
    ...    mass_barostat=K*(zeta*tau_V)**2/V,
    ...    dt=0.004)

    Notes
    -----

    .. [Grønbech2014b] Niels Grønbech-Jensen and Oded Farago,
       "Constant pressure and temperature discrete-time Langevin molecular dynamics",
       J. Chem. Phys. 141, 194108 (2014)
       https://doi.org/10.1063/1.4901303

    """

    def __init__(self, temperature, pressure, alpha: float, alpha_barostat: float, mass_barostat: float, dt: float, volume_velocity = 0.0, seed = 0) -> None:
        self.temperature = temperature
        self.pressure = pressure
        self.alpha = alpha 
        self.alpha_barostat = alpha_barostat
        self.mass_barostat = mass_barostat
        self.dt = dt
        self.volume_velocity = volume_velocity
        self.seed = seed

    def get_params(self, configuration: gp.Configuration, interactions_params: tuple, verbose=False) -> tuple:
        dt = np.float32(self.dt)
        alpha = np.float32(self.alpha)
        alpha_baro = np.float32(self.alpha_barostat)
        mass_baro = np.float32(self.mass_barostat)
        rng_states = create_xoroshiro128p_states(configuration.N+1, seed=self.seed) # +1 for barostat dynamics 
        barostat_state = np.array([1.0, self.volume_velocity], dtype=np.float64)       # [0] = new_vol / old_vol , [1] = vol velocity
        d_barostat_state = cuda.to_device(barostat_state)
        barostatVirial = np.array([0.0], dtype=np.float32)
        d_barostatVirial = cuda.to_device(barostatVirial)
        d_length_ratio = cuda.to_device(np.ones(configuration.D, dtype=np.float32))
        return (dt, alpha, alpha_baro, mass_baro, # Needs to be compatible with unpacking in step() below
                rng_states, d_barostat_state, d_barostatVirial, d_length_ratio)
    
    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags:dict[str,bool], interactions_kernel, verbose=False):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        # Convert temperature and pressure to a function if needed
        if callable(self.temperature):
            temperature_function = self.temperature
        else:
            temperature_function = gp.make_function_constant(value=float(self.temperature))
        if callable(self.pressure):
            pressure_function = self.pressure
        else:
            pressure_function = gp.make_function_constant(value=float(self.pressure))

        if verbose:
            print(f'Generating NPT langevin integrator for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')
        
        # Unpack indices for vectors and scalars to be compiled into kernel
        compute_k = compute_flags['K']
        compute_fsq = compute_flags['Fsq']
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        m_id = configuration.sid['m']
        if not compute_flags['W']:
            raise ValueError("NPT_Langevin requires virial")
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
        # get_volume = numba.njit(configuration.simbox.get_volume)


        def copyParticleVirial(scalars, integrator_params):
            dt, alpha, alpha_baro, mass_baro, rng_states, barostat_state, barostatVirial, length_ratio  = integrator_params
            global_id, my_t = cuda.grid(2)
            my_w = scalars[global_id][w_id]
            
            if global_id < num_part and my_t == 0:
                cuda.atomic.add(barostatVirial, 0, my_w)  # factor of 6 already accounted for using virial_factor and virial_factor_NIII
            
            return
    
        def update_barostat_state(sim_box, integrator_params, time):
            dt, alpha, alpha_baro, mass_baro, rng_states, barostat_state, barostatVirial, length_ratio = integrator_params
            temperature = temperature_function(time)
            pressure = pressure_function(time) 

            global_id, my_t = cuda.grid(2)
            if global_id == 0 and my_t == 0:

                #Copy barostat_state into current_barostat_state using a local aaray
                current_barostat_state = cuda.local.array(2, numba.float64)
                current_barostat_state[0] = barostat_state[0]
                current_barostat_state[1] = barostat_state[1]

                volume = numba.float32(1.0)
                for k in range(D):  # Use simbox function
                    volume *= sim_box[k]

                targetConfPressure = pressure - temperature * num_part / volume
                barostatForce = barostatVirial[0] / volume - targetConfPressure
                # print('press', barostatVirial[0] / volume+temperature * num_part / volume)


                random_number = xoroshiro128p_normal_float32(rng_states, 0)          # 0th random number state is reserved for barostat
                barostatRandomForce = math.sqrt(numba.float32(2.0) * alpha_baro * temperature * dt) * random_number 
            
                current_volume_velocity = current_barostat_state[1]
                inv_baro_mass = numba.float32(1.0) / mass_baro
                scaled_dt = numba.float32(0.5) * dt * alpha_baro * inv_baro_mass
                b_tilde = numba.float64(1.0) / (numba.float64(1.0) + scaled_dt)
                a_tilde = b_tilde * (numba.float64(1.0) - scaled_dt)

                new_volume_vel = a_tilde * current_volume_velocity + b_tilde * inv_baro_mass * (barostatForce * dt + barostatRandomForce)
                new_volume = volume + dt * new_volume_vel

                # Update the barostat state
                barostat_state[0] = new_volume / volume 
                barostat_state[1] = new_volume_vel
                
                vol_scale_factor = barostat_state[0]
                scale_factor = math.pow(vol_scale_factor, numba.float64(1.0) / numba.float64(D))  # Changes needed for aniso

                # update box length, assume Orthorhombic
                for k in range(D):
                    sim_box[k] = sim_box[k] * scale_factor
                
                # update length_ratio using cuda loop
                for k in range(D):
                    length_ratio[k] = scale_factor
                
                barostatVirial[0] = numba.float32(0.0)
            return

    
        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            """ Make one NPT timestep using Leap-frog
                Kernel configuration: [num_blocks, (pb, 1)]
                REF: https://arxiv.org/pdf/1303.7011.pdf
            """

            dt, alpha, alpha_baro, mass_baro, rng_states, barostat_state, barostatVirial, length_ratio = integrator_params
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
                    random_number = xoroshiro128p_normal_float32(rng_states, global_id + 1)  # +1 to avoid using the same random number state as the barostat
                    beta = math.sqrt(numba.float32(2.0) * alpha * temperature * dt) * random_number
                    
                    scaled_dt = numba.float32(0.5) * dt * alpha * numba.float32(1.0)/my_m
                    prm_b = numba.float64(1.0) / (numba.float64(1.0) + scaled_dt)
                    prm_a = prm_b * (numba.float64(1.0) - scaled_dt)

                    if compute_k:
                        my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  #  ke before
                    if compute_fsq:
                        my_fsq += my_f[k] * my_f[k]

                    my_v[k] = prm_a * my_v[k] + prm_b * (numba.float32(1.0)/my_m) * (my_f[k] * dt + beta)
                    if compute_k:
                        my_k += numba.float32(0.5) * my_m * my_v[k] * my_v[k]  #  ke after
                    
                    L_factor = 2.*length_ratio[k] / (1. + length_ratio[k]) 
                    my_r[k] = length_ratio[k] * my_r[k] + L_factor * my_v[k] * dt             
                     
                apply_PBC(my_r, r_im[global_id], sim_box)

                if compute_k:
                    scalars[global_id][k_id] = numba.float32(0.5) * my_k
                if compute_fsq:
                    scalars[global_id][fsq_id] = my_fsq

            return
            
        copyParticleVirial = cuda.jit(device=gridsync)(copyParticleVirial)
        update_barostat_state = cuda.jit(device=gridsync)(update_barostat_state)
        step = cuda.jit(device=gridsync)(step)

        if gridsync:                                              

            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                copyParticleVirial(scalars, integrator_params)
                grid.sync()
                update_barostat_state(sim_box, integrator_params, time)
                grid.sync()
                step(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                return

            return cuda.jit(device=gridsync)(kernel)

        else:

            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                copyParticleVirial[num_blocks, (pb, 1)](scalars, integrator_params)
                update_barostat_state[1, (1, 1)](sim_box, integrator_params, time)            
                step[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                return

        return kernel
