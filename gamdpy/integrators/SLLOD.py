import numpy as np
import numba
import gamdpy as gp
from numba import cuda
import math
from .integrator import Integrator



class SLLOD(Integrator):
    """ The SLLOD integrator

    Shear an atomic system in the xy-plane using the SLLOD equations [Edberg1986]_
    implimented with the operator splitting algorithm, see [Pan2005]_.
    This integrator needs a :class:`~gamdpy.LeesEdwards` simulation box.

    Parameters
    ----------

    shear_rate : float
        The shear rate of the system.

    dt : float
        The time step of the simulation.

    Raises
    ------
    TypeError
        If the simulation box is not :class:`~gamdpy.LeesEdwards` simulation box.

    References
    ----------
    .. [Edberg1986] Roger Edberg, Denis J. Evans and G. P. Morriss
    "Constrained molecular dynamics: Simulations of liquid alkanes with a new algorithm"
    J. Chem. Phys. 84, 6933–6939 (1986)
    https://doi.org/10.1063/1.450613

    .. [Pan2005] Guoai Pan, James F. Ely, Clare McCabe and Dennis J. Isbister
    "Operator splitting algorithm for isokinetic SLLOD molecular dynamics"
    J. Chem. Phys. 122, 094114 (2005)
    https://doi.org/10.1063/1.1858861

    Examples
    --------

    An example of how to set up a Lees Edwards simulation box and a SLLOD integrator in a Lennard-Jones like system.

    >>> configuration = gp.Configuration(D=3, compute_flags={'stresses': True})
    >>> configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
    >>> configuration['m'] = 1.0
    >>> configuration.randomize_velocities(temperature=0.7)
    >>> configuration.simbox = gp.LeesEdwards(configuration.D, configuration.simbox.get_lengths())
    >>> configuration.set_kinetic_temperature(temperature=0.7, ndofs=configuration.N*3-4)  # Note that we need to subtract 4
    >>> integrator = gp.SLLOD(shear_rate=0.02, dt=0.01)

    """
    def __init__(self, shear_rate, dt):
        self.shear_rate = shear_rate
        self.dt = dt
  
    def get_params(self, configuration, interactions_params, verbose=False):
        dt = np.float32(self.dt)
        #sr = np.float32(self.shear_rate)

        # three 'groups' of three sum variables
        self.thermostat_sums = np.zeros(9, dtype=np.float32)

        # before first time-step, group 0, ie elements 0, 1, 2 needs to be initialized with sum_pxpy, sum_pypy, sum_p2
        D, num_part = configuration.D, configuration.N

        v = configuration['v']
        m = configuration['m']

        self.thermostat_sums[0] =  np.sum(v[:,0] * v[:,1] * m)
        self.thermostat_sums[1] =  np.sum(v[:,1] * v[:,1] * m)
        self.thermostat_sums[2] =  np.sum(np.sum(v**2, axis=1)*m)

        self.d_thermostat_sums = cuda.to_device(self.thermostat_sums)

        return (dt, self.d_thermostat_sums)

    def get_kernel(self, configuration, compute_plan, compute_flags, interactions_kernel, verbose=False):

        # Expects a Lees Edwards type simulation box
        if not isinstance(configuration.simbox, gp.LeesEdwards):
            raise ValueError(f'The SLLOD integrator requires a Lees-Edwards simulation box, but got {type(configuration.simbox)}')

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        if callable(self.shear_rate):
            shear_rate_function = self.shear_rate
        else:
            shear_rate_function = gp.make_function_constant(value=float(self.shear_rate))

        if verbose:
            print(f'Generating SLLOD kernel for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')

        # Unpack indices for vectors and scalars

        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        compute_k = compute_flags['K']
        compute_fsq = compute_flags['Fsq']
        m_id = configuration.sid['m']
        if compute_k:
            k_id = configuration.sid['K']
        if compute_fsq:
            fsq_id = configuration.sid['Fsq']


        # JIT compile functions to be compiled into kernel
        shear_rate_function = numba.njit(shear_rate_function)
        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())
        update_box_shift = numba.njit(configuration.simbox.get_update_box_shift())

        def call_update_box_shift(sim_box, integrator_params, time):                              # pragma: no cover
            dt, thermostat_sums = integrator_params
            sr = shear_rate_function(time)
            global_id, my_t = cuda.grid(2)
            if global_id == 0 and my_t == 0:
                delta_shift = sim_box[1] * sr * dt
                update_box_shift(sim_box, delta_shift)


        def integrate_sllod_b1(grid, vectors, scalars, integrator_params, time):            # pragma: no cover
            dt, thermostat_sums = integrator_params
            sr = shear_rate_function(time)

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]

                # read sums from group 0
                sum_pxpy = thermostat_sums[0]
                sum_pypy = thermostat_sums[1]
                sum_p2 = thermostat_sums[2]
                # compute coefficients
                c1 = sr*sum_pxpy / sum_p2
                c2 = sr*sr * sum_pypy / sum_p2
                # g-factor for a half time-step
                g_factor = 1./math.sqrt(1. - (c1*dt  - 0.25 * c2 * dt**2))  # double precision

                # update velocity - multiply by g_factor in double precision
                my_v[0] = numba.float32(g_factor * (my_v[0] - 0.5*sr*dt*my_v[1]))
                for k in range(1, D):
                    my_v[k] = numba.float32(g_factor * my_v[k])

                # add to sums in group 1 needed for step B2
                my_p2 = numba.float32(0.)
                my_fp = numba.float32(0.)
                my_f2 = numba.float32(0.)
                for k in range(D):
                    my_p2 += my_v[k] * my_v[k] * my_m
                    my_fp += my_f[k] * my_v[k]
                    my_f2 += my_f[k] * my_f[k] / my_m
                cuda.atomic.add(thermostat_sums, 3, my_p2)
                cuda.atomic.add(thermostat_sums, 4, my_fp)
                cuda.atomic.add(thermostat_sums, 5, my_f2)

            # and reset group 2 sums to zero
            if global_id == 0 and my_t == 0:
                thermostat_sums[6] = 0.
                thermostat_sums[7] = 0.
                thermostat_sums[8] = 0.


        def integrate_sllod_b2(grid, vectors, scalars, integrator_params, time):            # pragma: no cover
            dt, thermostat_sums = integrator_params
            sr = shear_rate_function(time)
            
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]
                if compute_fsq:
                    my_fsq = numba.float32(0.0)  # force squared
                # read sums from group 1
                sum_p2 = thermostat_sums[3]
                sum_fp = thermostat_sums[4]
                sum_f2 = thermostat_sums[5]
                
                # compute coefficients
                alpha = sum_fp / sum_p2
                beta = math.sqrt(sum_f2 / sum_p2)
                h = (alpha + beta) / (alpha - beta)
                e = math.exp(-beta * dt)
                one = numba.float32(1.)
                integrate_coefficient1 = (one - h) / (e - h/e)
                integrate_coefficient2 = (one + h - e - h/e)/((one-h)*beta)
                # update velocity
                
                for k in range(D):
                    if compute_fsq:
                        my_fsq += my_f[k] * my_f[k]
                    my_v[k] = integrate_coefficient1 * (my_v[k] + integrate_coefficient2 * my_f[k] / my_m)

                # add to sums in group 2
                my_pxpy = my_v[0] * my_v[1] * my_m
                my_pypy = my_v[1] * my_v[1] * my_m
                my_p2 = numba.float32(0.)
                for k in range(D):
                    my_p2 += my_v[k]**2
                my_p2 *= my_m
                
                cuda.atomic.add(thermostat_sums, 6, my_pxpy)
                cuda.atomic.add(thermostat_sums, 7, my_pypy)
                cuda.atomic.add(thermostat_sums, 8, my_p2)
                
                if compute_fsq:
                    scalars[global_id][fsq_id] = my_fsq

            # and reset group 0 sums to zero
            if global_id == 0 and my_t == 0:
                thermostat_sums[0] = numba.float32(0.)
                thermostat_sums[1] = numba.float32(0.)
                thermostat_sums[2] = numba.float32(0.)


        def integrate_sllod_a_b1(grid, vectors, scalars, r_im, sim_box, integrator_params, time):   # pragma: no cover
            dt, thermostat_sums = integrator_params
            sr = shear_rate_function(time)

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_m = scalars[global_id][m_id]
            
                # read sums from group 2
                sum_pxpy = thermostat_sums[6]
                sum_pypy = thermostat_sums[7]
                sum_p2 = thermostat_sums[8]
                # compute coefficients
                c1 = sr*sum_pxpy / sum_p2
                c2 = sr*sr * sum_pypy / sum_p2
                # g-factor for a half time-step
                g_factor = 1./math.sqrt(1. - (c1*dt  - 0.25 * c2 * dt**2))  # double precision

                # update velocity - multiply by g_factor in double precision
                my_v[0] = numba.float32(g_factor * (my_v[0] - 0.5*sr*dt*my_v[1]))
                for k in range(1, D):
                    my_v[k] = numba.float32(g_factor * my_v[k])

                # update position and apply boundary conditions
                my_r[0] += sr*dt*my_r[1]
                for k in range(D):
                    my_r[k] += my_v[k] * dt

                apply_PBC(my_r, r_im[global_id], sim_box)
                
                # add to sums in group 0 for next time integrate_B1 is called (at the next time step)
                my_pxpy = my_v[0] * my_v[1] * my_m
                my_pypy = my_v[1] * my_v[1] * my_m
                my_p2 = numba.float32(0.)
                for k in range(D):
                    my_p2 += my_v[k]**2
                my_p2 *= my_m
                
                cuda.atomic.add(thermostat_sums, 0, my_pxpy)
                cuda.atomic.add(thermostat_sums, 1, my_pypy)
                cuda.atomic.add(thermostat_sums, 2, my_p2)
                # store ke of this particle
                if compute_k:
                    scalars[global_id][k_id] = numba.float32(0.5) * my_p2

            # and reset group 1 sums to zero
            if global_id == 0 and my_t == 0:
                thermostat_sums[3] = numba.float32(0.)
                thermostat_sums[4] = numba.float32(0.)
                thermostat_sums[5] = numba.float32(0.)
        

                
        call_update_box_shift = cuda.jit(call_update_box_shift)
        integrate_sllod_b1 = cuda.jit(device=gridsync)(integrate_sllod_b1)
        integrate_sllod_b2 = cuda.jit(device=gridsync)(integrate_sllod_b2)
        integrate_sllod_a_b1 = cuda.jit(device=gridsync)(integrate_sllod_a_b1)


        if gridsync:    # pragma: no cover
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                integrate_sllod_b1(grid, vectors, scalars, integrator_params, time)
                grid.sync()
                integrate_sllod_b2(grid, vectors, scalars, integrator_params, time)
                grid.sync()
                call_update_box_shift(sim_box, integrator_params, time)
                grid.sync()
                integrate_sllod_a_b1(grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                return
            return cuda.jit(device=gridsync)(kernel)
        else:           # pragma: no cover
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                integrate_sllod_b1[num_blocks, (pb, 1)](grid, vectors, scalars, integrator_params, time)
                integrate_sllod_b2[num_blocks, (pb, 1)](grid, vectors, scalars, integrator_params, time)
                call_update_box_shift[1, (1, 1)](sim_box, integrator_params, time)
                integrate_sllod_a_b1[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                return

            return kernel

