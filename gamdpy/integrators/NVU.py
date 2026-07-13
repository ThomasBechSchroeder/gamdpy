import numpy as np
import numba
from numba import cuda
import h5py
from ..configuration import Configuration
from ..misc.make_function import make_function_constant
from .integrator import Integrator


class NVU(Integrator):
    """
    Integrator keeping N (number of particles), V (volume), and U (potential energy) constant,
    using the NVU algorithm (see eq. 20 in http://jcdyre.dk/2011_JCP_135_104101.pdf)
    
    Parameters
    ----------
    U_0: float or function
        The potential energy (pr particle) to keep the system at. 
        If a function, it should take time as argument.
    
    dl : float
        The length to move along the constant potential energy landscape per integration step.
    
    """

    def __init__(self, U_0, dl: float) -> None:
        self.U_0 = U_0
        self.dt = dl
        self.f_dot_v = np.zeros(1, dtype = np.float32)
        self.f_dot_f = np.zeros(1, dtype = np.float32)
        self.denominator = np.zeros(1, dtype = np.float32)
        self.old_U = np.array([U_0], dtype = np.float32)
        self.U = np.zeros(1, dtype = np.float32)

        self.d_f_dot_v = cuda.to_device(self.f_dot_v)
        self.d_f_dot_f = cuda.to_device(self.f_dot_f)
        self.d_denominator = cuda.to_device(self.denominator)
        self.d_U = cuda.to_device(self.U)
        self.d_old_U = cuda.to_device(self.old_U)

    def get_params(self, configuration: Configuration, interactions_params: tuple, verbose=False) -> tuple:
        dt = np.float32(self.dt)
        return (dt, self.d_f_dot_v, self.d_f_dot_f, 
                self.d_denominator, self.d_U, self.d_old_U)   # Needs to be compatible with unpacking in
                                                                # step() and update_thermostat_state() below.

    def save_internal_state(self, output: h5py.File, group_name: str):
        pass

    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict, interactions_kernel, verbose=False):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        num_blocks = (num_part - 1) // pb + 1

        # Convert temperature to a function if isn't already (better be a number then...)
        if callable(self.U_0):
            U_0_function = self.U_0
        else:
            U_0_function = make_function_constant(value=float(self.U_0))

        if verbose:
            print(f'Generating NVT kernel for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        m_id, k_id, fsq_id, u_id = [configuration.sid[key] for key in ['m', 'K', 'Fsq', 'U']]

        # JIT compile functions to be compiled into kernel
        U_0_function = numba.njit(U_0_function)
        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())

        def reset(integrator_params):
            dt, f_dot_v, f_dot_f, denominator, U, old_U = integrator_params
            global_id, my_t = cuda.grid(2)
            if global_id == 0 and my_t == 0:
                old_U[0] = U[0]
                f_dot_v[0] = numba.float32(0.0)
                f_dot_f[0] = numba.float32(0.0)
                U[0] = numba.float32(0.0)
                denominator[0] = numba.float32(0.0)
            return

        def calc_dot_products(vectors, scalars, integrator_params):
            dt, f_dot_v, f_dot_f, denominator, U, old_U = integrator_params
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_f = vectors[f_id][global_id]
                my_v = vectors[v_id][global_id]
                my_u = scalars[global_id][u_id]
                my_f_dot_v = numba.float32(0.0)
                my_f_dot_f = numba.float32(0.0)
                for k in range(D):
                    my_f_dot_v += my_f[k] * my_v[k]
                    my_f_dot_f += my_f[k] * my_f[k]
                cuda.atomic.add(U, 0, my_u)
                cuda.atomic.add(f_dot_v, 0, my_f_dot_v)
                cuda.atomic.add(f_dot_f, 0, my_f_dot_f)
            return

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            """ Make one NVT timestep using Leap-frog
                Kernel configuration: [num_blocks, (pb, tp)]
            """

            # Unpack parameters. MUST be compatible with get_params() above
            dt, f_dot_v, f_dot_f, denominator, U, old_U = integrator_params
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_v = vectors[v_id][global_id]
                my_f = vectors[f_id][global_id]
                my_u = scalars[global_id][u_id]
                my_denominator = numba.float32(0.0)

                for k in range(D):
                    target_U_0 = U_0_function(time) * num_part
                    nominator = my_v[k] + (numba.float32(-2.)*f_dot_v[0]+old_U[0]-target_U_0) * my_f[k] / f_dot_f[0]
                    my_denominator += nominator*nominator
                    my_v[k] = nominator
                cuda.atomic.add(denominator, 0, my_denominator)
            return

        def step2(grid, vectors, scalars, r_im, sim_box, integrator_params, time):
            dt, f_dot_v, f_dot_f, denominator, U, old_U = integrator_params
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_v = vectors[v_id][global_id]
                my_r = vectors[r_id][global_id]
                my_f = vectors[f_id][global_id]
                my_fsq = numba.float32(0.0)  # force squared
                for k in range(D):
                    my_fsq += my_f[k] * my_f[k]
                    my_v[k] = dt*(my_v[k]/denominator[0]**0.5)
                    my_r[k] += my_v[k]

                apply_PBC(my_r, r_im[global_id], sim_box)
                scalars[global_id][fsq_id] = my_fsq

            return
        calc_dot_products = cuda.jit(device=gridsync)(calc_dot_products)
        step = cuda.jit(device=gridsync)(step)
        step2 = cuda.jit(device=gridsync)(step2)
        reset = cuda.jit(device=gridsync)(reset)

        if gridsync: # construct and return device function
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                reset(integrator_params)
                grid.sync()
                calc_dot_products(vectors, scalars, integrator_params)
                grid.sync()
                step(  grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                grid.sync()
                step2( grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                grid.sync()
                return
            return cuda.jit(device=gridsync)(kernel)
        else: # return python function, which makes kernel-calls
            def kernel(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
                reset[num_blocks, (pb, 1)](integrator_params)
                calc_dot_products[num_blocks, (pb, 1)](vectors, scalars, integrator_params)
                step[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                step2[num_blocks, (pb, 1)](grid, vectors, scalars, r_im, sim_box, integrator_params, time)
                return
            return kernel
