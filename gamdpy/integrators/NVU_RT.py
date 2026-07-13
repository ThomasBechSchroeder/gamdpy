import h5py
import math
import numpy as np
import numba as nb
from numba import cuda
import math
from typing import Any, Literal, Tuple
import numpy.typing as npt
from .integrator import Integrator


# Just to get rid of certain IDE errors but it is not important
CudaArray = npt.NDArray[Any]


class NVU_RT(Integrator):
    """Potential energy conserving integrator.
    Calculate the positions by reflecting on the constant potential energy
    hypersurface and doing Ray Tracing (RT).

    Uses parabola approximations, newton-rhapson or bisection to perform raytrcing.

    Parameters
    ----------

    target_u : float
        Target Potential Energy (U_0) to maintain constant along the simulation

    threshold : float
        Width of the potential energy "shell" relative to U_0.
        [Iterative Method] When the potential energy of a certain configuration is within threshold if the potential energy 
        of the fist step, iteration is finished. |U(t) / U_0 - 1| < threshold. It needs to be small enough so that the 
        iterative method is precise enough (it can get very chaotic if it is too high). For example: 1e-6.

    initial_step : float, default=0.1
        [Iterative Metod] Initial step in configuration space so that x = positions + d * initial_step 
        is the initial guess for the root algorithm. `d` is the velocity normalized. For method bisection,
        it needs to be big enough so that it steps away from the initial position but small enough so that it doesn't 
        reach out of the surface. If it is two big the algorithm will try to correct it. For method parabola it needs
        to be as big as possible but it needs to be a point that is below U_0. For example: 0.01. 
        It is better to be based on the density so a good value is something like 0.5 / rho^(1/3).

    initial_step_if_high : float, default=0.01
        [Iterative Metod] Initial step if the potential energy of the initial configuration (time == 0) is higher than the
        target potential energy. It should be high enough to "enter" the potential energy surface U = U_0. For example: same 
        setting as initial step.

    step : float, default=1
        [Iterative Method] (only for bisection method) Step to look for a point with u > u0. For example: 1.
        As in initial it is better for it to based on the density.

    max_steps : int, default=20
        [Iterative Method] Maximum calls to the interactions kernel to find a point outside the hypersurface. 
        Good enough value is maybe 10 for parabola method and 100 for bisection method.
    
    max_initial_step_corrections : int, default=20
        [Iterative Method] If initial_step is too big the algorithm will try to correct it this amount of times.
        At the nth correction step initial_step_n = initial_step_0  * (1/2)^n. In the first iteration if U > U0
        then initial step is corrected to make it bigger (s_n = s_0 * 2^n). For example: 10 (max correction will
        be approximately 1000).

    max_abs_val : float, default=2
        [Iterative Metod] (only for bisection) Some potential energy functions increas rapidly at a certain configurations.
        To prevent numerical errors, if the absolute potential energy of a given configuration, |U|, 
        is above U_0 * max_abs_val, the iterative method will discard that configuration as invalid and reach
        "less far" into configurational space. For example: 2.

    eps : float, default=1e-7
        [Iterative Method] Because of numerical inaccuracies, it could happen that the same value calculated twice 
        once is positive and once is negative. Values of the potential energy relative to the target potential enery
        with |x| < eps are considered neither positive or negative in the algo. Needs to be smaller than threshold.

    debug_print : bool, default=False
        If a root is not found, the 0th thread prints useful debugging information if debug_print is enabled.

    mode : {"reflection", "no-inertia", "reflection-mass_scaling"}, default = "reflection-mass_scaling"
        Mode to perform the reflection. 
        `reflection-mass_scaling` applies a correction that takes into account the mass of 
        each particle. If the setup does not include particles with different masses, ``reflection`` 
        will be faster (not that much).
        `no-inertia` is a testing feature: instead of reflecitng velocities in the hyper surface the new velocities 
        follow the direction of the normal vector (the force). 

    save_path_u : optional, default=False
        Save the potential energy between two consecutive points in the iteration

    raytracing_method : {"parabola", "parabola-newton", "bisection"}, default = "parabola"
        Methodology to find the next point in the surface with same potential energy

    float_type : {"64", "32"}, default = "64"
        Float type for the potential energy. Higher threshold can work with float32

    """

    outputs = ("its", "cos_v_f", "time", "dt", )

    def __init__(
        self, 
        target_u: float,
        threshold: float, 
        initial_step: float = 0.1, 
        initial_step_if_high: float = 0.01, 
        step: float = 1,
        max_steps: int = 20, 
        max_initial_step_corrections: int = 20,
        max_abs_val: float = 2, 
        eps: float = 1e-7,
        debug_print: bool = False,
        mode: Literal["reflection", "no-inertia", "reflection-mass_scaling"] = "reflection-mass_scaling",
        save_path_u: bool = False,
        raytracing_method: Literal["parabola", "parabola-newton", "bisection"] = "parabola",
        float_type: Literal["32", "64"] = "64"
    ):
        if str(float_type) == "32":
            u_dtype = np.float32
        elif str(float_type) == "64":
            u_dtype = np.float64
        else:
            raise ValueError(f"Expected \"64\" or \"32\" for parameter `float_type`, but got {float_type}")

        self.target_u = u_dtype(target_u)
        self.d_pot_energy = cuda.to_device(np.array([np.nan], dtype=u_dtype))
        self.max_abs_val = np.float32(max_abs_val)
        self.threshold = np.float32(threshold)
        self.step = np.float32(step)
        self.eps = np.float32(eps)
        if self.threshold <= 0:
            raise ValueError(f"`threshold` has to be strictly bigger than 0, got {threshold}")
        if self.eps >= self.threshold:
            raise ValueError(f"`eps` has to be smaller bigger than threshold ({threshold}), got {eps}")
        self.max_steps = np.int32(max_steps)
        self.max_initial_step_corrections = np.int32(max_initial_step_corrections)
        self.initial_step = np.float32(initial_step)
        self.d_initial_step = cuda.to_device(np.array([self.initial_step], dtype=np.float32))
        self.initial_step_if_high = np.float32(initial_step_if_high)
        # Simluation requires that integrators have dt
        self.dt = 1
        self.d_scalars_shared = cuda.device_array(16, dtype=np.float32)  # type: ignore
        self.output_ids = {name: idx for idx, name in enumerate(self.outputs)}
        self.d_integrator_output = cuda.device_array(len(self.outputs), dtype=np.float32)  # type: ignore
        self.d_path_u = cuda.to_device(np.zeros((100, 10, 5), dtype=np.float32) + np.nan)
        self.d_step = cuda.to_device(np.zeros(1, dtype=np.int32))
        self.debug_print = np.bool_(debug_print)
        self.mode = mode
        self.raytracing_method = raytracing_method
        self.save_path_u = np.bool_(save_path_u)
        self.d_broken_simulation = cuda.to_device(np.zeros(1, dtype=np.bool_))
        self.d_last_a = cuda.to_device(np.empty(1, dtype=np.float32) + np.nan)
        self.d_u_higher_than_target_in_time_0 = cuda.to_device(np.empty(1, dtype=np.bool_))

    def get_params(self, configuration, interaction_params, verbose = False):
        # NOTE: for some reason the first param has to be delta time
        return (
            self.dt,
            self.d_integrator_output,
            self.d_initial_step,
            interaction_params,
            self.d_scalars_shared,
            self.d_pot_energy,
            self.d_path_u,
            self.d_step,
            self.d_broken_simulation,
            self.d_last_a,
            self.d_u_higher_than_target_in_time_0,
        )

    def save_internal_state(self, output: h5py.File, group_name: str):
        pass

    def update_at_end_of_timeblock(self, storage: str, nblocks: int, block: int):
        if self.save_path_u:
            self.d_step[0] = 0
            if block == 0:
                with h5py.File(storage, 'a') as f:
                    if "path_u" in f.keys():
                        del f["path_u"]
                    f.create_dataset('path_u', shape=(nblocks, *self.d_path_u.shape),
                                    chunks=(1, *self.d_path_u.shape), dtype=np.float32)
            with h5py.File(storage, "a") as f:
                f["path_u"][block, :] = self.d_path_u.copy_to_host()  # type: ignore
    
    def get_kernel(self, configuration, compute_plan, compute_flags, interactions_kernel, verbose=False):
        # Unpack parameters from configuration and compute_plan
        num_dim, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        if verbose:
            print(f'Generating NVU kernel for {num_part} particles in {num_dim} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        m_id = configuration.sid['m']
        if not compute_flags['U'] or not compute_flags['Fsq']:
            raise ValueError('NVU_RT requires both U and Fsq in scalars')
        u_id = configuration.sid['U']
        fsq_id = configuration.sid['Fsq']


        (
            forces_sql_id, 
            vel_sql_id, 
            dot_v_f_id, 
            should_break_id,
            should_return_id,
            reached_max_abs_val_id,
            should_move_r_copy_id,
            should_flip_velocities_id,
            second_derivative_id,
            first_derivative_id,
            alpha_den_id,
            *debug_ids
        ) = range(self.d_scalars_shared.shape[0])
        debug_ids = tuple(debug_ids)

        o_its, o_cos_v_f, o_time, o_dt = (self.output_ids[name] for name in ["its", "cos_v_f", "time", "dt", ])

        # JIT compile functions to be compiled into kernel
        apply_PBC = nb.jit(configuration.simbox.get_apply_PBC())

        save_path_u = self.save_path_u
        save_path_max_saves = self.d_path_u.shape[0]
        save_path_u_divisions = self.d_path_u.shape[1]

        if gridsync:
            @cuda.jit(device=gridsync)
            def kernel(
                grid, vectors, scalars, r_im, sim_box, 
                integrator_params, time, ptype,
            ):
                (dt, d_integrator_output, d_initial_step, interaction_params, d_scalars_shared, d_pot_energy, d_path_u,
                    d_step, d_broken_simulation, d_last_a, d_u_higher_than_target_in_time_0) = integrator_params
                if time > 0:
                    d_u_higher_than_target_in_time_0[0] = False

                if d_broken_simulation[0]:
                    return

                global_id, my_t = cuda.grid(2)  # type: ignore
                my_m = scalars[global_id, m_id]
                velocities = vectors[v_id]
                my_v = velocities[global_id]
                forces = vectors[f_id]
                my_f = forces[global_id]
                positions = vectors[r_id]
                my_r = positions[global_id]
                my_r_im = r_im[global_id]
                its: CudaArray = cuda.local.array(1, dtype=np.int32)  # type: ignore
                its[0] = 0

                if global_id == 0 and my_t == 0:
                    for k in range(d_scalars_shared.shape[0]):
                        d_scalars_shared[k] = 0
                grid.sync()
                if global_id == 0 and my_t == 0:
                    d_integrator_output[o_time] = time

                if global_id < num_part and my_t == 0:
                    x = np.float32(0)
                    for k in range(num_dim):
                        x += my_f[k] * my_f[k]
                    cuda.atomic.add(d_scalars_shared, forces_sql_id, x)  # type: ignore
                    scalars[global_id][fsq_id] = x
                get_dot_in_conf_space(my_v, my_v, d_scalars_shared, vel_sql_id)
                get_dot_in_conf_space(my_f, my_v, d_scalars_shared, dot_v_f_id)
                grid.sync()
                if global_id == 0 and my_t == 0:
                    if d_scalars_shared[dot_v_f_id] > 0:
                        d_broken_simulation[0] = True
                vel_l = math.sqrt(d_scalars_shared[vel_sql_id])

                calculate_next_velocities(grid, my_f, d_scalars_shared, my_v, my_m)
                grid.sync()

                if global_id == 0 and my_t == 0:
                    cos_v_f = d_scalars_shared[dot_v_f_id] / vel_l / math.sqrt(d_scalars_shared[forces_sql_id])
                    d_integrator_output[o_cos_v_f] = cos_v_f

                my_dir: CudaArray = cuda.local.array(num_dim, velocities.dtype)  # type: ignore
                if global_id < num_part and my_t == 0:
                    for k in range(num_dim):
                        my_dir[k] = my_v[k] / vel_l

                ## R_{i+1} = R_i + t * V_{i+1} | U(R_{i+1}) = U_0
                #################################################
                r0: CudaArray = cuda.local.array(num_dim, my_r.dtype)  # type: ignore
                r0_im: CudaArray = cuda.local.array(num_dim, my_r_im.dtype)  # type: ignore
                copy_positions_and_images(my_r, my_r_im, r0, r0_im)

                step_x = raytracing_kernel(
                    time,
                    interactions_kernel, grid, vectors, scalars, ptype, 
                    sim_box, interaction_params, 
                    r_im,
                    d_pot_energy, 
                    its,
                    d_last_a,
                    d_scalars_shared,
                    vel_l,
                    my_dir,
                    d_u_higher_than_target_in_time_0,
                    d_initial_step,
                )

                if save_path_u and d_step[0] < save_path_max_saves:
                    save_my_r: CudaArray = cuda.local.array(num_dim, my_r.dtype)  # type: ignore
                    save_my_r_im: CudaArray = cuda.local.array(num_dim, my_r_im.dtype)  # type: ignore
                    copy_positions_and_images(my_r, my_r_im, save_my_r, save_my_r_im)

                    save_potential_energy_path(
                        save_path_u_divisions, step_x, d_path_u, d_step, d_scalars_shared,
                        r0, r0_im, my_r, my_r_im, my_dir, 
                        interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its
                    )

                    copy_positions_and_images(save_my_r, save_my_r_im, my_r, my_r_im)
                    calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)
                if global_id == 0 and my_t == 0:
                    d_integrator_output[o_dt] = step_x / vel_l
                    d_integrator_output[o_its] = its[0]
        else:
            raise NotImplementedError()

        (
            get_dot_in_conf_space, 
            copy_positions_and_images,
            calculate_potential_energy,
            add_step_in_dir,
        ) = util_functions(gridsync, num_part, num_dim, apply_PBC, u_id)
        calculate_next_velocities = self.get_update_velocities_kernel(
            gridsync=gridsync,
            num_part=num_part,
            num_dim=num_dim,
            dot_v_f_id=dot_v_f_id,
            forces_sql_id=forces_sql_id,
            vel_sql_id=vel_sql_id,
            alpha_den_id=alpha_den_id,
        )

        @cuda.jit(device=gridsync)
        def save_potential_energy_path(
            save_path_u_divisions, delta_x, d_path_u, d_step, d_scalars_shared,
            r0, r0_im, my_r, my_r_im, my_dir,
            interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its
        ):
            global_id, my_t = cuda.grid(2)  # type: ignore
            it = np.int32(0)
            while it < save_path_u_divisions:
                path_step = it * delta_x / (save_path_u_divisions - 1)
                add_step_in_dir(r0, r0_im, my_dir, path_step, my_r, my_r_im, sim_box)
                grid.sync()
                calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)
                if global_id == 0 and my_t == 0:
                    d_path_u[d_step[0], it, 0] = path_step
                    d_path_u[d_step[0], it, 1] = d_pot_energy[0]
                    d_path_u[d_step[0], it, 2] = 0
                    d_path_u[d_step[0], it, 3] = d_scalars_shared[vel_sql_id]
                    # v · f is of the non-reflected v so we change signs
                    d_path_u[d_step[0], it, 4] = d_scalars_shared[dot_v_f_id]
                    d_path_u[d_step[0], it, 5] = delta_x
                grid.sync()
                if global_id < num_part and my_t == 0:
                    cuda.atomic.add(d_path_u, (d_step[0], it, 2), scalars[global_id][lap_id])  # type: ignore
                grid.sync()
                it += 1
            if global_id == 0 and my_t == 0:
                d_step[0] += 1

        if self.raytracing_method == "bisection":
            raytracing_kernel = self.get_kernel_bisection(
                gridsync=gridsync,
                num_dim=num_dim,
                r_id=r_id,
                should_break_id=should_break_id,
                should_move_r_copy_id=should_move_r_copy_id,
                reached_max_abs_val_id=reached_max_abs_val_id,
                should_return_id=should_return_id,
                copy_positions_and_images=copy_positions_and_images,
                add_step_in_dir=add_step_in_dir,
                calculate_potential_energy=calculate_potential_energy,
            )
        elif self.raytracing_method == "parabola":
            raytracing_kernel = self.get_kernel_parabola(
                gridsync=gridsync,
                num_part=num_part,
                num_dim=num_dim,
                r_id=r_id,
                u_id=u_id,
                dot_v_f_id=dot_v_f_id,
                should_break_id=should_break_id,
                copy_positions_and_images=copy_positions_and_images,
                add_step_in_dir=add_step_in_dir,
                calculate_potential_energy=calculate_potential_energy,
            )
        elif self.raytracing_method == "parabola-newton":
            raytracing_kernel = self.get_kernel_parabola_newton(
                gridsync=gridsync,
                num_part=num_part,
                num_dim=num_dim,
                r_id=r_id,
                f_id=f_id,
                u_id=u_id,
                dot_v_f_id=dot_v_f_id,
                should_break_id=should_break_id,
                first_derivative_id=first_derivative_id,
                copy_positions_and_images=copy_positions_and_images,
                add_step_in_dir=add_step_in_dir,
                calculate_potential_energy=calculate_potential_energy,
            )
        else:
            assert False, "unreachable"

        return kernel

    def get_update_velocities_kernel(
        self,
        gridsync: bool,
        num_part: int,
        num_dim: int,
        dot_v_f_id: int,
        forces_sql_id: int,
        vel_sql_id: int,
        alpha_den_id: int,
    ):
        if self.mode == "reflection-mass_scaling":
            @cuda.jit(device=gridsync)
            def update_velocities_kernel(
                grid, my_f, d_scalars_shared, my_v, my_m,
            ):
                global_id, my_t = cuda.grid(2)  # type: ignore

                if global_id < num_part and my_t == 0:
                    den = 0
                    for k in range(num_dim):
                        den += my_f[k] ** 2
                    den = den / my_m
                    cuda.atomic.add(d_scalars_shared, alpha_den_id, den)  # type: ignore
                grid.sync()
                    
                alpha = - 2 * d_scalars_shared[dot_v_f_id] / d_scalars_shared[alpha_den_id]

                if global_id < num_part and my_t == 0:
                    for k in range(num_dim):
                        my_v[k] = my_v[k] + alpha / my_m * my_f[k]

                if global_id == 0 and my_t == 0:
                    d_scalars_shared[dot_v_f_id] = - d_scalars_shared[dot_v_f_id] 

        elif self.mode == "reflection":
            @cuda.jit(device=gridsync)
            def update_velocities_kernel(
                grid, my_f, d_scalars_shared, my_v, my_m,
            ):
                global_id, my_t = cuda.grid(2)  # type: ignore
                
                ## v_{i+1} = v_i -2 * (F_i * v_i)/|F_i|**2 * F_i
                ################################################
                if global_id < num_part and my_t == 0:
                    for k in range(num_dim):
                        my_v[k] = my_v[k] - 2 * d_scalars_shared[dot_v_f_id] / d_scalars_shared[forces_sql_id] * my_f[k]
                if global_id == 0 and my_t == 0:
                    d_scalars_shared[dot_v_f_id] = - d_scalars_shared[dot_v_f_id] 
        elif self.mode == "no-inertia":
            @cuda.jit(device=gridsync)
            def update_velocities_kernel(
                grid, my_f, d_scalars_shared, my_v, my_m,
            ):
                global_id, my_t = cuda.grid(2)  # type: ignore
                
                vel_l = math.sqrt(d_scalars_shared[vel_sql_id])
                force_l = math.sqrt(d_scalars_shared[forces_sql_id])
                if global_id < num_part and my_t == 0:
                    for k in range(num_dim):
                        my_v[k] = vel_l * my_f[k] / force_l
                if global_id == 0 and my_t == 0:
                    d_scalars_shared[dot_v_f_id] = vel_l * force_l
        else:
            assert False, "Unreachable"
        return update_velocities_kernel

    def get_setup_kernel(self, configuration, compute_plan, interactions_kernel):
        num_dim, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, v_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'v', 'f']]
        u_id, m_id = configuration.sid['U'], configuration.sid['m']

        apply_PBC = nb.jit(configuration.simbox.apply_PBC)
        (
            forces_sql_id, 
            vel_sql_id, 
            dot_v_f_id, 
            should_break_id,
            should_return_id,
            reached_max_abs_val_id,
            should_move_r_copy_id,
            should_flip_velocities_id,
            second_derivative_id,
            first_derivative_id,
            alpha_den_id,
            *debug_ids
        ) = range(self.d_scalars_shared.shape[0])
        debug_ids = tuple(debug_ids)

        max_initial_step_corrections = self.max_initial_step_corrections
        initial_step_if_high = self.initial_step_if_high
        threshold = self.threshold
        eps = self.eps
        target_u = self.target_u
        debug_print = self.debug_print

        if gridsync:
            @cuda.jit(device=gridsync)
            def kernel_setup(
                grid, vectors, scalars, r_im, sim_box, 
                integrator_params, ptype,
            ):
                (dt, d_integrator_output, d_initial_step, interaction_params, d_scalars_shared, d_pot_energy, d_path_u,
                    d_step, d_broken_simulation, d_last_a, d_u_higher_than_target_in_time_0) = integrator_params
                global_id, my_t = cuda.grid(2)  # type: ignore

                if global_id == 0 and my_t == 0:
                    for k in range(d_scalars_shared.shape[0]):
                        d_scalars_shared[k] = 0
                grid.sync()

                my_m = scalars[global_id, m_id]
                my_v = vectors[v_id, global_id]
                forces = vectors[f_id]
                my_f = forces[global_id]
                positions = vectors[r_id]
                my_r = positions[global_id]
                my_r_im = r_im[global_id]
                its: CudaArray = cuda.local.array(1, dtype=np.int32)  # type: ignore
                its[0] = 0

                calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)
                get_dot_in_conf_space(my_f, my_f, d_scalars_shared, forces_sql_id)
                get_dot_in_conf_space(my_v, my_v, d_scalars_shared, vel_sql_id)
                get_dot_in_conf_space(my_f, my_v, d_scalars_shared, dot_v_f_id)
                grid.sync()
                vel_l = math.sqrt(d_scalars_shared[vel_sql_id])

                if global_id == 0 and my_t == 0:
                    u = d_pot_energy[0]
                    du_rel = (u - target_u) / abs(target_u)
                    if abs(du_rel) < threshold:
                        pass
                    elif du_rel > threshold:
                        d_u_higher_than_target_in_time_0[0] = True
                        if debug_print:
                            print("Initial configuration has U > U0:", u, target_u, du_rel, math.log10(abs(du_rel)))
                            print("In the past, this is not a `good` point to start the simulation. If it does not work "
                                  "try running with a configuration that has U <= U0")
                    else:
                        if debug_print:
                            print("Initial configuration has U < U0:", u, target_u, du_rel, math.log10(abs(du_rel)))

                    if d_u_higher_than_target_in_time_0[0]:
                        if global_id == 0 and my_t == 0:
                            d_initial_step[0] = initial_step_if_high
                    if d_scalars_shared[dot_v_f_id] > 0:
                        d_scalars_shared[should_flip_velocities_id] = True
                grid.sync()
                if global_id < num_part and my_t == 0:
                    if d_scalars_shared[should_flip_velocities_id]:
                        for k in range(num_dim):
                            my_v[k] = my_v[k] * np.float32(-1)
                        if global_id == 0 and my_t == 0:
                            d_scalars_shared[dot_v_f_id] = - d_scalars_shared[dot_v_f_id] 
                grid.sync()

                my_dir: CudaArray = cuda.local.array(num_dim, my_v.dtype)  # type: ignore
                if global_id < num_part and my_t == 0:
                    for k in range(num_dim):
                        my_dir[k] = my_v[k]

                calculate_next_velocities(grid, my_f, d_scalars_shared, my_dir, my_m)
                grid.sync()
                if global_id < num_part and my_t == 0:
                    for k in range(num_dim):
                        my_dir[k] = my_dir[k] / vel_l

                # Used to reset positions later
                r0: CudaArray = cuda.local.array(num_dim, my_r.dtype)  # type: ignore
                r0_im: CudaArray = cuda.local.array(num_dim, my_r_im.dtype)  # type: ignore
                copy_positions_and_images(my_r, my_r_im, r0, r0_im)
                setup_raytracing(
                    my_f, d_broken_simulation, d_last_a,
                    d_scalars_shared, debug_print, 0,
                    r0, r0_im, my_dir, d_initial_step, my_r, my_r_im, 
                    interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its,
                    d_u_higher_than_target_in_time_0,
                )
                copy_positions_and_images(r0, r0_im, my_r, my_r_im)
        else:
            raise NotImplementedError()

        (
            get_dot_in_conf_space, 
            copy_positions_and_images,
            calculate_potential_energy,
            add_step_in_dir,
        ) = util_functions(gridsync, num_part, num_dim, apply_PBC, u_id)
        calculate_next_velocities = self.get_update_velocities_kernel(
            gridsync=gridsync,
            num_part=num_part,
            num_dim=num_dim,
            dot_v_f_id=dot_v_f_id,
            forces_sql_id=forces_sql_id,
            vel_sql_id=vel_sql_id,
            alpha_den_id=alpha_den_id,
        )

        if self.raytracing_method == "bisection":
            @cuda.jit(device=gridsync)
            def setup_raytracing(
                my_f, d_broken_simulation, d_last_a,
                d_scalars_shared, debug_print, time,
                r0, r0_im, my_dir, d_initial_step, my_r, my_r_im, 
                interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its,
                d_u_higher_than_target_in_time_0,
            ):
                return

        elif self.raytracing_method == "parabola-newton" or self.raytracing_method == "parabola":
            @cuda.jit(device=gridsync)
            def setup_raytracing(
                my_f, d_broken_simulation, d_last_a,
                d_scalars_shared, debug_print, time,
                r0, r0_im, my_dir, d_initial_step, my_r, my_r_im, 
                interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its,
                d_u_higher_than_target_in_time_0,
            ):
                global_id, my_t = cuda.grid(2)  # type: ignore
                force0: CudaArray = cuda.local.array(num_dim, my_f.dtype)  # type: ignore
                for k in range(num_dim):
                    force0[k] = my_f[k]

                if global_id == 0 and my_t == 0:
                    d_scalars_shared[should_break_id] = False
                grid.sync()

                x_initial_step = d_initial_step[0]
                corrected_step = np.int32(0)
                while corrected_step <= max_initial_step_corrections and not d_scalars_shared[should_break_id]:
                    corrected_step += 1

                    # Initial step go into a bit further from the surface into lower energies
                    add_step_in_dir(r0, r0_im, my_dir, x_initial_step, my_r, my_r_im, sim_box)
                    grid.sync()
                    calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)

                    if global_id == 0 and my_t == 0:
                        u = d_pot_energy[0]
                        du_rel = (u - target_u) / abs(target_u)
                        if du_rel < -eps:
                            d_scalars_shared[should_break_id] = True
                    grid.sync()
                    if not d_scalars_shared[should_break_id]:
                        if d_u_higher_than_target_in_time_0[0]:
                            x_initial_step = x_initial_step * 2
                        else:
                            x_initial_step = x_initial_step / 2

                if corrected_step > max_initial_step_corrections and not d_scalars_shared[should_break_id]:
                    if global_id == 0 and my_t == 0:
                        d_broken_simulation[0] = True
                    uf = d_pot_energy[0]
                    copy_positions_and_images(r0, r0_im, my_r, my_r_im)
                    grid.sync()
                    calculate_potential_energy(
                        interactions_kernel, grid, vectors, scalars, ptype, 
                        sim_box, interaction_params, d_pot_energy, its)
                    d = d_scalars_shared[dot_v_f_id]
                    u1 = d_pot_energy[0]
                    if global_id == 0 and my_t == 0:
                        d_scalars_shared[dot_v_f_id] = 0
                    grid.sync()
                    get_dot_in_conf_space(my_f, my_dir, d_scalars_shared, dot_v_f_id)
                    grid.sync()
                    b1 = - d_scalars_shared[dot_v_f_id]

                    add_step_in_dir(r0, r0_im, my_dir, d_initial_step[0], my_r, my_r_im, sim_box)
                    grid.sync()
                    calculate_potential_energy(
                        interactions_kernel, grid, vectors, scalars, ptype, 
                        sim_box, interaction_params, d_pot_energy, its)
                    copy_positions_and_images(r0, r0_im, my_r, my_r_im)
                    u2 = d_pot_energy[0]
                    if global_id == 0 and my_t == 0:
                        d_scalars_shared[dot_v_f_id] = 0
                    grid.sync()
                    get_dot_in_conf_space(my_f, my_dir, d_scalars_shared, dot_v_f_id)
                    grid.sync()
                    b2 = - d_scalars_shared[dot_v_f_id]
                    a = 0.5 * (b2 - b1) / d_initial_step[0]
                    if global_id == 0 and my_t == 0:
                        if debug_print:
                            print("ERROR:", time, "Reached max_initial_step_corrections")
                            print("  uf =", uf)
                            print("  u1 =", u1)
                            print("  b1 =", b1)
                            print("  u2 =", u2)
                            print("  b2 =", b2)
                            print("  a  =", a)
                            print("  initial_step0 = ", d_initial_step[0])
                            print("  initial_stepf = ", x_initial_step)
                            print("  dot_v_f0 = ", d)
                    # return
                upp = get_second_derivative(
                    grid, my_f, force0, my_dir, x_initial_step, d_scalars_shared
                )
                if global_id == 0 and my_t == 0:
                    a = upp / 2
                    d_last_a[0] = a
        else:
            assert False, "Unreachable"

        @cuda.jit(device=gridsync)
        def get_second_derivative(
            grid, my_f, force0, my_dir, delta, d_scalars_shared
        ):
            global_id, my_t = cuda.grid(2)  # type: ignore
            if global_id < num_part and my_t == 0:
                acc = np.float32(0)
                for k in range(num_dim):
                    acc += - (my_f[k] - force0[k]) * my_dir[k] / delta
                cuda.atomic.add(d_scalars_shared, second_derivative_id, acc)  # type: ignore
            grid.sync()
            return d_scalars_shared[second_derivative_id]

        if gridsync:
            @cuda.jit
            def integrator_setup(
                vectors, scalars, r_im, sim_box, 
                integrator_params, ptype,
            ):
                grid = cuda.cg.this_grid()
                kernel_setup(
                    grid, vectors, scalars, r_im, sim_box, 
                    integrator_params, ptype,
                )
            return integrator_setup[num_blocks, (pb, tp)]
        else:
            raise NotImplementedError()

    def get_kernel_parabola(
        self, 
        gridsync: bool, 
        num_part: int, 
        num_dim: int,
        r_id: int,
        u_id: int,
        dot_v_f_id: int,
        should_break_id: int,
        
        copy_positions_and_images: Any,
        add_step_in_dir: Any,
        calculate_potential_energy: Any,
    ):
        max_steps = self.max_steps
        target_u = self.target_u
        threshold = self.threshold

        if gridsync:
            @cuda.jit(device=gridsync)
            def parabola_kernel(
                time,
                interactions_kernel, grid, vectors, scalars, ptype, 
                sim_box, interaction_params, 
                r_im,
                d_pot_energy, 
                its,
                d_last_a,
                d_scalars_shared,
                vel_l,
                my_dir,
                d_u_higher_than_target_in_time_0,
                d_initial_step,
            ):
                global_id, my_t = cuda.grid(2)  # type: ignore

                positions = vectors[r_id]
                my_r = positions[global_id]
                my_r_im = r_im[global_id]
                r0: CudaArray = cuda.local.array(num_dim, my_r.dtype)  # type: ignore
                r0_im: CudaArray = cuda.local.array(num_dim, my_r_im.dtype)  # type: ignore
                copy_positions_and_images(my_r, my_r_im, r0, r0_im)

                a = d_last_a[0]
                b = - d_scalars_shared[dot_v_f_id] / vel_l

                if global_id == 0 and my_t == 0:
                    d_pot_energy[0] = 0
                grid.sync()
                if global_id < num_part and my_t == 0:
                    cuda.atomic.add(d_pot_energy, 0, scalars[global_id][u_id])  # type: ignore
                grid.sync()
                c = d_pot_energy[0] - target_u

                if global_id == 0 and my_t == 0:
                    d_scalars_shared[should_break_id] = False
                grid.sync()

                x1 = - b / (2*a) * (math.sqrt(1 - 4 * a * c / b**2) + 1)
                add_step_in_dir(r0, r0_im, my_dir, x1, my_r, my_r_im, sim_box)
                grid.sync()
                calculate_potential_energy(
                    interactions_kernel, grid, vectors, scalars, ptype, 
                    sim_box, interaction_params, d_pot_energy, its)
                u = d_pot_energy[0]
                if global_id == 0 and my_t == 0:
                    du_rel = (u - target_u) / abs(target_u)
                    if abs(du_rel) < threshold:
                        d_scalars_shared[should_break_id] = True
                grid.sync()
                if d_scalars_shared[should_break_id]:
                    return x1

                u1 = u - target_u
                x2 = b * x1**2 / (b*x1 - u1)
                u2 = 0  # Only to solve unbound issues

                steps_done = 1
                while steps_done <= max_steps and not d_scalars_shared[should_break_id]:
                    steps_done += 1
                    add_step_in_dir(r0, r0_im, my_dir, x2, my_r, my_r_im, sim_box)
                    grid.sync()
                    calculate_potential_energy(
                        interactions_kernel, grid, vectors, scalars, ptype, 
                        sim_box, interaction_params, d_pot_energy, its)
                    u = d_pot_energy[0]
                    if global_id == 0 and my_t == 0:
                        du_rel = (u - target_u) / abs(target_u)
                        if abs(du_rel) < threshold:
                            d_scalars_shared[should_break_id] = True
                    grid.sync()
                    u2 = u - target_u
                    if not d_scalars_shared[should_break_id]:
                        s = (u2 * x1**2 - u1 * x2**2) / (u2 * x1 - u1 * x2)
                        u1 = u2
                        x1 = x2
                        x2 = s
                if global_id == 0 and my_t == 0:
                    d_last_a[0] = - b / x2
                return x2
        else:
            raise NotImplementedError()
        return parabola_kernel

    def get_kernel_parabola_newton(
        self, 
        gridsync: bool, 
        num_part: int, 
        num_dim: int,
        r_id: int,
        f_id: int,
        u_id: int,
        dot_v_f_id: int,
        should_break_id: int,
        first_derivative_id: int,
        
        copy_positions_and_images: Any,
        add_step_in_dir: Any,
        calculate_potential_energy: Any,
    ):
        max_steps = self.max_steps
        target_u = self.target_u
        threshold = self.threshold

        if gridsync:
            @cuda.jit(device=gridsync)
            def parabola_newton_kernel(
                time,
                interactions_kernel, grid, vectors, scalars, ptype, 
                sim_box, interaction_params, 
                r_im,
                d_pot_energy, 
                its,
                d_last_a,
                d_scalars_shared,
                vel_l,
                my_dir,
                d_u_higher_than_target_in_time_0,
                d_initial_step,
            ):
                global_id, my_t = cuda.grid(2)  # type: ignore

                forces = vectors[f_id]
                my_f = forces[global_id]
                positions = vectors[r_id]
                my_r = positions[global_id]
                my_r_im = r_im[global_id]
                r0: CudaArray = cuda.local.array(num_dim, my_r.dtype)  # type: ignore
                r0_im: CudaArray = cuda.local.array(num_dim, my_r_im.dtype)  # type: ignore
                copy_positions_and_images(my_r, my_r_im, r0, r0_im)

                a = d_last_a[0]

                b = - d_scalars_shared[dot_v_f_id] / vel_l

                if global_id == 0 and my_t == 0:
                    d_pot_energy[0] = 0
                grid.sync()
                if global_id < num_part and my_t == 0:
                    cuda.atomic.add(d_pot_energy, 0, scalars[global_id][u_id])  # type: ignore
                grid.sync()
                c = d_pot_energy[0] - target_u

                if global_id == 0 and my_t == 0:
                    d_scalars_shared[should_break_id] = False
                grid.sync()

                step = - b / (2*a) * (math.sqrt(1 - 4 * a * c / b**2) + 1)

                if global_id == 0 and my_t == 0:
                    d_scalars_shared[should_break_id] = False
                grid.sync()

                steps_done = np.int32(0)
                while steps_done <= max_steps and not d_scalars_shared[should_break_id]:
                    steps_done += 1
                    add_step_in_dir(r0, r0_im, my_dir, step, my_r, my_r_im, sim_box)
                    grid.sync()
                    calculate_potential_energy(
                        interactions_kernel, grid, vectors, scalars, ptype, 
                        sim_box, interaction_params, d_pot_energy, its)

                    if global_id == 0 and my_t == 0:
                        u = d_pot_energy[0]
                        du_rel = (u - target_u) / abs(target_u)
                        if abs(du_rel) < threshold:
                            d_scalars_shared[should_break_id] = True
                    grid.sync()
                    if not d_scalars_shared[should_break_id]:
                        u_prime = get_first_derivative(
                            grid, my_f, my_dir, d_scalars_shared
                        )
                        # x_i+1 = x_i - f(x_n) / f'(x_n)
                        step = step - (d_pot_energy[0] - target_u) / u_prime
                return step
        else:
            raise NotImplementedError()

        @cuda.jit(device=gridsync)
        def get_first_derivative(
            grid, my_f, my_dir, d_scalars_shared
        ):
            global_id, my_t = cuda.grid(2)  # type: ignore
            if global_id == 0 and my_t == 0:
                d_scalars_shared[first_derivative_id] = 0
            grid.sync()
            if global_id < num_part and my_t == 0:
                acc = np.float32(0)
                for k in range(num_dim):
                    acc += - my_f[k] * my_dir[k]
                cuda.atomic.add(d_scalars_shared, first_derivative_id, acc)  # type: ignore
            grid.sync()
            return d_scalars_shared[first_derivative_id]

        return parabola_newton_kernel

    def get_kernel_bisection(
        self, 
        gridsync: bool, 
        num_dim: int,
        r_id: int,
        should_break_id: int,
        should_move_r_copy_id: int,
        reached_max_abs_val_id: int,
        should_return_id: int,

        copy_positions_and_images: Any,
        add_step_in_dir: Any,
        calculate_potential_energy: Any,
    ):
        max_steps = self.max_steps
        target_u = self.target_u
        threshold = self.threshold
        eps = self.eps
        initial_step0 = self.initial_step
        debug_print = self.debug_print
        step = self.step
        max_initial_step_corrections = self.max_initial_step_corrections
        max_abs_val = self.max_abs_val

        if gridsync: # construct and return device function
            @cuda.jit(device=gridsync)
            def bisection_kernel(
                time,
                interactions_kernel, grid, vectors, scalars, ptype, 
                sim_box, interaction_params, 
                r_im,
                d_pot_energy, 
                its,
                d_last_a,
                d_scalars_shared,
                vel_l,
                my_dir,
                d_u_higher_than_target_in_time_0,
                d_initial_step,
            ):
                global_id, my_t = cuda.grid(2)  # type: ignore
                positions = vectors[r_id]
                my_r = positions[global_id]
                my_r_im = r_im[global_id]
                # INITIAL STEP
                ##############
                r_copy: CudaArray = cuda.local.array(num_dim, my_r.dtype)  # type: ignore
                r_im_copy: CudaArray = cuda.local.array(num_dim, r_im.dtype)  # type: ignore
                r_copy_u: CudaArray = cuda.local.array(1, d_pot_energy.dtype)  # type: ignore
                r_copy_u[0] = d_pot_energy[0]
                copy_positions_and_images(my_r, my_r_im, r_copy, r_im_copy)

                delta_x = 0
                if global_id == 0 and my_t == 0:
                    d_scalars_shared[should_break_id] = False
                grid.sync()

                x_initial_step = d_initial_step[0]
                corrected_step = np.int32(0)
                while corrected_step <= max_initial_step_corrections and not d_scalars_shared[should_break_id]:
                    corrected_step += 1

                    # Initial step go into a bit further from the surface into lower energies
                    add_step_in_dir(r_copy, r_im_copy, my_dir, x_initial_step, my_r, my_r_im, sim_box)
                    grid.sync()
                    calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)

                    if global_id == 0 and my_t == 0:
                        u = d_pot_energy[0]
                        du_rel = (u - target_u) / abs(target_u)
                        if du_rel < -eps:
                            d_scalars_shared[should_break_id] = True
                    grid.sync()
                    if not d_scalars_shared[should_break_id]:
                        if d_u_higher_than_target_in_time_0[0]:
                            x_initial_step = x_initial_step * 2
                        else:
                            x_initial_step = x_initial_step / 2

                if global_id == 0 and my_t == 0:
                    if d_u_higher_than_target_in_time_0[0]:
                        d_initial_step[0] = initial_step0
                    else:
                        d_initial_step[0] = x_initial_step

                delta_x += x_initial_step
                if corrected_step > max_initial_step_corrections and not d_scalars_shared[should_break_id]:
                    # TODO: deal with this: reached max steps for the initial step correction
                    # Most likely this is an equilibrium position. 
                    # We should think about setting the reflected ray to a random direction
                    if global_id == 0 and my_t == 0:
                        if debug_print:
                            print("ERROR:", time, "Reached max_initial_step_corrections =>", 
                                  (d_pot_energy[0] - target_u) / abs(target_u),
                                  x_initial_step)
                    return delta_x

                r_copy_u[0] = d_pot_energy[0]
                copy_positions_and_images(my_r, my_r_im, r_copy, r_im_copy)

                # FIND POINT OUTSIDE SURFACE
                ############################

                should_return_above, x_above, steps_done = perform_find_point_above(
                    d_scalars_shared, time,
                    r_copy, r_im_copy, my_dir, step, my_r, my_r_im, r_copy_u,
                    interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its
                )
                delta_x += x_above
                if should_return_above:
                    return delta_x

                ## PERFORM BISECTION
                ####################

                ### Root between r_copy(r_copy_u) and my_r(d_pot_energy[0])
                
                if global_id == 0 and my_t == 0 and debug_print:
                    before_bisec = (r_copy_u[0] - target_u) / abs(target_u), (d_pot_energy[0] - target_u) / abs(target_u)
                else:
                    before_bisec = np.nan, np.nan

                should_return_bisection, x_bisection = perform_bisection(
                    steps_done, time, before_bisec,
                    d_scalars_shared, 
                    r_copy, r_im_copy, my_dir, step, my_r, my_r_im, r_copy_u,
                    interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its
                )
                delta_x += x_bisection
                if should_return_bisection:
                    return delta_x
                return delta_x
        else: # return python function, which makes kernel-calls
            raise ValueError("Currently no gridsync is not supported for NVU_RT")

        @cuda.jit(device=gridsync)
        def perform_find_point_above(
            d_scalars_shared, time,
            r_copy, r_im_copy, my_dir, step, my_r, my_r_im, r_copy_u,
            interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its
        ):
            global_id, my_t = cuda.grid(2)  # type: ignore
            if global_id == 0 and my_t == 0:
                d_scalars_shared[should_break_id] = False
                # print("Potential Energy before searching:", d_scalars_shared[pot_energy_id])
            grid.sync()

            delta_x = np.float32(0)
            steps_done = np.int32(0)
            while steps_done <= max_steps and not d_scalars_shared[should_break_id]:
                steps_done += 1

                add_step_in_dir(r_copy, r_im_copy, my_dir, step, my_r, my_r_im, sim_box)
                grid.sync()
                calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)

                if global_id == 0 and my_t == 0:
                    du = d_pot_energy[0] - target_u
                    du_rel = du / abs(target_u)
                    # Max Abs Val is there to prevent any overflows that can happen if U suddenly rises
                    d_scalars_shared[should_move_r_copy_id] = False
                    if abs(du_rel) > max_abs_val:
                        if debug_print:
                            print("ERROR:", time, "Reached max_abs_val in", steps_done, "step", 
                                  du_rel, "Consider using a lower `step` parameter")
                        d_scalars_shared[should_break_id] = True
                        d_scalars_shared[reached_max_abs_val_id] = True
                    elif abs(du_rel) < threshold:
                        d_scalars_shared[should_break_id] = True
                        d_scalars_shared[should_return_id] = True
                    elif du_rel > eps:
                        d_scalars_shared[should_break_id] = True
                    elif du_rel < -eps:
                        d_scalars_shared[should_move_r_copy_id] = True
                    else:
                        # This means that abs(du_rel) < eps, which is a case already covered with theshold
                        # because we requiere eps < threshold
                        pass
                grid.sync()
                if d_scalars_shared[should_move_r_copy_id]:
                    delta_x += step
                    copy_positions_and_images(my_r, my_r_im, r_copy, r_im_copy)
                    r_copy_u[0] = d_pot_energy[0]
                grid.sync()

            if d_scalars_shared[should_return_id]:
                # We found the root sooner
                # if global_id == 0 and my_t == 0:
                #     print("SUCCESS before bisec:", time, " => ", du_rel)
                delta_x += step
                return True, delta_x, steps_done
            if d_scalars_shared[reached_max_abs_val_id]:
                copy_positions_and_images(r_copy, r_im_copy, my_r, my_r_im)
                if global_id == 0 and my_t == 0:
                    if debug_print:
                        du_rel = (d_pot_energy[0] - target_u)/ abs(target_u)
                        print("ERROR:", time, "Reached max_abs_val before bisection.", du_rel)
                return True, delta_x, steps_done
            if not d_scalars_shared[should_break_id] and steps_done > max_steps:
                # TODO: deal with this: reached max steps
                # Use r_copy because my_r is propbably above U0 bevause if not in the wjiole loop r_copy 
                #  should have been moved and so r_copy == my_r
                copy_positions_and_images(r_copy, r_im_copy, my_r, my_r_im)
                if global_id == 0 and my_t == 0:
                    if debug_print:
                        du_rel = (d_pot_energy[0] - target_u)/ abs(target_u)
                        print("ERROR:", time, "Reached max_steps before bisection.", du_rel, math.log10(abs(du_rel)))
                return True, delta_x, steps_done
            return False, delta_x, steps_done

        @cuda.jit(device=gridsync)
        def perform_bisection(
            steps_done, time, before_bisec,
            d_scalars_shared, 
            r_copy, r_im_copy, my_dir, step, my_r, my_r_im, r_copy_u,
            interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its
        ):
            global_id, my_t = cuda.grid(2)  # type: ignore

            if global_id == 0 and my_t == 0:
                d_scalars_shared[should_break_id] = False
            grid.sync()

            delta_x = np.float32(0)
            while steps_done <= max_steps and not d_scalars_shared[should_break_id]:
                steps_done += 1
                step = step / 2

                add_step_in_dir(r_copy, r_im_copy, my_dir, step, my_r, my_r_im, sim_box)
                grid.sync()
                calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)

                if global_id == 0 and my_t == 0:
                    d_scalars_shared[should_move_r_copy_id] = False
                    du = d_pot_energy[0] - target_u
                    du_rel = du / abs(target_u)
                    if abs(du_rel) > max_abs_val:
                        pass
                    elif abs(du_rel) < threshold:
                    # elif (abs(u / target_u - 1) < threshold):
                        d_scalars_shared[should_break_id] = True
                    elif du_rel < -eps:
                        d_scalars_shared[should_move_r_copy_id] = True
                    elif du_rel > eps:
                        pass
                    else:
                        pass
                grid.sync()

                if d_scalars_shared[should_move_r_copy_id]:
                    delta_x += step
                    copy_positions_and_images(my_r, my_r_im, r_copy, r_im_copy)
                    r_copy_u[0] = d_pot_energy[0]
                grid.sync()

            if d_scalars_shared[should_break_id] and steps_done > max_steps:
                # TODO: deal with this: reached max steps. 
                # For now use the lower energy positions
                copy_positions_and_images(r_copy, r_im_copy, my_r, my_r_im)
                grid.sync()
                if debug_print:
                    if global_id == 0 and my_t == 0:
                        du_rel = (d_pot_energy[0] - target_u) / abs(target_u)
                    else:
                        du_rel = np.nan
                    calculate_potential_energy(interactions_kernel, grid, vectors, scalars, ptype, sim_box, interaction_params, d_pot_energy, its)
                    if global_id == 0 and my_t == 0:
                        du_rel2 = (d_pot_energy[0] - target_u) / abs(target_u)
                        print("ERROR:", time, "Reached max steps within bisection.", steps_done)
                        print("Before bisec:", before_bisec[0], before_bisec[1], math.log10(abs(before_bisec[0])), math.log10(abs(before_bisec[1])))
                        print("After bisec:", du_rel2, du_rel, math.log10(abs(du_rel2)), math.log10(abs(du_rel)))
                return True, delta_x

            delta_x += step
            return False, delta_x
        return bisection_kernel



def util_functions(
    gridsync: bool, num_part: int, num_dim: int, apply_PBC: Any,
    u_id: int,
):
    @cuda.jit(device=gridsync)
    def get_dot_in_conf_space(my_a, my_b, result_arr, result_id):
        global_id, my_t = cuda.grid(2)  # type: ignore
        if global_id < num_part and my_t == 0:
            x = np.float32(0)
            for k in range(num_dim):
                x += my_a[k] * my_b[k]
            cuda.atomic.add(result_arr, result_id, x)  # type: ignore
    @cuda.jit(device=gridsync)
    def copy_positions_and_images(source_r, source_r_im, dest_r, dest_r_im):
        global_id, my_t = cuda.grid(2)  # type: ignore
        if global_id < num_part and my_t == 0:
            for k in range(num_dim):
                dest_r[k] = source_r[k]
                dest_r_im[k] = source_r_im[k]

    @cuda.jit(device=gridsync)
    def calculate_potential_energy(
        interactions_kernel, grid, vectors, scalars, ptype, 
        sim_box, interaction_params, d_pot_energy, its):
        its[0] += 1
        global_id, my_t = cuda.grid(2)  # type: ignore
        interactions_kernel(grid, vectors, scalars, ptype, sim_box, interaction_params)
        grid.sync()
        if global_id == 0 and my_t == 0:
            d_pot_energy[0] = 0
        grid.sync()
        if global_id < num_part and my_t == 0:
            cuda.atomic.add(d_pot_energy, 0, scalars[global_id][u_id])  # type: ignore
        grid.sync()

    @cuda.jit(device=gridsync)
    def add_step_in_dir(source, source_im, dir, step, dest, dest_im, sim_box):
        global_id, my_t = cuda.grid(2)  # type: ignore
        if global_id < num_part and my_t == 0:
            for k in range(num_dim):
                dest[k] = source[k] + dir[k] * step
                dest_im[k] = source_im[k]
            apply_PBC(dest, dest_im, sim_box)  # type: ignore

    return (
        get_dot_in_conf_space, 
        copy_positions_and_images,
        calculate_potential_energy,
        add_step_in_dir,
    )

