import numpy as np
import numba
from numba import cuda
import math
import h5py
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
from ..configuration import Configuration
from .integrator import Integrator


class ActiveBP(Integrator):
    

    def __init__(self, DT: float,DR:float,mu:float, v0: float, dt: float, seed = 0) -> None:
        self.DT = DT      #sqrt of termal diffusion
        self.DR = DR     #sqrt of active diffusion 
        self.mu = mu            #mobility
        self.v0 = v0          #persistence time of the colored noise
        self.dt = dt            #timestep
        self.seed = seed        

    def get_params(self, configuration: Configuration, interactions_params: tuple, verbose=False) -> tuple:
        DT = np.float32(self.DT)
        DR = np.float32(self.DR)
        mu = np.float32(self.mu)
        v0 = np.float32(self.v0)
        dt = np.float32(self.dt)
        rng_states = create_xoroshiro128p_states(configuration.N, seed=self.seed)
        orientations = np.zeros((configuration.N, configuration.D), dtype=np.float32)
        phi = np.random.uniform(0, 2*np.pi, configuration.N)
        u   = np.random.uniform(-1, 1, configuration.N)
        s = np.sqrt(1 - u*u)
        orientations[:,0] = s*np.cos(phi)
        orientations[:,1] = s*np.sin(phi)
        orientations[:,2] = u
        d_orientations = cuda.to_device(orientations)
        return (dt, DT, DR, mu, v0, rng_states, d_orientations)  
        
        

    def save_internal_state(self, output: h5py.File, group_name: str):
        pass


    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool], interactions_kernel, verbose=False):
        import math
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1


        if verbose:
            print(f'Generating Active Brownian integrator for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')
        

        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
            dt, DT, DR, mu, v0, rng_states, orientations = integrator_params
            DT_sq = math.sqrt(DT)
            dt_sq = math.sqrt(dt)
            sqrt_2 = math.sqrt(numba.float32(2.0))
            
            factor = math.sqrt(2.0 * DR * dt)
            global_id, my_t = cuda.grid(2)

            
            
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_f = vectors[f_id][global_id]
                my_n = orientations[global_id]

                for k in range(D):
                    xi = xoroshiro128p_normal_float32(rng_states, global_id)
            
                    #evolve positions      
                    my_r[k] += sqrt_2 * DT_sq*dt_sq*xi +mu*my_f[k] *dt + (v0*my_n[k])*dt
                    
                gx = xoroshiro128p_normal_float32(rng_states, global_id)
                gy = xoroshiro128p_normal_float32(rng_states, global_id)
                gz = xoroshiro128p_normal_float32(rng_states, global_id)
                    
                #evolve the orientations                   
                nx = orientations[global_id, 0]
                ny = orientations[global_id, 1]
                nz = orientations[global_id, 2]
						  
                dnx = factor * (gy * nz - gz * ny)
                dny = factor * (gz * nx - gx * nz)
                dnz = factor * (gx * ny - gy * nx)
						  
                nx += dnx
                ny += dny
                nz += dnz
						  
                norm = math.sqrt(nx*nx + ny*ny + nz*nz)
                nx /= norm
                ny /= norm
                nz /= norm
						  
                orientations[global_id, 0] = nx
                orientations[global_id, 1] = ny
                orientations[global_id, 2] = nz
						  
						  	  

                apply_PBC(my_r, r_im[global_id], sim_box)
               
            return

        step = cuda.jit(device=gridsync)(step)

        if gridsync:
            return step  # return device function
        else:
            return step[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters 

                



            