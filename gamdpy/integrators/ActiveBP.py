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
	
    """
    Integrator for Active Brownian Particles (ABP).

    This class implements the time evolution of active particles with
    translational and rotational diffusion as well as self-propulsion.

    Parameters
    ----------
    DT : float or array-like
        Translational diffusion coefficient (can depend on particle type).
    DR : float or array-like
        Rotational diffusion coefficient (can depend on particle type).
    mu : float or array-like
        Particle mobility (can depend on particle type).
    v0 : float or array-like
        Self-propulsion speed (can depend on particle type).
    dt : float
        Integration time step.
    seed : int
        Seed for the GPU random number generator.

    Methods
    --------
    get_params(configuration, interactions_params, verbose=False)
        Prepares integration parameters and RNG states for GPU execution.

    get_kernel(configuration, compute_plan, compute_flags, interactions_kernel, verbose=False)
        Builds the CUDA kernel used for time evolution of the system.

    save_internal_state(output, group_name)
        Placeholder for saving internal state (currently not implemented).
        
        
    Example
    --------
    configuration.randomize_orientations()
    integrator = gp.integrators.ActiveBP(DT=[5.0, 5.0], DR=[1.0, 1.0], mu=[0.0, 0.0], v0=[5.0, 5.0] , dt=0.005, seed=2028)

    Notes
    -----
    - Particles must have an orientation in this integrator. 
    -In the initializaton, make use of randomize_oreintations() to get random initial orientations.
    """
    

    def __init__(self, DT,DR,mu, v0, dt: float, seed = 0) -> None:
        self.DT = DT      
        self.DR = DR     
        self.mu = mu           
        self.v0 = v0         
        self.dt = dt           
        self.seed = seed        

    def get_params(self, configuration: Configuration, interactions_params: tuple, verbose=False) -> tuple:
        DT_sq = np.sqrt(np.array(self.DT, dtype= np.float32))
        DR_factor = np.sqrt(2  * np.array(self.DR, dtype= np.float32)*self.dt) 
        mu = np.array(self.mu,  dtype= np.float32)
        v0 = np.array(self.v0,  dtype= np.float32)
        dt = np.float32(self.dt)
        rng_states = create_xoroshiro128p_states(configuration.N, seed=self.seed)
	        

	        
        return (dt, DT_sq, DR_factor, mu, v0, rng_states)  
        
        

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
        

        r_id, f_id, n_id = (lambda d: (_ for _ in ()).throw(KeyError("Missing keys r/f/n")) if not all(k in d for k in ('r','f','n')) else [d[k] for k in ('r','f','n')])(configuration.vectors.indices)

        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
            dt, DT_sq, DR_factor, mu, v0, rng_states = integrator_params
            
            dt_sq = math.sqrt(dt)
            sqrt_2 = math.sqrt(numba.float32(2.0))
            
            global_id, my_t = cuda.grid(2)

            
            
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_f = vectors[f_id][global_id]
                my_n = vectors[n_id][global_id]
                my_type= ptype[global_id]
                #DT_sq = math.sqrt(DT[my_type]) #In the moment, we calculate DT_sq (and also the factor) for each particle. In fact, we would only need to 
                                               #calculate it for each type. 
                #factor= math.sqrt(2.0 * DR[my_type] * dt)
                

                for k in range(D):
                    xi = xoroshiro128p_normal_float32(rng_states, global_id)
                    
                    #evolve positions      
                    my_r[k] += sqrt_2 * DT_sq[my_type]*dt_sq*xi +mu[my_type]*my_f[k] *dt + (v0[my_type]*my_n[k])*dt
                    
                if D==3:    
	                gx = xoroshiro128p_normal_float32(rng_states, global_id)
	                gy = xoroshiro128p_normal_float32(rng_states, global_id)
	                gz = xoroshiro128p_normal_float32(rng_states, global_id)
	                    
	                #evolve the orientations                   
	                nx = my_n[0]
	                ny = my_n[1]
	                nz = my_n[2]
							  
	                dnx = DR_factor[my_type] * (gy * nz - gz * ny)
	                dny = DR_factor[my_type] * (gz * nx - gx * nz)
	                dnz = DR_factor[my_type] * (gx * ny - gy * nx)
							  
	                nx += dnx
	                ny += dny
	                nz += dnz
							  
	                norm = 1.0/ math.sqrt(nx*nx + ny*ny + nz*nz)
	                nx *= norm
	                ny *= norm
	                nz *= norm
							  
	                my_n[0] = nx
	                my_n[1] = ny
	                my_n[2] = nz
	                
                if D==2:
	                g = xoroshiro128p_normal_float32(rng_states, global_id)
	                dtheta = DR_factor * g

	                nx = my_n[0]
	                ny = my_n[1]
						
	                c = math.cos(dtheta)
	                s = math.sin(dtheta)
						
	                nx_new = c*nx - s*ny
	                ny_new = s*nx + c*ny
						
	                my_n[0] = nx_new
	                my_n[1] = ny_new
												  	  	  

                apply_PBC(my_r, r_im[global_id], sim_box)
               
            return

        step = cuda.jit(device=gridsync)(step)

        if gridsync:
            return step  # return device function
        else:
            return step[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters 

                



            