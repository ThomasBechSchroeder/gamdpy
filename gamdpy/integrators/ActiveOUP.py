import numpy as np
import numba
from numba import cuda
import math
import h5py
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
from ..configuration import Configuration
from .integrator import Integrator


class ActiveOUP(Integrator):
    r""" Active Ornstein-Uhlenbeck Particle (AOUP).

    Implementation of overdamped active Ornstein-Uhlenbeck dynamics.
    Active Ornstein-Uhlenbeck Particle, is an active matter model in which overdamped particles are subject to a colored Ornstein-Uhlenbeck noise,
    see Ref. [Fodor2016]_.

    The equations of motion:

    .. math::
        \dot{x_i} = -\mu f_i + \eta_i + \xi_i

    where :math:`\mu` is a mobility, and :math:`f` is an interaction potential.
    :math:`\xi` is an uncorrelated gauss distributed thermal noise:

    .. math::
        \langle \xi_{i_\alpha}(t)\xi_{j\beta}(t')\rangle = 2D_T \delta_{ij} \delta_{\alpha \beta} \delta(t-t')

    here :math:`D_T` is a diffusion coefficient, and Greek letters correspond to spatial components.
    :math:`\eta_i` are Ornstein-Uhlenbeck processes, solution of

    .. math::
            \dot{\mathbf{\eta}}_i = -\frac{\mathbf{\eta}_i}{\tau} + \frac{\sqrt{D_A}}{\tau} \mathbf{\zeta}_i

    with :math:`\langle \zeta_{i\alpha}(t) \zeta_{j\beta}(t')\rangle = 2\delta_{ij} \delta_{\alpha \beta} \delta(t-t')`
    
    The autocorrelation of :math:`\eta` then is

    .. math::
            \langle \eta_{i\alpha}(t)\eta_{j\beta}(t')\rangle= \delta_{ij} \delta_{\alpha \beta} \frac{D_A}{\tau} e^{\frac{-|t-t'|}{\tau}}
    
    so :math:`D_A` controls the amplitude of the noise and :math:`\tau` its persistence time.

    The equations are discretized using the Euler-Maruyama method [Higham2001]_ with a timestep :math:`dt`:

    .. math::
        \mathbf{r}_i(t+dt) = \mathbf{r}_i(t)+(\mu\mathbf{F}_i + \mathbf{\eta}_i)dt + \sqrt{2 D_T}\sqrt{dt} N(0,1)

    .. math::
        \mathbf{\eta}_i(t+dt) = \mathbf{\eta}_i(t)-\frac{\mathbf{\eta}_i}{\tau}dt + \frac{\sqrt{2 D_A}}{\tau} N(0,1)

    where :math:`N(0,1)` is a normal distributed pseudo random number.

    
    Parameters
    ----------
    DT: float
        Thermal diffusion coefficient
    DA: float
        "Active" diffusion coefficient
    mu: float
        mobility
    tau:float
        persistence time of the colored noise
    dt: float
        timestep
    seed:int, optional
        seed for pseudo random number generator

        
    References
    ----------

    .. [Fodor2016] Etienne Fodor et al.
       “How Far from Equilibrium Is Active Matter?”
       Phys. Rev. Lett. 117 (3 July 2016), p. 038103
       https://doi.org/10.1103/PhysRevLett.117.038103

    .. [Higham2001] Higham, D. J. (2001).
       "An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations."
       SIAM Review, 43(3), 525–546.
       https://doi.org/10.1137/S0036144500378302

    Examples
    --------
    >>> import gamdpy as gp
    >>> integrator = gp.integrators.ActiveOUP(DT=0.25, DA=0.5, mu=1.0, tau=0.75, dt=0.001, seed=0)

    """

    def __init__(self, DT: float,DA:float,mu:float, tau: float, dt: float, seed = 0) -> None:
        self.DT = DT      #sqrt of termal diffusion
        self.DA = DA      #sqrt of active diffusion 
        self.mu = mu            #mobility
        self.tau = tau          #persistence time of the colored noise
        self.dt = dt            #timestep
        self.seed = seed        

    def get_params(self, configuration: Configuration, interactions_params: tuple, verbose=False) -> tuple:
        DT = np.float32(self.DT)
        DA = np.float32(self.DA)
        mu = np.float32(self.mu)
        tau = np.float32(self.tau)
        dt = np.float32(self.dt)
        rng_states = create_xoroshiro128p_states(configuration.N, seed=self.seed)
        eta = np.zeros((configuration.N, configuration.D), dtype=np.float32)    #colored noise
        d_eta = cuda.to_device(eta)
        return (dt, DT, DA, mu, tau, rng_states, d_eta) # Needs to be compatible with unpacking in
                                                   # step() 

    def save_internal_state(self, output: h5py.File, group_name: str):
        pass


    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool], interactions_kernel, verbose=False):
        import math
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1


        if verbose:
            print(f'Generating Active Ornstein-Uhlenbeck integrator for {num_part} particles in {D} dimensions:')
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks * pb}')
            print(f'\tNumber of threads {num_blocks * pb * tp}')
        

        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        apply_PBC = numba.njit(configuration.simbox.get_apply_PBC())

        def step(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype):
            dt, DT, DA, mu, tau, rng_states, eta = integrator_params
            DT_sq = math.sqrt(DT)
            DA_sq = math.sqrt(DA)
            dt_sq = math.sqrt(dt)
            sqrt_2 = math.sqrt(numba.float32(2.0))
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_r = vectors[r_id][global_id]
                my_f = vectors[f_id][global_id]
                               


                for k in range(D):
                    xi = sqrt_2*xoroshiro128p_normal_float32(rng_states, global_id)        #white gaussian noise for position evolution
                    xi_noise = sqrt_2*xoroshiro128p_normal_float32(rng_states, global_id)  # the same for noise evolution

                    #evolve positions
                    my_r[k] += (eta[global_id,k]+mu*my_f[k])*dt + DT_sq*dt_sq*xi                   
                    #evolve noises
                    eta[global_id, k] += -eta[global_id, k]*dt/tau + DA_sq*dt_sq*xi_noise/tau

                apply_PBC(my_r, r_im[global_id], sim_box)
               
            return

        step = cuda.jit(device=gridsync)(step)

        if gridsync:
            return step  # return device function
        else:
            return step[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters 

                



            

