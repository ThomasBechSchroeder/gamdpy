import math

import numpy as np
import numba
from numba import cuda

from ..simulation.get_default_compute_plan import get_default_compute_plan

class CalculatorWidomInsertion:
    """ Calculator class for Widom's particle insertion method for computing chemical potentials

    This class is used to calculate the chemical potential of a
    not to dense fluid using Widom's particle insertion method.
    The excess chemical potential, μᵉˣ, fluid can be computed 
    as μᵉˣ = -kT ln 〈exp(-ΔU/kT)〉 where ΔU is the energy difference 
    between the system with and without a ghost particle.
    Here, 〈...〉 is an average for all possible positions 
    of the ghost particle (sometimes writte as an integral over space).

    The method should be used on a NVT trajectory in equlibrium.

    Parameters
    ----------

    configuration : gamdpy.Configuration
        The configuration object representing the system.

    pair_potential : gamdpy.PairPotential
        The pair potential object representing the interaction between particles.

    temperature : float
        The temperature of the system.

    ghost_positions : ndarray
        The positions of the ghost particles for which the chemical potential is to be calculated.

    ptype_ghost : int, optional
        The particle type of the ghost particles. Default is 0.

    compute_plan : dict, optional
        The compute plan for the system. Default is None (then a default plan is used).

    backend : str, optional
        The backend to use for the calculations. Default is 'GPU'. 
        Supported backends are 'CPU' (testing) and 'GPU' (recomended).
    
    Example
    -------
    >>> import numpy as np
    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()  # Replace with your own equbriliated simulation
    >>> pair_pot = sim.interactions[0]
    >>> num_ghost_particles = 500_000
    >>> ghost_positions = np.random.rand(num_ghost_particles, sim.configuration.D) * sim.configuration.simbox.get_lengths()
    >>> calc_widom = gp.CalculatorWidomInsertion(sim.configuration, pair_pot, sim.integrator.temperature, ghost_positions)
    >>> for block in sim.run_timeblocks():
    ...     calc_widom.update()
    >>> calc_widom_data = calc_widom.read()  # Dictionary with chemical potential and more
    >>> calc_widom_data.keys()
    dict_keys(['chemical_potential', 'boltzmann_factors', 'chemical_potentials'])
    >>> mu_ex = calc_widom_data['chemical_potential']  # Estimated excess chemical potential
    """

    KNOWN_BACKENDS = ['CPU', 'GPU']

    def __init__(self, configuration, pair_potential, temperature, ghost_positions, ptype_ghost=0, compute_plan=None, backend='GPU') -> None:
        if backend not in self.KNOWN_BACKENDS:
            raise ValueError(f'Unknown backend. Try one of {self.KNOWN_BACKENDS=}')

        self.configuration = configuration
        self.pair_potential = pair_potential
        self.temperature = np.float64(temperature)
        self.ghost_positions = np.array(ghost_positions, dtype=np.float32)
        self.ptype_ghost = ptype_ghost  # The ghost particle of this type
        self.num_updates = 0  # How many times have statistics been added to?
        self.chemical_potential = 0  # Current best estimate of the chemical potential
        self.chemical_potentials = []  # All estimates of the chemical potential

        # Expect that positions are in the same dimension as the configuration
        if len(self.ghost_positions.shape) != 2:
            raise ValueError('The ghost positions must be a 2D array (like [[x1, y1, z1], [x2, y2, z2], ...])')
        if self.ghost_positions.shape[1] != self.configuration.D:
            raise ValueError('The ghost positions must have the same spatial dimension as the configuration')

        self.compute_plan = compute_plan
        if self.compute_plan is None:
            self.compute_plan = get_default_compute_plan(configuration=configuration)

        self.backend = backend
        self.boltzmann_factors = np.zeros(len(self.ghost_positions), dtype=np.float64)
        self.boltzmann_factors_sum = np.zeros_like(self.boltzmann_factors)
        self.boltzmann_factors_timeblock = []
        
        # Make kernel
        if self.backend == 'GPU':
            self.update_kernel = self._make_updater_kernel()

    def _update_CPU(self):
        """ Return the boltzmann factors for the ghost particles in the current configuration (CPU backend). """
        pair_pot_func = self.pair_potential.pairpotential_function
        this_boltzmann_factors = np.zeros(len(self.ghost_positions), dtype=np.float32)
        for idx_ghost in range(len(self.ghost_positions)):
            ghost_pos = self.ghost_positions[idx_ghost]
            this_u = 0.0
            for n in range(self.configuration.N):
                ptype = self.configuration.ptype[n]
                r = self.configuration.vectors['r'][n]
                dr2 = self.configuration.simbox.dist_sq_function(r, ghost_pos, self.configuration.simbox.get_lengths())
                params = self.pair_potential.params[ptype, self.ptype_ghost]
                r_cut = params[-1]
                if dr2 < r_cut*r_cut:
                    dr = np.sqrt(dr2)
                    this_u += pair_pot_func(dr, params)[0]
            this_boltzmann_factors[idx_ghost] = math.exp(-this_u / self.temperature)
        return this_boltzmann_factors

    def _make_updater_kernel(self):
        """ Make the kernel for updating the radial distribution function (GPU backend). """
        # Unpack parameters from configuration and compute_plan
        D, num_part = self.configuration.D, self.configuration.N
        num_ghost_particles = len(self.ghost_positions)
        pb, tp, gridsync, UtilizeNIII = [self.compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_ghost_particles - 1) // pb + 1

        # Allocate to device
        self.d_ghost_positions = cuda.to_device(self.ghost_positions)
        self.d_ptype_ghost = np.int32(self.ptype_ghost)
        self.d_boltzmann_factors = cuda.to_device(self.boltzmann_factors_sum)
        self.d_temperature = cuda.to_device(self.temperature)

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [self.configuration.vectors.indices[key] for key in ['r', 'f']]
        
        # Prepare user-specified functions for inclusion in kernel(s)
        ptype_function = numba.njit(self.configuration.ptype_function)
        params_function = numba.njit(self.pair_potential.params_function)
        # pairpotential_function = numba.njit(self.pair_potential.pairpotential_function)
        pairpotential_function = self.pair_potential.pairpotential_function 
        dist_sq_function = numba.njit(self.configuration.simbox.get_dist_sq_function())

        def update_kernel(vectors, sim_box, ptype, params, ghost_positions, ptype_ghost, temperature, boltzmann_factors):
            global_id = cuda.grid(1)
            if global_id < len(ghost_positions):
                boltzmann_factors[global_id] = np.float64(0.0)

                this_u = np.float64(0.0)
                for other_global_id in range(0, num_part, 1):
                    dist_sq = dist_sq_function(vectors[r_id][other_global_id],
                                               ghost_positions[global_id], sim_box)
                    ij_params = params_function(ptype[other_global_id], ptype_ghost, params)
                    cut = ij_params[-1]
                    if dist_sq < cut*cut:
                        dist = math.sqrt(dist_sq)
                        pair_energy = pairpotential_function(dist, ij_params)[0]
                        this_u += np.float64(pair_energy)
                boltzmann_factors[global_id] = math.exp(-this_u/temperature)

            return

        return cuda.jit(device=False)(update_kernel)[num_blocks, (pb,)]

    def update(self):
        """ Update state the current configuration. """
        self.num_updates += 1
        this_boltzmann_factors = None
        if self.backend == 'CPU':
            this_boltzmann_factors = self._update_CPU()
        elif self.backend == 'GPU':
            self.update_kernel(self.configuration.d_vectors,
                            self.configuration.simbox.d_data,
                            self.configuration.d_ptype,
                            self.pair_potential.d_params,
                            self.d_ghost_positions,
                            self.d_ptype_ghost,
                            self.temperature,
                            self.d_boltzmann_factors)
            this_boltzmann_factors = self.d_boltzmann_factors.copy_to_host()
        else:
            raise ValueError(f'Unknown backend. Try one of {self.KNOWN_BACKENDS=}')
        self.boltzmann_factors_sum += this_boltzmann_factors
        self.boltzmann_factors = self.boltzmann_factors_sum/self.num_updates
        self.chemical_potential = -self.temperature*math.log(np.mean(self.boltzmann_factors))
        this_chemical_potential = -self.temperature*math.log(np.mean(this_boltzmann_factors))
        self.chemical_potentials.append(this_chemical_potential)

    def read(self):
        """ Read data
        
        Return the current chemical potential, 
        average Boltzmann factors, and timeblock-specific chemical potentials for the ghost particles.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            
            - 'chemical_potential': float
                The best estimate of the overall chemical potential for the system.
                
            - 'boltzmann_factors': ndarray
                The average Boltzmann factors for each ghost particle, representing the statistical weights based on their interaction energies.
                
            - 'chemical_potentials': ndarray
                An array of estimated chemical potentials for each timeblock, providing insight into the variability of the chemical potential over time.
        """
        
        return {
            'chemical_potential': self.chemical_potential,  # Best estimate of the chemical potential
            'boltzmann_factors': self.boltzmann_factors,    # Average Boltzmann factors for the ghost particles
            'chemical_potentials': self.chemical_potentials # Estimates of the chemical potentials for each timeblock
        }

