import numpy as np
import numba
import math, cmath
from numba import cuda
import matplotlib.pyplot as plt
import gamdpy as gp
from .interaction import Interaction

class Electrostatics(Interaction):
    r"""Electrostatic Coulomb interactions. 
    Deal with standard Ewald sums and Wolff method.

    Standard methods (off-lattice) methods to deal with electrostatics in MD rely on 
    a Gaussian screened potential, controlled by a damping parameter :math:`\kappa`
    (cf. gaussian_screened_coulomb in potential_functions). 

    The simple Wolff method approximates long-range Coulomb interactions
    with such screened potential. The latter also correponds to the real-space summation of the Ewald method. 

    The compensating long-range potential can be efficiently computed in reciprocal space
    (cf. Allen & Tildesley eq. 6.6 page 220 second edition):

    .. math::
        U_\mathrm{reciprocal}=\frac{1}{2}\sum_i q_i\sum_{\mathbf{k}\neq 0}G(k)\rho(\mathbf{k})\exp(i\mathbf{k}\cdot\mathbf{r}_i),
    
    where :math:`\mathbf{k}=\frac{2\pi}{L}\mathbf{n}` is an element of the dual space
    of the box simulation, :math:`G(k)=\frac{2\pi}{V}\frac{\exp(-k^2/4\kappa^2)}{k^2}` is the convoluting 
    kernel of the Poisson equation, and :math:`\rho(\mathbf{k})=\sum_j q_j\exp(-i\mathbf{k}\cdot\mathbf{r}_j)`. 

    The force on atom i subsequently follows as

    .. math::
        \mathbf{f}^i_\mathrm{reciprocal}=-q_i\sum_\mathbf{k\neq 0}\mathbf{k}G(k)\Im(\rho(-\mathbf{k})\exp(-i\mathbf{k}\cdot{r}_j)).

    Note 1: The reciprocal contribution to the potential can be expressed as a single sum over wave-vectors
    but because the force requires both particle and wave-vector indices, we kept this form
    to take advantage of our current threading choice.

    Note 2: There is a self-energy that must be substracted due to the screening procedure
    (see function compute_self_energy) but it is a constant that must be computed only once
    and does not depend on the Configuration. You can add it in post-processing for
    benchmarking purposes.

    Parameters
    ----------
    damping : float
        Decay rate of the electrostatic gaussian screening. 
        If 0, the normal Coulomb potential is used.

    cutoff : nested list of floats
        Real-space cutoff associated to pair-wise interactions between charges classes.
        Rows and columns MUST be sorted in ascending charges.
        :TODO: Add a check of this.

    max_num_nbs : int
        Maximum number of neighbors per particle to allocate in the neighbor list.

    exclusions : array_like
        List of particle indices to exclude from interactions for each particle.
    """

    def __init__(self, damping, cutoff, max_num_nbs, exclusions=None):
        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]
            return result
        self.params_function = params_function
        self.damping = numba.float32(damping)
        self.cutoff = cutoff
        self.exclusions = exclusions 
        self.max_num_nbs = max_num_nbs

        if self.damping != 0.0:
            self.real_space_pot = gp.apply_shifted_potential_cutoff(gp.gaussian_screened_coulomb)
        else:
            self.real_space_pot = gp.apply_shifted_force_cutoff(gp.make_IPL_n(n=1))

        self.ewald = False

    def set_ewald(self, ncut):
        '''
        Compute Ewald sums on top of the Wolff method.

        Parameters
        ----------
        ncut : int
            Given a k-point in a single direction 2pi/L * n, this sets the maximum absolue value of n. 
        '''
        self.ncut = ncut
        self.ewald = True

    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:
        # Gathering charges properties
        self.charges, self.charged_idx = configuration.get_charged_particles()
        coulomb_matrix, unique_charges, self.charges_types = self.build_pair_coulomb_matrix(self.charges)

        if self.damping == 0.0:
            params = [coulomb_matrix, self.cutoff]
        else:
            # Need to change this: decay rate is not a type-of-pairs quantity
            # Keeping it like that atm because it fits the current params format
            decay_rate = np.full_like(coulomb_matrix, self.damping, dtype=np.float32)
            params = [coulomb_matrix, decay_rate, self.cutoff]

        # Formatting params for kernels
        self.params, max_cut = self.format_pot_params(params)

        # Building reciprocal space attributes
        if self.ewald:
            kpoints = self.gen_k_grid(self.ncut, configuration.simbox.get_lengths())
            poisson = self.compute_poisson_grid(kpoints, self.damping, configuration.get_volume())
            # Sorting to compute strongest poisson points first
            new_order = np.flip(np.argsort(poisson))
            kpoints, poisson  = [x[new_order] for x in [kpoints, poisson]]
            # Filter out kpoints not giving any contribution with single precision
            purge_zeroes = (poisson != 0.0)
            self.kpoints, self.poisson = [x[purge_zeroes] for x in [kpoints, poisson]]
            self.self_energy = self.compute_self_energy(self.charges, self.damping)

            self.num_kpoints = len(self.kpoints)
            self.real_fourier_density = np.zeros_like(self.poisson, dtype=np.float32)
            self.imag_fourier_density = np.zeros_like(self.poisson, dtype=np.float32)

        self.copy_to_device()

        # Deal with neighbours lists
        if compute_plan['nblist'] == 'N squared':
            self.nblist = gp.NbList2(configuration, self.exclusions, self.max_num_nbs)
        elif compute_plan['nblist'] == 'linked lists':
            self.nblist = gp.NbListLinkedLists(configuration, self.exclusions, self.max_num_nbs)
        else:
            raise ValueError(f"No lblist called: {compute_plan['nblist']}. Use either 'N squared' or 'linked lists'")
        nblist_params = self.nblist.get_params(max_cut, compute_plan, verbose)

        if self.ewald:
            return (
                self.d_params,
                self.d_charges,
                self.d_charged_idx,
                self.d_charges_types,
                self.d_kpoints,
                self.d_poisson,
                self.d_real_fourier_density,
                self.d_imag_fourier_density,
                self.nblist.d_nblist, 
                nblist_params
            )
        else:
            return (
                self.d_params,
                self.d_charges,
                self.d_charged_idx,
                self.d_charges_types,
                self.nblist.d_nblist,
                nblist_params
            )

    def copy_to_device(self):
        self.d_params = cuda.to_device(self.params)
        self.d_charges = cuda.to_device(self.charges)
        self.d_charged_idx = cuda.to_device(self.charged_idx)
        self.d_charges_types = cuda.to_device(self.charges_types)
        if self.ewald:
            self.d_kpoints = cuda.to_device(self.kpoints)
            self.d_poisson = cuda.to_device(self.poisson)
            self.d_real_fourier_density = cuda.to_device(self.real_fourier_density)
            self.d_imag_fourier_density = cuda.to_device(self.imag_fourier_density)

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        num_cscalars = configuration.num_cscalars

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']

        # Unpack parameters
        D, N = configuration.D, configuration.N
        num_kpoints = self.num_kpoints
        self_energy = self.self_energy
        num_charged = len(self.charged_idx)
        damping = self.damping
        vol = configuration.get_volume()

        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        if gridsync:
            num_part = N
        else:
            num_part = num_charged
        num_blocks = (num_part - 1) // pb + 1 

        if verbose:
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks*pb}')
            print(f'\tNumber of threads {num_blocks*pb*tp}')
    
        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        if compute_u:
            u_id = configuration.sid['U']
        if compute_w:
            w_id = configuration.sid['W']
        if compute_lap:
            lap_id = configuration.sid['lapU']

        real_space_pot = self.real_space_pot

        virial_factor = numba.float32( 0.5/configuration.D )
        def real_space_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
            u, s, umm = real_space_pot(ij_dist, ij_params)
            half = numba.float32(0.5)
            for k in range(D):
                my_f[k] = my_f[k] - dr[k]*s                             # Force
                if compute_w:
                    cscalars[w_id] += dr[k]*dr[k]*s*virial_factor       # Virial
            if compute_u:
                cscalars[u_id] += half*u                                # Potential energy
            if compute_lap:
                cscalars[lap_id] += numba.float32(1-D)*s + umm          # Laplacian 
                return

        diel_pref = numba.float32(2.0) * numba.float32(math.pi) / (numba.float32(3.0) * vol)
        def add_dielectric_drift(ri, rj, qiqj, my_f, cscalars):
            two = numba.float32(2.0)
            for d in range(D):
                my_f[d] = my_f[d] - two * diel_pref * qiqj * rj[d]
                if compute_u:
                    cscalars[u_id] += diel_pref * qiqj * ri[d] * rj[d]

        def fourier_space_calculator(r, qi, kpoint, poisson_k, real_rho_k, imag_rho_k, my_f, cscalars):
            # Helper variables
            dot_rk = numba.float32(0.0)
            two = numba.float32(2.0)
            for d in range(D):
                dot_rk = dot_rk + r[d] * kpoint[d]
            cos_rk = math.cos(dot_rk)
            sin_rk = math.sin(dot_rk)
            real_cross_k = real_rho_k * cos_rk - imag_rho_k * sin_rk
            imag_cross_k = imag_rho_k * cos_rk + real_rho_k * sin_rk

            for d in range(D):
                my_f[d] = my_f[d] + two * qi * kpoint[d] * poisson_k * imag_cross_k
            if compute_u:
                cscalars[u_id] += qi * poisson_k * real_cross_k
            return

        params_function = numba.njit(self.params_function)
        real_space_calculator = numba.njit(real_space_calculator)
        add_dielectric_drift = numba.njit(add_dielectric_drift)
        fourier_space_calculator = numba.njit(fourier_space_calculator)
        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())

        @cuda.jit( device=gridsync )
        def sum_real_space(vectors, cscalars, sim_box, charges_idx, charges_types, nblist, params):
            """ 
            Sum real-space Ewald contributions.
            """

            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y

            max_nbs = nblist.shape[1]-1

            my_f = cuda.local.array(shape=D,dtype=numba.float32)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)
            my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)

            if global_id < num_charged:
                for k in range(D):
                    #my_r[k] = vectors[r_id][global_id,k]
                    my_f[k] = numba.float32(0.0)
                for k in range(num_cscalars):
                    my_cscalars[k] = numba.float32(0.0)

            cuda.syncthreads() # Make sure initializing global variables to zero is done

            if global_id < num_charged:
                part_id = charges_idx[global_id]
                my_charge_type = charges_types[global_id]
                for i in range(my_t, nblist[part_id, max_nbs], tp):
                    other_part_id = nblist[part_id, i]
                    other_charge_type = charges_types[other_part_id] # this will only work if every particle has a charge
                    ij_params = params_function(my_charge_type, other_charge_type, params)
                    qiqj = ij_params[0]
                    # add_dielectric_drift(vectors[r_id][part_id], vectors[r_id][other_part_id], qiqj, my_f, my_cscalars)
                    dist_sq = dist_sq_dr_function(vectors[r_id][other_part_id], vectors[r_id][part_id], sim_box, my_dr)
                    cut = ij_params[-1]
                    if dist_sq < cut*cut:
                        real_space_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, 0, vectors[f_id], other_part_id)
                    
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (part_id, k), my_f[k])
   
                for k in range(num_cscalars):
                    cuda.atomic.add(cscalars, (part_id, k), my_cscalars[k])

            return 

        @cuda.jit( device=gridsync )  
        def sum_fourier_space(
            vectors,
            cscalars,
            charges,
            charges_idx,
            kpoints,
            poisson_grid,
            real_fourier_density,
            imag_fourier_density
        ):
            """ 
            Sum reciprocal-space Ewald contributions.
            """

            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y

            my_f = cuda.local.array(shape=D, dtype=numba.float32)
            my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)

            if global_id < num_charged:
                for d in range(D):
                    my_f[d] = numba.float32(0.0)
                for s in range(num_cscalars):
                    my_cscalars[s] = numba.float32(0.0)
                # if compute_u:
                #     my_cscalars[u_id] -= self_energy

            cuda.syncthreads()

            if global_id < num_charged:
                part_id = charges_idx[global_id]
                my_q = charges[global_id]
                my_r = vectors[r_id][part_id]
                for k_id in range(my_t, num_kpoints, tp):
                    kpoint = kpoints[k_id]
                    poisson_k = poisson_grid[k_id]
                    real_rho_k = real_fourier_density[k_id]
                    imag_rho_k = imag_fourier_density[k_id]
                    fourier_space_calculator(my_r, my_q, kpoint, poisson_k, real_rho_k, imag_rho_k, my_f, my_cscalars)

                for d in range(D):
                    cuda.atomic.add(vectors[f_id], (part_id, d), my_f[d])

                for s in range(num_cscalars):
                    cuda.atomic.add(cscalars, (part_id, s), my_cscalars[s])

            return 

        @cuda.jit
        def init_fourier_density(real_fourier_density, imag_fourier_density):
            """
            Zeroing the fourier transform of the charge density at the start of each step.
            """
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            my_t = cuda.threadIdx.y

            tid_in_block = my_t * pb + local_id # mapping from 2d threads to 1d threads
            threads_per_block = pb * tp

            global_id  = my_block * threads_per_block + tid_in_block # global thread id
            step = num_blocks * threads_per_block

            for k_id in range(global_id, num_kpoints, step):
                real_fourier_density[k_id] = numba.float32(0.0)
                imag_fourier_density[k_id] = numba.float32(0.0)

            return 

        @cuda.jit( device=gridsync )  
        def update_fourier_density(
            vectors,
            charges,
            charges_idx,
            kpoints,
            real_fourier_density,
            imag_fourier_density
        ):
            """
            Update the fourier transform of the charge density after the zeroing.
            """
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y

            if global_id < num_charged:
                part_id = charges_idx[global_id]
                my_q = charges[global_id]
                my_r = vectors[r_id][part_id]
                for k_id in range(my_t, num_kpoints, tp):
                    kpoint = kpoints[k_id]
                    dot_rk = numba.float32(0.0)
                    for d in range(D):
                        dot_rk = dot_rk + my_r[d] * kpoint[d]
                    real_rho = my_q * math.cos(dot_rk)
                    imag_rho = -my_q * math.sin(dot_rk)
                    cuda.atomic.add(real_fourier_density, k_id, real_rho)
                    cuda.atomic.add(imag_fourier_density, k_id, imag_rho)

            return 

        nblist_check_and_update = self.nblist.get_kernel(configuration, compute_plan, compute_flags, verbose)

        if gridsync:
            # A device function, 
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                (params, charges, charged_idx, charges_types, kpoints, poisson_grid, real_fourier_density, imag_fourier_density, nblist, nblist_parameters
                ) = interaction_parameters
                init_fourier_density(real_fourier_density, imag_fourier_density)
                grid.sync()
                update_fourier_density(
                    vectors,
                    charges,
                    charged_idx,
                    kpoints,
                    real_fourier_density,
                    imag_fourier_density
                )
                grid.sync()
                sum_fourier_space(
                    vectors,
                    scalars,
                    charges,
                    charged_idx,
                    kpoints,
                    poisson_grid,
                    real_fourier_density,
                    imag_fourier_density
                )
                grid.sync()
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                grid.sync()
                sum_real_space(
                    vectors,
                    scalars,
                    sim_box,
                    charged_idx,
                    charges_types,
                    nblist,
                    params
                )
                return
            return compute_interactions

        else:
            # A python function, 
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                (params, charges, charged_idx, charges_types, kpoints, poisson_grid, real_fourier_density, imag_fourier_density, nblist, nblist_parameters
                ) = interaction_parameters
                init_fourier_density[num_blocks, (pb, tp)](real_fourier_density, imag_fourier_density)
                update_fourier_density[num_blocks, (pb, tp)](
                    vectors,
                    charges,
                    charged_idx,
                    kpoints,
                    real_fourier_density,
                    imag_fourier_density
                )
                sum_fourier_space[num_blocks, (pb, tp)](
                    vectors,
                    scalars,
                    charges,
                    charged_idx,
                    kpoints,
                    poisson_grid,
                    real_fourier_density,
                    imag_fourier_density
                )
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                sum_real_space[num_blocks, (pb, tp)](
                    vectors,
                    scalars,
                    sim_box,
                    charged_idx,
                    charges_types,
                    nblist,
                    params
                )
                return
            return compute_interactions

    @staticmethod
    def compute_self_energy(charges, kappa):
        r"""
        Compute the self-energy associated to the screening procedure:

        .. math::
            \frac{\kappa}{N\sqrt{\pi}}\sum_i q_i^2.

        Note that the self-energy is divided by N to make it a per-atom quantity.
        This quantity does not depend on the state of the system and can be computed 
        only once. It only has benchmarking purposes.

        Parameters
        ----------
        charges : numpy array
            Assigning a charge to each particle

        kappa : float
            Decay rate of the electrostatic gaussian screening. 

        Returns
        -------
        self_energy : numpy array
            Per-atom self-energy associated to the screening procedure.
        """

        N = len(charges)
        pref = numba.float32((kappa / (N * math.sqrt(math.pi))))
        return pref * np.sum(charges**2)

    @staticmethod
    def build_pair_coulomb_matrix(charges):
        """
        Build the matrix of all unique charges-pair products.

        Parameters
        ----------
        charges : numpy array
            Assigning a charge to each particle

        Returns
        -------
        coulomb_matrix : numpy array
            Matrix of all unique q_i*q_j products

        unique_charges : numpy array
            Sorted array of unique charges composing the coulomb_matrix

        charges_types : numpy array
            Charge class associated to each particle.
            A particle with class i and another with class j refer to the (ij) entry
            of the coulomb_matrix.
        """
        unique_charges, charges_types = np.unique(charges, return_inverse=True)
        coulomb_matrix = np.outer(unique_charges, unique_charges)
        return coulomb_matrix, unique_charges, charges_types

    @staticmethod
    def gen_k_grid(ncut, box_size):
        """
        Gen a k-point grid taking into account the (k,-k) symmetry.

        :TODO: Generalize for any dimension. CURRENTLY DESIGNED FOR D=3.
        """
        # Building complete mesh
        nx, ny = [np.arange(-ncut, ncut+1) for _ in range(2)]
        nz = np.arange(0, ncut+1)
        M = np.stack(np.meshgrid(nx, ny, nz, indexing='ij'), axis=-1).reshape(-1, 3)

        # drop k=0
        M = M[np.any(M != 0, axis=1)]

        # Discard half of the k-points
        on_plane = (M[:,2] == 0) # nz = 0
        keep = (M[:,2] > 0) | (on_plane & (M[:,1] > 0)) | (on_plane & (M[:,1] == 0) & (M[:,0] > 0))
        # (nz != 0) or (nz=0 and ny>0) or (nz=0 and ny = 0 and nx > 0)
        M = M[keep]

        # Only consider those in kcut sphere
        norm_M = np.linalg.norm(M, axis=-1)
        M = M[norm_M <= ncut]

        k_points = (2.0 * np.pi * M) / box_size
        return k_points.astype(np.float32)

    @staticmethod
    def compute_poisson_grid(k_points, kappa, volume):
        """
        Compute the Poisson equation convolution kernel over a whole k-point grid.
        """
        # Helper variables
        four = numba.float32(4.0)
        kappa2 = kappa * kappa
        k2 = np.linalg.norm(k_points, axis=-1)**2
        poisson = four * numba.float32(math.pi) * np.exp(-k2 / (four * kappa2)) / (k2 * volume)
        return poisson

    @staticmethod
    def format_pot_params(params_):
        num_classes = params_[0].shape[0]
        num_params = len(params_)

        params = np.zeros((num_classes, num_classes), dtype="f,"*num_params)
        for i in range(num_classes):
            for j in range(num_classes):
                plist = []
                for parameter in params_:
                    plist.append(parameter[i,j])
                params[i,j] = tuple(plist)

        max_cut = np.float32(np.max(params_[-1]))

        return params, max_cut