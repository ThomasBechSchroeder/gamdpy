import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt
import gamdpy as gp
from .interaction import Interaction

class EAM_ZJW_2004(Interaction):
    r"""  Embedded atom many-body potential for alloys involving 16 elements as published by Zhou, Johnson, Wadley, Phys. Rev. B 2004.

    Parameters
    ----------
    
    params : list of floats or nested list of floats
        Interaction parameters for the pair potential function. Use a nested list for multiple types of particles.
        The last element of each list is the cutoff distance of the pair potential.
    max_num_nbs : int
        Maximum number of neighbors per particle to allocate in the neighbor list.
    exclusions : array_like
        List of particle indices to exclude from interactions for each particle.

    References
    ----------

        [Zhou2004]X. W. Zhou, R. A. Johnson, H. N. G. Wadley,
        "Misfit-energy-increasing dislocations in vapor-deposited CoFe/NiFe multilayers"
        Phys. Rev. B 69, 144113 (2004)
        https://doi.org/10.1103/PhysRevB.69.144113

    Examples
    --------

    Example of creaating a potential for pure copper.

    >>> paramsCu = gp.EAM_ZJW_2004_params['Cu']
    >>> cut = math.sqrt(5.) * paramsCu[0] # the original paper doesn't specify the actual value of the cutoff to use'
    >>> paramsCu = np.append(paramsCu, cut)
    >>> eam_pot = gp.EAM_ZJW_2004([paramsCu], max_num_nbs=1000) # for alloys just include other elements' parameters in the list


    """

    def __init__(self, params, max_num_nbs, exclusions=None):


        self.params_user = params
        self.exclusions = exclusions 
        self.max_num_nbs = max_num_nbs

    def convert_user_params(self):
        num_types = len(self.params_user)
        num_params = len(self.params_user[0])
        assert num_params == 21

        # first recreate the list of arrays but with dtype=float32
        params_list = []
        cutoff_list = []
        for parameter in self.params_user:
            assert len(parameter.shape) == 1
            assert parameter.shape[0] == num_params
            params_list.append(np.array(parameter, dtype=np.float32))
            cutoff_list.append(parameter[-1])

        # Convert params to the format required by kernels  num_types-array of tuples (p0, p1, ..., cutoff (???))
        params = np.zeros((num_types,), dtype="f,"*num_params)
        for i in range(num_types):
            plist = []
            for parameter in self.params_user[i]:
                plist.append(parameter)
            params[i] = tuple(plist)


        max_cut = np.float32(np.max(np.array(cutoff_list)))
        return params, max_cut



    def check_datastructure_validity(self) -> bool:
        nbflag = self.nblist.d_nbflag.copy_to_host()
        if nbflag[0] != 0 or nbflag[1] != 0:
            raise RuntimeError(f'Neighbor-list is invalid. Try allocating space for more neighbors (max_num_nbs in PairPot object). Allocated size: {self.max_num_nbs}, but {nbflag[1]+1} neighbours found. {nbflag=}.')
        return True

    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:
        
        self.params, max_cut = self.convert_user_params()
        self.d_params = cuda.to_device(self.params)

        N = configuration.N
        self.electron_density = np.zeros(N, np.float32)
        self.embedding_energy_and_grad = np.zeros((N, 3), np.float32)
        
        self.d_electron_density = cuda.to_device(self.electron_density)
        self.d_embedding_energy_and_grad = cuda.to_device(self.embedding_energy_and_grad)
        
        if compute_plan['nblist'] == 'N squared':
            self.nblist = gp.NbList2(configuration, self.exclusions, self.max_num_nbs)
        elif compute_plan['nblist'] == 'linked lists':
            self.nblist = gp.NbListLinkedLists(configuration, self.exclusions, self.max_num_nbs)
        else:
            raise ValueError(f"No lblist called: {compute_plan['nblist']}. Use either 'N squared' or 'linked lists'")
        nblist_params = self.nblist.get_params(max_cut, compute_plan, verbose)

        return (self.d_params, self.nblist.d_nblist, nblist_params, self.d_electron_density, self.d_embedding_energy_and_grad)

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        num_cscalars = configuration.num_cscalars

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']
        compute_stresses = compute_flags['stresses']

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_part - 1) // pb + 1  

        if verbose:
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks*pb}')
            print(f'\tNumber of threads {num_blocks*pb*tp}')
            if compute_stresses:
                print('\tIncluding computation of stress tensor in pair potential')
        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        if compute_u:
            u_id = configuration.sid['U']
        if compute_w:
            w_id = configuration.sid['W']
        if compute_lap:
            lap_id = configuration.sid['lapU']

        if compute_stresses:
            sx_id = configuration.vectors.indices['sx']
            if D > 1:
                sy_id = configuration.vectors.indices['sy']
                if D > 2:
                    sz_id = configuration.vectors.indices['sz']
                    if D > 3:
                        sw_id = configuration.vectors.indices['sw']

        def electron_density_function(dist, params):
            # parameters needed: r_e, f_e, beta, lambda
            r_e = params[0]
            f_e = params[1]
            beta = params[5]
            lamb = params[9]
            r_sc = dist/r_e
            one = numba.float32(1.0)
            nineteen = numba.float32(19.)
            twenty = numba.float32(20.)
            pow19 = math.pow(r_sc - lamb, 19)
            pow20 = math.pow(r_sc - lamb, 20)
            f = f_e * math.exp(-beta*(r_sc - one)) / (one + pow20)
            f_p = - f * (beta + twenty*pow19 / (one + pow20) )/r_e # 1st derivative
            f_pp = -f_p*(beta + (twenty*pow19)/(one+pow20)) - f*( (twenty*nineteen*math.pow(r_sc - lamb, 18))/(one+pow20) - ((twenty*pow19)/(one+pow20))**2)/r_e # second derivative
            f_pp /= r_e
            # instead of f_p, it is more useful to return -f_p/r (corresponding to the variable 's' in the context of pair potentials)
            return f, -f_p/dist, f_pp

        assert UtilizeNIII == False # FOR NOW (?)

        virial_factor = numba.float32( 1.0/configuration.D )
        virial_factor_half = numba.float32( 0.5/configuration.D )

        electron_density_function = numba.njit(electron_density_function)
        ptype_function = numba.njit(configuration.ptype_function)
        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
        dist_sq_function = numba.njit(configuration.simbox.get_dist_sq_function())
    
        @cuda.jit( device=gridsync )
        def calc_electron_density_and_embedding_energy(vectors, cscalars, ptype, sim_box, nblist, params, elec_dens, embed_en_grad):
            """
            Calculate electron density as pair sum and store in rho array in scalars
            """
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y
            zero = numba.float32(0.)
            one = numba.float32(1.)
            two = numba.float32(2.)
            max_nbs = nblist.shape[1]-1
            if global_id < num_part:
                elec_dens[global_id] = zero

            cuda.syncthreads() # necessary (I think) so nobody is setting elec_dens for this particle to zero after other threads have begun to add to it
            my_type = ptype_function(global_id, ptype)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)

            if global_id < num_part:
                for i in range(my_t, nblist[global_id, max_nbs], tp):
                    other_id = nblist[global_id, i]
                    other_type = ptype_function(other_id, ptype)
                    params_other_type = params[other_type]
                    dist_sq = dist_sq_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box)
                    cut = params_other_type[-1]
                    if dist_sq < cut*cut:
                        rho, rho_s, rho_pp = electron_density_function(math.sqrt(dist_sq), params_other_type)
                        #print(global_id, other_id, math.sqrt(dist_sq), rho, rho_s, rho_pp)
                        cuda.atomic.add(elec_dens, global_id, rho)

            cuda.syncthreads() # synchronize to ensure that all threads updating the electron density for a given particle have finished
            # now can calculate the embedding energy for this particle
            params_my_type = params[my_type]
            if global_id < num_part and my_t == 0:
                rho = elec_dens[global_id]
                rho_e = params_my_type[2]
                rho_s = params_my_type[3]
                rho_n = numba.float32(0.85)*rho_e
                rho_0 = numba.float32(1.15)*rho_e
                Fn0, Fn1, Fn2, Fn3 = params_my_type[10], params_my_type[11], params_my_type[12], params_my_type[13]
                F0, F1, F2, F3 = params_my_type[14], params_my_type[15], params_my_type[16], params_my_type[17]
                eta = params_my_type[18]
                F_e = params_my_type[19]

                if rho < rho_n:
                    rrn1 = rho/rho_n-one
                    F_rho = Fn0 + Fn1*rrn1 + Fn2 * rrn1**2 + Fn3*rrn1**3
                    F_prime = (Fn1 + 2*Fn2 * rrn1 + 3*Fn3*rrn1**2)/rho_n
                    F_primeprime = (2*Fn2 + 6*Fn3*rrn1)/rho_n**2
                elif rho < rho_0:
                    rre1 = rho/rho_e-one
                    F_rho = F0 + F1*rre1 + F2 * rre1**2 + F3*rre1**3
                    F_prime = (F1 + 2*F2 * rre1 + 3*F3*rre1**2)/rho_e
                    F_primeprime = (2*F2 + 6*F3*rre1)/rho_e**2
                else:
                    rrs = rho/rho_s
                    F_rho = F_e * (one-eta*math.log(rrs))*pow(rrs, eta)
                    F_prime = -F_e*eta**2*pow(rrs, eta-one)*math.log(rrs) / rho_s
                    F_primeprime = -F_e*eta**2*pow(rrs, eta-two) * (one + (eta-one) * math.log(rrs)) / rho_s**2
                embed_en_grad[global_id, 0] = F_rho # maybe not needed if we just write to the energy array now?
                embed_en_grad[global_id, 1] = F_prime
                embed_en_grad[global_id, 2] = F_primeprime
                # We can write the embedding part of the potential energy to the global array already now
                cscalars[global_id, u_id] = F_rho

        # Should I jit it after defining it?
        @cuda.jit( device = gridsync )
        def pair_contribution(dist, params):
            r_e = params[0]
            alpha = params[4]
            beta = params[5]
            A = params[6]
            B = params[7]
            kappa = params[8]
            lamb = params[9]
            one = numba.float32(1.0)
            r_sc = dist / r_e

            nineteen = numba.float32(19.)
            twenty = numba.float32(20.)
            pow19_kap = math.pow(r_sc - kappa, 19)
            pow20_kap = math.pow(r_sc - kappa, 20)
            pow19_lam = math.pow(r_sc - lamb, 19)
            pow20_lam = math.pow(r_sc - lamb, 20)

            denom_kap = one + pow20_kap
            denom_lam = one + pow20_lam
            phi_A = A * math.exp(-alpha*(r_sc-one)) / denom_kap
            phi_B = B * math.exp(-beta*(r_sc-one)) / denom_lam
            phi = phi_A - phi_B

            phi_A_p = - phi_A * (alpha + twenty*pow19_kap / (one + pow20_kap) ) / r_e # use denom_kap ?
            phi_B_p = - phi_B * (beta + twenty*pow19_lam / (one + pow20_lam) ) / r_e
            phi_p = phi_A_p - phi_B_p


            phi_A_pp = -phi_A_p*(alpha + (twenty*pow19_kap)/(one+pow20_kap)) - phi_A*( (twenty*nineteen*math.pow(r_sc - kappa, 18))/(one+pow20_kap) -
                                                                                      ((twenty*pow19_kap)/(one+pow20_kap))**2) / r_e
            phi_A_pp /= r_e
            phi_B_pp = -phi_B_p*(beta + (twenty*pow19_lam)/(one+pow20_lam)) - phi_B*( (twenty*nineteen*math.pow(r_sc - lamb, 18))/(one+pow20_lam) -
                                                                                     ((twenty*pow19_lam)/(one+pow20_lam))**2) / r_e
            phi_B_pp /= r_e
            phi_pp = phi_A_pp - phi_B_pp

            return phi, -phi_p/dist, phi_pp

        @cuda.jit( device=gridsync )  
        def calc_forces(vectors, cscalars, ptype, sim_box, nblist, params, elec_dens, embed_en_grad):
            """ Calculate forces as given by pairpotential_calculator() (needs to exist in outer-scope) using nblist 
                Kernel configuration: [num_blocks, (pb, tp)]        
            """
            
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y
            
            max_nbs = nblist.shape[1]-1


            my_f = cuda.local.array(shape=D,dtype=numba.float32)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)
            my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)
            if compute_stresses:
                my_stress = cuda.local.array(shape=(D,D), dtype=numba.float32)
            else:
                my_stress = cuda.local.array(shape=(1,1), dtype=numba.float32)
        
            if global_id < num_part:
                
                for k in range(D):
                    #my_r[k] = r[global_id, k]
                    my_f[k] = numba.float32(0.0)
                    if compute_stresses:
                        for k2 in range(D):
                            my_stress[k,k2] = numba.float32(0.0)
                for k in range(num_cscalars):
                    my_cscalars[k] = numba.float32(0.0)
                my_type = ptype_function(global_id, ptype)
            
            cuda.syncthreads() # Make sure initializing global variables to zero is done
            assert UtilizeNIII == False
            if global_id < num_part:
                my_type = ptype_function(global_id, ptype)
                params_my_type = params[my_type]
                my_embedding_energy = embed_en_grad[global_id, 0]
                my_embedding_grad = embed_en_grad[global_id, 1]
                my_embedding_second_der = embed_en_grad[global_id, 2]

                for i in range(my_t, nblist[global_id, max_nbs], tp):
                    other_id = nblist[global_id, i] 
                    other_type = ptype_function(other_id, ptype)
                    params_other_type = params[other_type]
                    dist_sq = dist_sq_dr_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box, my_dr)
                    my_cut = params_my_type[-1]
                    other_cut = params_other_type[-1]
                    cut = max(my_cut, other_cut)
                    if dist_sq < cut*cut:
                        dist = math.sqrt(dist_sq)
                        other_embedding_grad = embed_en_grad[other_id, 1]

                        # See comment about possible optimization a few lines down
                        my_rho, my_rho_s, my_rho_pp = electron_density_function(dist, params_my_type)
                        other_rho, other_rho_s, other_rho_pp = electron_density_function(dist, params_other_type)

                        effective_s = other_rho_s * my_embedding_grad + my_rho_s * other_embedding_grad
                        for k in range(D):
                            my_f[k] = my_f[k] - my_dr[k]*effective_s                        # Force
                            if compute_stresses:
                                for k2 in range(D):
                                    my_stress[k,k2] -= my_embedding_grad * other_rho_s * my_dr[k]*my_dr[k2]      # stress tensor
                        if compute_w:
                            my_cscalars[w_id] += my_embedding_grad*dist_sq*other_rho_s*virial_factor       # Virial

                        # Now for the pair part. Similar PairPotential.pairpotential_calculator but have to combine the "my" and "other" parts"
                        my_phi, my_phi_s, my_phi_pp = pair_contribution(dist, params_my_type)
                        other_phi, other_phi_s, other_phi_pp = pair_contribution(dist, params_other_type)
                        half = numba.float32(0.5)
                        zero = numba.float32(0.)
                        # Some optimization is possible by splitting the alloy pair potential between threads.
                        # So this thread  only calculates pair_contribution for this type, and adds the corresponding contribution to the neighbor's force also
                        u_pair = half * ((other_rho / my_rho) * my_phi + (my_rho / other_rho) * other_phi)
                        s_pair = half * ((other_rho_s/my_rho) * my_phi - (other_rho/my_rho**2) * my_rho_s * my_phi + (other_rho/my_rho) * my_phi_s +
                                         (my_rho_s / other_rho) * other_phi - (my_rho / other_rho**2) * other_rho_s * other_phi + (my_rho / other_rho) * other_phi_s )
                        umm_pair = zero # NOT IMPLEMENTED YET (will be  a monster expression, more complicated than s_pair, but built out of the derivatives already calculated)

                        for k in range(D):
                            my_f[k] = my_f[k] - my_dr[k]*s_pair                         # Force
                            if compute_w:
                                my_cscalars[w_id] += my_dr[k]*my_dr[k]*s_pair*virial_factor_half       # Virial
                            if compute_stresses:
                                for k2 in range(D):
                                    my_stress[k,k2] -= half*my_dr[k]*my_dr[k2]*s_pair      # stress tensor
                        if compute_u:
                            my_cscalars[u_id] += half*u_pair                                # Potential energy
                        if compute_lap:
                            my_cscalars[lap_id] += numba.float32(1-D)*s_pair + umm_pair          # Laplacian


                        ## TO DO
                        # 1. Test energy conservation with two actually different types DONE
                        # 2. Find a way to test physical properties for a pure system DONE
                        # 3. Find a physics test for an alloy. DONE
                        # 4. include many-body contribution to stresses DONE
                        # 5. Include Laplacian

                        
                # Now add this thread's contribution to the global force array (and stresses)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])
                    if compute_stresses:
                        cuda.atomic.add(vectors[sx_id], (global_id, k), my_stress[0,k])
                        if D > 1:
                            cuda.atomic.add(vectors[sy_id], (global_id, k), my_stress[1,k])
                            if D > 2:
                                cuda.atomic.add(vectors[sz_id], (global_id, k), my_stress[2,k])
                                if D > 3:
                                    cuda.atomic.add(vectors[sw_id], (global_id, k), my_stress[3,k])


                # (... and scalars)
                for k in range(num_cscalars):
                    cuda.atomic.add(cscalars, (global_id, k), my_cscalars[k])

            return 

        nblist_check_and_update = self.nblist.get_kernel(configuration, compute_plan, compute_flags, verbose)

        if gridsync:
            # A device function, calling a number of device functions, using gridsync to syncronize
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters, elec_dens, embed_en_grad = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                grid.sync()
                calc_electron_density_and_embedding_energy(vectors, scalars, ptype, sim_box, nblist, params, elec_dens, embed_en_grad)
                grid.sync()
                calc_forces(vectors, scalars, ptype, sim_box, nblist, params, elec_dens, embed_en_grad)
                return
            return compute_interactions
        
        else:
            # A python function, making several kernel calls to syncronize  
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters, elec_dens, embed_en_grad = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                calc_electron_density_and_embedding_energy(vectors, scalars, ptype, sim_box, nblist, params, elec_dens, embed_en_grad)
                calc_forces[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, nblist, params, elec_dens, embed_en_grad)
                return
            return compute_interactions


