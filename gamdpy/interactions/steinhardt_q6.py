import numpy as np
import numba
import math
from numba import cuda
import matplotlib.pyplot as plt
import gamdpy as gp
from .interaction import Interaction

class Steinhardt_Q6(Interaction):
    r"""  Potential for constraining the Steinhardt order parameter Q6 for use in interface pinning simulations.
    
    
    Parameters
    ----------
    
    kappa6: float
    anchor6: float
    switch_cutoff: float
    switch_width:float
    max_num_nbs : int
        Maximum number of neighbors per particle to allocate in the neighbor list.

    References
    ----------

        Interface pinning paper (URP), Steinhardt paper


    Examples
    --------

    Example of creating a potential for 

    >>> 
    >>> 
    >>> 
    >>>


    """

    def __init__(self, kappa6: float, anchor6: float, switch_cutoff: float, switch_width: float, max_num_nbs: int):


        self.kappa6 = kappa6
        self.anchor6 = anchor6
        self.switch_cutoff = switch_cutoff
        self.switch_width = switch_width
        self.max_num_nbs = max_num_nbs
        self.nb_cut = switch_cutoff + switch_width*math.log(1e5-1)
        print(f'nb_cut {self.nb_cut}')
        # in the future might have different parmaeters for different types, in particular cutoff/switch functions....
        self.params = np.array([kappa6, anchor6, switch_cutoff, switch_width, self.nb_cut], dtype=np.float32)
        

    def check_datastructure_validity(self) -> bool:
        nbflag = self.nblist.d_nbflag.copy_to_host()
        if nbflag[0] != 0 or nbflag[1] != 0:
            raise RuntimeError(f'Neighbor-list is invalid. Try allocating space for more neighbors (max_num_nbs in PairPot object). Allocated size: {self.max_num_nbs}, but {nbflag[1]+1} neighbours found. {nbflag=}.')
        return True

    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:
        

        self.d_params = cuda.to_device(self.params)

        N = configuration.N
        
        if compute_plan['nblist'] == 'N squared':
            self.nblist = gp.NbList2(configuration, None, self.max_num_nbs)
        elif compute_plan['nblist'] == 'linked lists':
            self.nblist = gp.NbListLinkedLists(configuration, None, self.max_num_nbs)
        else:
            raise ValueError(f"No lblist called: {compute_plan['nblist']}. Use either 'N squared' or 'linked lists'")
        nblist_params = self.nblist.get_params(self.nb_cut, compute_plan, verbose)

        # create storage for q6: seven complex numbers, as well as the normalization (sum of switching_factors)
        self.q6_sum = np.zeros(7, dtype=np.complex64)
        self.switch_sum_Q6_energy = np.zeros(3, dtype=np.float32)

        self.d_q6_sum = cuda.to_device(self.q6_sum)
        self.d_switch_sum_Q6_energy = cuda.to_device(self.switch_sum_Q6_energy)
        return (self.d_params, self.nblist.d_nblist, nblist_params, self.d_switch_sum_Q6_energy, self.d_q6_sum)

    def GetQ6(self):
        self.switch_sum_Q6_energy = self.d_switch_sum_Q6_energy.copy_to_host()
        switch_sum, Q6_device, Q6_energy = self.switch_sum_Q6_energy
        self.q6_sum = self.d_q6_sum.copy_to_host()
        sum_qlm_non_norm2 = self.q6_sum[0].real**2

        for m in range(1, 7):
            sigma_f_Y6m = self.q6_sum[m]
            sum_qlm_non_norm2 += 2.*(sigma_f_Y6m.real**2 + sigma_f_Y6m.imag**2)

        four_pi_two_lp1 = 4.*math.pi/(2*6+1)
        Q6 = math.sqrt(four_pi_two_lp1 * sum_qlm_non_norm2)/switch_sum
        prefactor_A = -self.kappa6 * (Q6-self.anchor6) * four_pi_two_lp1 / Q6

        return Q6

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        num_cscalars = configuration.num_cscalars

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
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

 
        # Maybe print a warning or comment that UtilizeNIII is not relevant here
        #assert UtilizeNIII == False

        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
        dist_sq_function = numba.njit(configuration.simbox.get_dist_sq_function())

        # Initialize Amm
        Amm = np.ones(7, dtype=np.float32)
        for m in range(1, 7):
            Amm[m] = Amm[m-1] * (2*m+1)/(2*m)
        Amm = np.sqrt(Amm/(4.*math.pi))

        zero = numba.float32(0.0)
        one = numba.float32(1.0)
        two = numba.float32(2.0)
        four = numba.float32(4.0)
        six = numba.float32(6.0)
        eleven = numba.float32(11.)
        thirteen = numba.float32(13.)
        four_pi_two_lp1 = 4.*math.pi / thirteen
        # default value of phi when theta is zero or pi # CHANGE TO ZERO to avoid defining these!!!!
        #phi_def = numba.float32(math.pi/2)
        #cos_phi_def = numba.float32(math.cos(phi_def))
        #sin_phi_def = numba.float32(math.sin(phi_def))
        EPS = numba.float32(1.e-7)
        # TO DO
        # 1. In loop over neighbors, initialize angles and distances DONE 27/6
        # 2. Copy over m, l recursion loop and remove the Y4 stuff DONE  29/6
        # 3. Access parameters switch_cutoff and switch_width DONE 30/6
        # 4. Set up arrays for normalization and q6m sums DONE 1/7
        # 5. Atomically add contributions to normalization and q6m sums (FORCE==False) DONE 1/7
        # 6. Read in summed norm+q6m at beginning (FORCE==True) DONE 13/7
        # 7. Try testing in gamdpy DONE 14/7. Runs (though incomplete). Makes the minimal example slower by a factor 100 though.
        # 8. Add force loop DONE 17/7 Forces give nans
        # 9. Next test: Run without forces on an LJ sample and calculate the value of Q6. DONE, and debugged enough to compare
        # and get at least roughly correct Q6. But using print statments in the kernel to see it. Have gained a factor 13 in speed in the process!
        # 9.5 Make a better way to display the value of Q6 once per block (ie from the main loop) DONE 23/7
        # 10. Once have correct Q6, start debugging forces DONE 24/7
        # 11. Once force debugged, test speed again. DONE 24/7, much faster but still a factor 13 slower than ordinary minimal
        # 12. Start examining 32 vs 64 bit floats and reducing unnecessary double-precision and unnecessary casts. Look for places to improve the code generally
        # 13. Implement a proper way to store and extract the value of Q6 (needed for IP calculations). DONE 8/9
        # 14. Think about how to use this class not as an interaction but just to measure Q6 (right now can do this by setting kappa=0 but it would still
        # be calculated every time step so very slow)
        @cuda.jit(device=gridsync)
        def zero_sum_arrays(switch_sum_Q6_energy, q6_sum):
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y

            if global_id == 0 and my_t == 0:
                switch_sum_Q6_energy[0] = zero
                switch_sum_Q6_energy[1] = zero
                switch_sum_Q6_energy[2] = zero
                for m in range(7):
                    q6_sum[m] = complex(zero, zero)

        @cuda.jit( device=gridsync )  
        def calc_q6_forces(FORCES: bool, vectors, cscalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum):
            """ Calculate forces as given by pairpotential_calculator() (needs to exist in outer-scope) using nblist 
                Kernel configuration: [num_blocks, (pb, tp)]        
            """
            
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y
            
            kappa6, anchor6, switch_cutoff, switch_width, cut = params
            max_nbs = nblist.shape[1]-1


            my_f = cuda.local.array(shape=D,dtype=numba.float32)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)
            my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)
    
            my_Y6 = cuda.local.array(shape=7, dtype=numba.complex64)
            my_switch_factor_sum = zero



            assert UtilizeNIII == False

            if global_id < num_part:
                # define local Y6m

                sigma_f_Y6 = cuda.local.array(shape=7, dtype=numba.complex64)
                sigma_f = one
                Q6 = zero
                prefactor_A = zero
                # for use when actually determining force contributions:
                if(FORCES):
                    # Read in the summed values
                    sigma_f = switch_sum_Q6_energy[0]
                    sigma_f_Y6[0] = q6_sum[0]
                    sum_qlm_non_norm2 = sigma_f_Y6[0].real**2
                    for m in range(1, 7):
                        sigma_f_Y6[m] = q6_sum[m]
                        sum_qlm_non_norm2 += 2*(sigma_f_Y6[m].real**2 + sigma_f_Y6[m].imag**2)


                    Q6 = math.sqrt(four_pi_two_lp1 * sum_qlm_non_norm2)/sigma_f
                    prefactor_A = -kappa6 * (Q6-anchor6) * four_pi_two_lp1 / Q6
                    Q6_energy = kappa6 * (Q6 - anchor6)**2/2
                    if global_id == 0 and my_t == 0:
                        switch_sum_Q6_energy[1] = Q6
                        switch_sum_Q6_energy[2] = Q6_energy

                for i in range(my_t, nblist[global_id, max_nbs], tp):
                    other_id = nblist[global_id, i] 
                    
                    # calling the distance function with the first two arguments the other way around to match
                    # what RUMD does (otherwise have to include an extra minus sign on the forces somewhere)
                    dist_sq = dist_sq_dr_function(vectors[r_id][global_id], vectors[r_id][other_id], sim_box, my_dr)
                    if dist_sq < cut*cut:
                        dist = numba.float32(math.sqrt(dist_sq))
                        dist_xy2 = my_dr[0]*my_dr[0] + my_dr[1]*my_dr[1]
                        dist_xy = numba.float32(math.sqrt(dist_xy2))
                        cos_theta = my_dr[2] / dist
                        abs_sin_theta = dist_xy/dist
                        inv_sin_theta = one/abs_sin_theta
  
                        if(abs_sin_theta < EPS):
                            inv_sin_theta = one/EPS
                            phi, cos_phi, sin_phi = zero, one, zero # phi_def, cos_phi_def, sin_phi_def
                        else:
                            phi = numba.float32(math.atan2(my_dr[1], my_dr[0]))
                            cos_phi = my_dr[0] / dist_xy
                            sin_phi = my_dr[1] / dist_xy

                        # Next, calculate Ylms for these angles.
                        # Need l=6 with 7 different m values (only positive m)

                        sin_theta_m = one # will be multiplied by sin_theta at the end of the m-loop
                        sin_theta_m_minus_1 = one # will be multiplied by sin_theta at the end of the m-loop for m> 0 (so it will be incorrect for m=0, but it won't be used in that case)

                        exp_switch = numba.float32(math.exp( (dist-switch_cutoff)/switch_width))
                        switch_factor = one/(one + exp_switch)
                        # the following expression should be safe against the exp factor getting huge
                        switch_deriv = - one/( (exp_switch + two + one/exp_switch) * switch_width)
                    
                        my_switch_factor_sum += switch_factor;
                        P6_m, P6_m_over_sin_theta = zero, zero

                        #cos_m_phi, sin_m_phi = one, one

                        for m in range(7):

                            # phi part of Ylm
                            #cos_m_phi, sin_m_phi = math.cos(m*phi), math.sin(m*phi)
                            cos_sin_m_phi = complex(numba.float32(math.cos(m*phi)), numba.float32(math.sin(m*phi)))

                            # initialize the first two l values for the recurrence, namely l=m and l=m+1
                            P_current_minus_2 = Amm[m]# * sin_theta_m;
                            # Would like to include this here: * switch_factor;

                            P_current_minus_1 = numba.float32(math.sqrt((four*(m+1)*(m+1) - one)/(two*m+one)) * cos_theta * P_current_minus_2)
                            P6_deriv_sin1 = zero # the suffix sin1 indicates that this includes the factor 1/sin(theta)  (only one power of the sine)

                            # some of the needed values are already available before the loop:
                            if m == 5:

                                if(FORCES):
                                    P6_m = P_current_minus_1 * sin_theta_m
                                    P6_m_over_sin_theta = P_current_minus_1 * sin_theta_m_minus_1
                                     # sqrt(13) comes from same formula as above with l=6,, m=5
                                    P6_deriv_sin1 = -( six* cos_theta*P_current_minus_1 - math.sqrt(thirteen)*P_current_minus_2) * sin_theta_m_minus_1#sin_theta_m * inv_sin_theta
                                else:
                                    #my_Y6[5] += complex64(switch_factor * P_current_minus_1 * sin_theta_m * cos_m_phi, switch_factor * P_current_minus_1 * sin_theta_m * cos_m_phi)
                                    my_Y6[5] += switch_factor * P_current_minus_1 * sin_theta_m * cos_sin_m_phi

                            elif m == 6:
                                if(FORCES):
                                    P6_m = P_current_minus_2 * sin_theta_m;
                                    P6_m_over_sin_theta = P_current_minus_2 * sin_theta_m_minus_1
                                    P6_deriv_sin1 = - six * cos_theta*P_current_minus_2 * sin_theta_m_minus_1 # sin_theta_m * inv_sin_theta
                                else:
                                    my_Y6[6] += switch_factor * P_current_minus_2 * sin_theta_m * cos_sin_m_phi
                                    #Y_6[6].x += switch_factor * P_current_minus_2 * sin_theta_m *cos_m_phi
                                    #Y_6[6].y += switch_factor * P_current_minus_2 * sin_theta_m *sin_m_phi

                            # recurrence starts from l=m+2
                            for l in range(m+2, 7):
                                a_lm = numba.float32(math.sqrt( float(4*l*l-1)/float(l*l-m*m) ))
                                b_lm = -numba.float32(math.sqrt( float(2*l+1)/float(2*l-3) * float((l-1)*(l-1) - m*m) / float(l*l-m*m) ))
                                P_current = a_lm * cos_theta * P_current_minus_1 + b_lm * P_current_minus_2


                                if l == 6: # m must be at most 4, but m=5,6 already taken care of CAN WE MOVE THIS OUT OF THE LOOP BEAUSE l=6 AT THE END??
                                    if(FORCES):
                                        P6_m = P_current * sin_theta_m
                                        P6_m_over_sin_theta = P_current * sin_theta_m_minus_1 # incorrect, but needed, for m=0
                                        if m>0:
                                            P6_deriv_sin1 = -(six*cos_theta * P_current - math.sqrt((thirteen/eleven) * (6+m)*(6-m)) * P_current_minus_1) * sin_theta_m_minus_1
                                        else:
                                            # for m=0 we have dP_l(x)/dx are just polynomials in x=cos(theta) whose product with sin(theta) must vanish as theta->0. The expression in brackets gives always (I checked explicitly up to l=4) a factor sin^2(theta),ie it vanishes fast enough that dividing by sin theta doesn't change it. ALTERNATIVE: Use the end-point friendly recursion for derivatives of Legendre polynomials (m=0), then we don't need inv_sin_theta
                                            P6_deriv_sin1 = -(6*cos_theta * P_current - math.sqrt((thirteen/eleven) * 6*6) * P_current_minus_1) * inv_sin_theta

                                    else:
                                        my_Y6[m] +=  switch_factor * P_current * sin_theta_m *  cos_sin_m_phi
                                        #numba.float32(switch_factor * P_current * sin_theta_m) * numba.complex64( cos_sin_m_phi)
                                        #Y_6[m].x += switch_factor * P_current * sin_theta_m *cos_m_phi
                                        #Y_6[m].y += switch_factor * P_current * sin_theta_m *sin_m_phi

                                # The following line is why we can't move the if  l== 6 out of the loop: P_current_minus_1 is needed before it gets updated.
                                P_current_minus_2 = P_current_minus_1
                                P_current_minus_1 = P_current
                                # end loop over l


                            if(FORCES):
                                q6_m = sigma_f_Y6[m] / sigma_f
                                double_counting_factor = 2
                                if m > 0:
                                    double_counting_factor = 4
                                k1 = double_counting_factor*switch_deriv/(sigma_f*sigma_f*dist)
                                D1 = k1*(-sigma_f_Y6[m] + (sigma_f * P6_m) * cos_sin_m_phi)
                                s = prefactor_A * (q6_m.real * D1.real + q6_m.imag * D1.imag)

                                for k in range(3):
                                    my_f[k] = s*my_dr[k]

                                D2 = double_counting_factor * switch_factor * P6_deriv_sin1 / (sigma_f * dist_sq)*cos_sin_m_phi #  Term  2, involving derivative of cos theta
                                s = prefactor_A * (q6_m.real * D2.real + q6_m.imag * D2.imag)

                                my_f[0] += - s * cos_phi * my_dr[2]
                                my_f[1] += - s * sin_phi * my_dr[2]
                                my_f[2] += s * dist_xy

                                if(m>0):
                                    k4 = double_counting_factor * switch_factor * P6_m_over_sin_theta * m/(sigma_f*dist );
                                    D_m3 = complex(k4 * cos_sin_m_phi.imag, -k4 * cos_sin_m_phi.real) #  Term 3, involving derivatives of exp (i m phi)
                                    s = prefactor_A * (q6_m.real * D_m3.real + q6_m.imag * D_m3.imag)

                                    my_f[0] += s*sin_phi
                                    my_f[1] += -s*cos_phi


                                for k in range(D):
                                    cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])



                            sin_theta_m *= abs_sin_theta
                            if m > 0:
                                sin_theta_m_minus_1 *= abs_sin_theta
                        # End loop over m


                        #for k in range(D):
                        #    my_f[k] = my_f[k] - my_dr[k] # XXXXX                      # Force
        
                        #if compute_w:
                        #    my_cscalars[w_id] += 0. # XXXXXX       # Virial


                        #half = numba.float32(0.5)

                        #for k in range(D):
                        #    my_f[k] = my_f[k] - my_dr[k] #                # Force
                        #    if compute_w:
                        #        my_cscalars[w_id] += 0.#       # Virial
                    
                        #if compute_u:
                        #    my_cscalars[u_id] += half                            # Potential energy

                    # end if (within cutoff)
                # Now add this thread's contribution to the global force array (and stresses)


                # (... and scalars)
                #for k in range(num_cscalars):
                #    cuda.atomic.add(cscalars, (global_id, k), my_cscalars[k])

            # Outside the loop over neighbors for my particle handled by this thread, now each thread must add to the global sums
            if not FORCES:
                cuda.atomic.add(switch_sum_Q6_energy, 0, my_switch_factor_sum)
                for m in range(7):
                    # NEED TO SEPARATELY ADD REAL AND IMAGINARY PARTS
                    cuda.atomic.add(q6_sum.real, m, my_Y6[m].real)
                    cuda.atomic.add(q6_sum.imag, m, my_Y6[m].imag)

            return


        def make_calc_q6_with_without_forces():
            @cuda.jit(device = gridsync)
            def calc_q6_without_forces(vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum):
                calc_q6_forces(False, vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum)
            @cuda.jit(device = gridsync)
            def calc_q6_with_forces(vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum):
                calc_q6_forces(True, vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum)

            return calc_q6_without_forces, calc_q6_with_forces

        calc_q6_without_forces, calc_q6_with_forces  = make_calc_q6_with_without_forces()

        nblist_check_and_update = self.nblist.get_kernel(configuration, compute_plan, compute_flags, verbose)

        if gridsync:
            # A device function, calling a number of device functions, using gridsync to syncronize
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters, switch_sum_Q6_energy, q6_sum = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                grid.sync()
                zero_sum_arrays(switch_sum_Q6_energy, q6_sum)
                grid.sync()
                calc_q6_without_forces(vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum)
                grid.sync()
                calc_q6_with_forces(vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum)
                return
            return compute_interactions
        
        else:
            # A python function, making several kernel calls to syncronize  
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters, switch_sum_Q6_energy, q6_sum = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                zero_sum_arrays(switch_sum_Q6_energy, q6_sum)
                calc_q6_without_forces[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum)
                calc_q6_with_forces[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, nblist, params, switch_sum_Q6_energy, q6_sum)
                return
            return compute_interactions


