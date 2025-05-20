import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # bonds is an example of 'fixed' interactions
from .interaction import Interaction

# Abstract Base Class and type annotation
from .interaction import Interaction
from gamdpy import Configuration

class Dihedrals(Interaction): 

    def __init__(self, potential, indices, parameters):
        self.potential = potential
        self.indices = np.array(indices, dtype=np.int32) 
        self.params = np.array(parameters, dtype=np.float32)

    def get_params(self, configuration: Configuration, compute_plan: dict, verbose=False) -> tuple:
        self.d_indices = cuda.to_device(self.indices)
        self.d_params = cuda.to_device(self.params)
        
        return (self.d_indices, self.d_params)

    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1

        if D != 3:
            raise ValueError("Dihedral interactions only support D=3 (three dimensional) systems")

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']
        compute_stresses = compute_flags['stresses']
        if compute_stresses:
            print('WARNING: computation of stresses is not implemented yet for dihedrals')

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

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        #u_id = configuration.sid['U']

        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
        potential_function = numba.njit(self.potential)
   
        def dihedral_calculator(vectors, scalars, ptype, sim_box, indices, values):
            '''
            Algorithm is based on D.C. Rapaport, The Art of Molecular Dynamics Simulations, 
            Cambridge University Press (1995).
            '''
            
            nparams = 6
            p = cuda.local.array(shape=nparams,dtype=numba.float32)
            for n in range(nparams):
                p[n] = values[indices[4]][n]

            one = numba.float32(1.0)

            dr_1 = cuda.local.array(shape=D,dtype=numba.float32)
            dr_2 = cuda.local.array(shape=D,dtype=numba.float32)
            dr_3 = cuda.local.array(shape=D,dtype=numba.float32)
            
            dist_sq_dr_function(vectors[r_id][indices[1]], vectors[r_id][indices[0]], sim_box, dr_1)
            dist_sq_dr_function(vectors[r_id][indices[2]], vectors[r_id][indices[1]], sim_box, dr_2)
            dist_sq_dr_function(vectors[r_id][indices[3]], vectors[r_id][indices[2]], sim_box, dr_3)

            c11 = dr_1[0]*dr_1[0] + dr_1[1]*dr_1[1] + dr_1[2]*dr_1[2]
            c12 = dr_1[0]*dr_2[0] + dr_1[1]*dr_2[1] + dr_1[2]*dr_2[2]
            c13 = dr_1[0]*dr_3[0] + dr_1[1]*dr_3[1] + dr_1[2]*dr_3[2]
            c22 = dr_2[0]*dr_2[0] + dr_2[1]*dr_2[1] + dr_2[2]*dr_2[2]
            c23 = dr_2[0]*dr_3[0] + dr_2[1]*dr_3[1] + dr_2[2]*dr_3[2]
            c33 = dr_3[0]*dr_3[0] + dr_3[1]*dr_3[1] + dr_3[2]*dr_3[2]

            cA = c13*c22 - c12*c23
            cB1 = c11*c22 - c12*c12
            cB2 = c22*c33 - c23*c23
            cD = math.sqrt(cB1*cB2)
            cd = cA/cD

            t1 = cA
            t2 = c11*c23 - c12*c13
            t3 = -cB1
            t4 = cB2
            t5 = c13*c23 - c12*c33
            t6 = -cA
            cR1 = c12/c22
            cR2 = c23/c22

            if cd > one:
                cd = one
            elif cd < -one:
                cd = -one

            dihedral =  math.pi - math.acos(cd)
            u, f = potential_function(dihedral, p)
            
            for k in range(3):
                f1 = f*c22*(t1*dr_1[k] + t2*dr_2[k] + t3*dr_3[k])/(cD*cB1)
                f2 = f*c22*(t4*dr_1[k] + t5*dr_2[k] + t6*dr_3[k])/(cD*cB2)

                cuda.atomic.add(vectors, (f_id, indices[0], k), f1)      # Force
                cuda.atomic.add(vectors, (f_id, indices[1], k), -(one + cR1)*f1 + cR2*f2)
                cuda.atomic.add(vectors, (f_id, indices[2], k), cR1*f1 - (one + cR2)*f2)
                cuda.atomic.add(vectors, (f_id, indices[3], k), f2)

            u_per_part = numba.float32(0.25)*u    

            if compute_u:
                for n in range(4):
                    cuda.atomic.add(scalars, (indices[n], u_id), u_per_part) 

            return
        
        return make_fixed_interactions(configuration, dihedral_calculator, compute_plan, verbose=False)
    

    def get_exclusions(self, configuration, max_number_exclusions=20):
            
        exclusions = np.zeros( (configuration.N, max_number_exclusions+1), dtype=np.int32 ) 
        
        nangles = len(self.indices)
        for n in range(nangles):
            pidx = self.indices[n][:4]
            for k in range(4):

                offset = exclusions[pidx[k]][-1]
                if offset > max_number_exclusions-2:
                    raise ValueError("Number of max. exclusion breached")

                if k==0:
                    idx = [1, 2, 3]
                elif k==1:
                    idx = [0, 2, 3]
                elif k==2:
                    idx = [0, 1, 3]
                else:
                    idx = [0, 1, 2]

                for kk in idx:
                    if dihedrals_entry_not_exists(pidx[kk], exclusions[pidx[k]], offset):
                        exclusions[pidx[k]][offset] = pidx[kk]
                        offset += 1
                
                exclusions[pidx[k]][-1] = offset

        return exclusions  
                 

    def get_dihedral(self, dihedral_idx, configuration):
        
        pidx = self.indices[dihedral_idx][:4]

        r0 = configuration['r'][pidx[0]]
        r1 = configuration['r'][pidx[1]]
        r2 = configuration['r'][pidx[2]]
        r3 = configuration['r'][pidx[3]]

        dr_1 = dihedrals_get_dist_vector(r1, r0, configuration.simbox)
        dr_2 = dihedrals_get_dist_vector(r2, r1, configuration.simbox)
        dr_3 = dihedrals_get_dist_vector(r3, r2, configuration.simbox)

        c11 = dr_1[0]*dr_1[0] + dr_1[1]*dr_1[1] + dr_1[2]*dr_1[2]
        c12 = dr_1[0]*dr_2[0] + dr_1[1]*dr_2[1] + dr_1[2]*dr_2[2]
        c13 = dr_1[0]*dr_3[0] + dr_1[1]*dr_3[1] + dr_1[2]*dr_3[2]
        c22 = dr_2[0]*dr_2[0] + dr_2[1]*dr_2[1] + dr_2[2]*dr_2[2]
        c23 = dr_2[0]*dr_3[0] + dr_2[1]*dr_3[1] + dr_2[2]*dr_3[2]
        c33 = dr_3[0]*dr_3[0] + dr_3[1]*dr_3[1] + dr_3[2]*dr_3[2]

        cA = c13*c22 - c12*c23
        cB1 = c11*c22 - c12*c12
        cB2 = c22*c33 - c23*c23
        cD = math.sqrt(cB1*cB2)
        cc = cA/cD
        
        dihedral = math.pi - math.acos(cc)
      
        return dihedral 
 
# Helpers (copies of angles helpers: This should be centralized somehow)
def dihedrals_get_dist_vector(ri, rj, simbox):
    """
    Function for checking that angles are correctly calculated. Assumes Orthorhombic box
    """
    dr = np.zeros(3)
    lengths = simbox.get_lengths()
    for k in range(simbox.D): 
        dr[k] = ri[k] - rj[k]
        box_k = lengths[k]
        #PP
        dr[k] += (-box_k if 2.0*dr[k] > +box_k else (+box_k if 2.0*dr[k] < -box_k else 0.0)) 

    return dr

def dihedrals_entry_not_exists(idx, exclusion_list, nentries):

    for n in range(nentries):
        if exclusion_list[n]==idx:
            return False

    return True

