import numpy as np
import numba
import math
from numba import cuda
from .make_fixed_interactions import make_fixed_interactions   # bonds is an example of 'fixed' interactions

# Abstract Base Class and type annotation
from .interaction import Interaction
from gamdpy import Configuration

class Angles(Interaction): 

    def __init__(self, potential, indices, parameters):
        self.potential = potential
        self.indices = np.array(indices, dtype=np.int32) 
        self.params = np.array(parameters, dtype=np.float32)


    def get_params(self, configuration: Configuration, compute_plan: dict, verbose=False) -> tuple:
        self.d_indices = cuda.to_device(self.indices)
        self.d_params = cuda.to_device(self.params);
        return (self.d_indices, self.d_params)

    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, N = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (N - 1) // pb + 1

        if D != 3:
            raise ValueError("Angle interactions only support D=3 (three dimensional) simulations")

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']
        compute_stresses = compute_flags['stresses']
        if compute_stresses:
            print('WARNING: computation of stresses is not implemented yet for angles')

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        u_id = configuration.sid['U']

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

        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
        potential_function = numba.njit(self.potential)

        def angle_calculator(vectors, scalars, ptype, sim_box, indices, values):
            '''
            Algorithm is based on D.C. Rapaport, The Art of Molecular Dynamics Simulations, 
            Cambridge University Press (1995).
            '''
            
            nparams = 2 #numba.int32(values.shape[1])
            params = cuda.local.array(shape=nparams, dtype=numba.float32)

            for n in range(nparams):
                params[n] = values[indices[3]][n]
            
            dr_1 = cuda.local.array(shape=D,dtype=numba.float32)
            dr_2 = cuda.local.array(shape=D,dtype=numba.float32)
            
            one = numba.float32(1.0)

            dist_sq_dr_function(vectors[r_id][indices[1]], vectors[r_id][indices[0]], sim_box, dr_1)
            dist_sq_dr_function(vectors[r_id][indices[2]], vectors[r_id][indices[1]], sim_box, dr_2)

            c11 = dr_1[0]*dr_1[0] + dr_1[1]*dr_1[1] + dr_1[2]*dr_1[2]
            c12 = dr_1[0]*dr_2[0] + dr_1[1]*dr_2[1] + dr_1[2]*dr_2[2]
            c22 = dr_2[0]*dr_2[0] + dr_2[1]*dr_2[1] + dr_2[2]*dr_2[2]

            cD = math.sqrt(c11*c22)
            ca = c12/cD
           
            if  ca > one:
                ca = one
            elif ca < -one:
                ca = -one
            angle = math.acos(ca) 

            u, f = potential_function(angle, params)

            for k in range(D):
                f_1 = f*( (c12/c11)*dr_1[k] - dr_2[k] )/cD
                f_2 = f*( dr_1[k] - (c12/c22)*dr_2[k] )/cD

                cuda.atomic.add(vectors, (f_id, indices[0], k), f_1)      
                cuda.atomic.add(vectors, (f_id, indices[1], k), -f_1-f_2)
                cuda.atomic.add(vectors, (f_id, indices[2], k), f_2)

            onethird = numba.float32(1.0/3.0)*u;
            cuda.atomic.add(scalars, (indices[0], u_id), onethird) 
            cuda.atomic.add(scalars, (indices[1], u_id), onethird)
            cuda.atomic.add(scalars, (indices[2], u_id), onethird)

            return
        
        return make_fixed_interactions(configuration, angle_calculator, compute_plan, verbose=False)
    
    def get_exclusions(self, configuration, max_number_exclusions=20):
            
        exclusions = np.zeros( (configuration.N, max_number_exclusions+1), dtype=np.int32 ) 
        
        nangles = len(self.indices)
        for n in range(nangles):
            pidx = self.indices[n][:3]
            for k in range(3):

                offset = exclusions[pidx[k]][-1]
                if offset > max_number_exclusions-2:
                    raise ValueError("Number of max. exclusion breached")
                
                if k==0:
                    idx = [1, 2]
                elif k==1:
                    idx = [0, 2]
                else:
                    idx = [0, 1]

                for kk in idx:
                    if angles_entry_not_exists(pidx[1], exclusions[pidx[k]],offset):
                        exclusions[pidx[k]][offset] = pidx[1]
                        offset += 1
                    
                exclusions[pidx[k]][-1] = offset

        return exclusions  
                 
    def get_angle(self, angle_idx, configuration):
        
        pidx = self.indices[angle_idx][:3]

        r1 = configuration['r'][pidx[0]]
        r2 = configuration['r'][pidx[1]]
        r3 = configuration['r'][pidx[2]]
        
        dr_1 = angles_get_dist_vector(r1, r2, configuration.simbox)
        dr_2 = angles_get_dist_vector(r3, r2, configuration.simbox)

        c11 = dr_1[0]*dr_1[0] + dr_1[1]*dr_1[1] + dr_1[2]*dr_1[2]
        c12 = dr_1[0]*dr_2[0] + dr_1[1]*dr_2[1] + dr_1[2]*dr_2[2]
        c22 = dr_2[0]*dr_2[0] + dr_2[1]*dr_2[1] + dr_2[2]*dr_2[2]

        cD = math.sqrt(c11*c22)
        cc = c12/cD 

        angle = math.acos(cc)
        
        return angle
 
# Helpers 
def angles_get_dist_vector(ri, rj, simbox):
    """
    Assumes Orthorhombic box
    """
    dr = np.zeros(3)
    lengths = simbox.get_lengths()
    for k in range(simbox.D): 
        dr[k] = ri[k] - rj[k]
        box_k = lengths[k]
        #PP
        dr[k] += (-box_k if 2.0*dr[k] > +box_k else (+box_k if 2.0*dr[k] < -box_k else 0.0)) 

    return dr

def angles_entry_not_exists(idx, exclusion_list, nentries):

    for n in range(nentries):
        if exclusion_list[n]==idx:
            return False

    return True
