
import numpy as np
import numba
import math
from numba import cuda

class NbListLinkedLists():
    
    def __init__(self, configuration, exclusions, max_num_nbs):
        self.configuration = configuration
        self.N, self.D = configuration.N, configuration.D
        self.nblist = np.zeros((configuration.N, max_num_nbs+1), dtype=np.int32) 
        self.nbflag = np.zeros(3, dtype=np.int32)
        self.r_ref = np.zeros_like(configuration['r']) # Inherents also data type
        self.exclusions = exclusions  # Should be able to be a list (eg from bonds, angles, etc), and merge
        self.d_simbox_last_rebuild = cuda.to_device(np.zeros(configuration.simbox.len_sim_box_data, dtype=np.float32))

    def copy_to_device(self):
        self.d_nblist = cuda.to_device(self.nblist)
        self.d_nbflag = cuda.to_device(self.nbflag)
        self.d_r_ref = cuda.to_device(self.r_ref)
        self.d_exclusions = cuda.to_device(self._exclusions)
        self.d_cells_per_dimension = cuda.to_device(self.cells_per_dimension) # Mapping cells to 1D, so need 'shape'
        self.d_cells = cuda.to_device(self.cells)
        self.d_my_cell = cuda.to_device(self.my_cell)
        self.d_next_particle_in_cell = cuda.to_device(self.next_particle_in_cell)
    
    def get_params(self, max_cut, compute_plan, verbose=True):
        self.max_cut = max_cut
        self.skin, = [compute_plan[key] for key in ['skin']]

        if type(self.exclusions) == np.ndarray: # Don't change user-set properties
            self._exclusions = self.exclusions.copy()
        else:
            self._exclusions = np.zeros((self.r_ref.shape[0], 2), dtype=np.int32)

        self.cells_per_dimension = np.zeros((self.D), dtype=np.int32)
        min_cells_size = (self.max_cut + self.skin)/2
        for i in range(self.D):
            self.cells_per_dimension[i] = int(self.configuration.simbox.lengths[i]/min_cells_size)
            assert self.cells_per_dimension[i] > 4
            if i == 0:
                assert self.cells_per_dimension[i] > 6
            # TODO: Take care of
            # - changing simbox size during simulation (NPT, compression)


        #print(self.cells_per_dimension)
        self.my_cell = np.zeros((self.N, self.D), np.int32) # my_cell[:, -1] 1d index
        self.cells = -np.ones(self.cells_per_dimension, dtype=np.int32)
        #self.cells = -np.ones(np.prod(self.cells_per_dimension), dtype=np.int32) # Map to 1D array
        #print(self.cells.shape)

        self.next_particle_in_cell = -np.ones(self.N, dtype=np.int32) # -1 = no further particles in list
        self.copy_to_device()                     
        return (np.float32(self.max_cut), np.float32(self.skin), self.d_nbflag, self.d_r_ref, self.d_exclusions, 
                self.d_cells_per_dimension, self.d_cells, self.d_my_cell, self.d_next_particle_in_cell, self.d_simbox_last_rebuild)

    def get_kernel(self, configuration, compute_plan, compute_flags, verbose=False, force_update=False):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_part - 1) // pb + 1
        compute_stresses = compute_flags['stresses']

        loop_x_addition = configuration.simbox.get_loop_x_addition()

        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]
        if compute_stresses:
            sx_id = configuration.vectors.indices['sx']
            if D > 1:
                sy_id = configuration.vectors.indices['sy']
                if D > 2:
                    sz_id = configuration.vectors.indices['sz']
                    if D > 3:
                        sw_id = configuration.vectors.indices['sw']


        # JIT compile functions to be compiled into kernel
        dist_sq_function = numba.njit(configuration.simbox.get_dist_sq_function())
        dist_moved_exceeds_limit_function = numba.njit(configuration.simbox.get_dist_moved_exceeds_limit_function())
        loop_x_shift_function = numba.njit(configuration.simbox.get_loop_x_shift_function())

        @cuda.jit( device=gridsync )
        def nblist_check(vectors, sim_box, skin, r_ref, nbflag, simbox_last_rebuild, cut):
            """ Check validity of nblist, i.e. did any particle mode more than skin/2 since last nblist update?
                Each tread-block checks the assigned particles (global_id)
                nbflag[0] = 0          : No update needed
                nbflag[0] = num_blocks : Update needed
                Kernel configuration: [num_blocks, (pb, tp)]
            """

            global_id, my_t = cuda.grid(2)
            if force_update: # nblist update forced (for benchmark or similar)
                if global_id==0 and my_t==0:
                    nbflag[0]=num_blocks

            if global_id < num_part and my_t==0:
                if dist_moved_exceeds_limit_function(vectors[r_id][global_id], r_ref[global_id], sim_box, simbox_last_rebuild, skin, cut):
                    nbflag[0] = num_blocks

            #if global_id < num_part and my_t==0:
            #    dist_sq = dist_sq_function(vectors[r_id][global_id], r_ref[global_id], sim_box)
            #    if dist_sq > skin*skin*numba.float32(0.25):
            #        nbflag[0]=num_blocks


            return
   
        @cuda.jit(device=gridsync)
        def put_particles_in_cells(vectors, sim_box, nbflag, cells_per_dimension, cells, my_cell, next_particle_in_cell):
            """ Each particle computers its cells coordinates, and inserts itself in cell-list
                Kernel configuration: [num_blocks, (pb, tp)]
            """

            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                for k in range(D):
                    #my_cell[global_id,k] = int(math.floor(vectors[r_id][global_id,k]*cells_per_dimension[k]/sim_box[k]))%cells_per_dimension[k]
                    my_cell[global_id,k] = int(math.floor((vectors[r_id][global_id,k]+0.5*sim_box[k])*cells_per_dimension[k]/sim_box[k]))%cells_per_dimension[k]
                    #if my_cell[global_id,k]<0:
                    #    print(global_id,k, my_cell[global_id,k],vectors[r_id][global_id,k])
                index = (my_cell[global_id,0], my_cell[global_id,1], my_cell[global_id,2])      # 3D 
                next_particle_in_cell[global_id] = cuda.atomic.exch(cells, index, global_id)    # index needs to be tuple when multidim
                
        @cuda.jit(device=gridsync)
        def nblist_update_from_linked_lists(vectors, sim_box, cut_plus_skin, nbflag,  cells_per_dimension, cells, my_cell, next_particle_in_cell, nblist, r_ref, exclusions):
            """ Order N neighbor-list update from linked lists 
                Kernel configuration: [num_blocks, (pb, tp)] USING ONLY MY_T == 0 for now...
            """

            global_id, my_t = cuda.grid(2)
            local_id = cuda.threadIdx.x 

            cell_length_x = sim_box[0] / cells_per_dimension[0]
            loop_x_shift = loop_x_shift_function(sim_box, cell_length_x)

            max_nbs = nblist.shape[1]-1 # Last index is used for storing number of neighbors

            if global_id < num_part and my_t==0:
                my_num_nbs = 0
                my_num_exclusions = exclusions[global_id, -1]

                for ix in range(-2-loop_x_addition,3+loop_x_addition,1):
                    for iy in range(-2,3,1):
                        # Correct handling of LEBC requires modifyng the loop over neighbor cells to take the box shift into account.
                        other_cell_y_unwrapped = my_cell[global_id, 1]+iy
                        y_wrap_cell = 1 if other_cell_y_unwrapped >= cells_per_dimension[1] else -1 if other_cell_y_unwrapped < 0 else 0
                        shifted_ix = ix + y_wrap_cell * loop_x_shift

                        for iz in range(-2,3,1):
                            other_index = (
                                (my_cell[global_id, 0]+shifted_ix)%cells_per_dimension[0],
                                (my_cell[global_id, 1]+iy)%cells_per_dimension[1],
                                (my_cell[global_id, 2]+iz)%cells_per_dimension[2])
                            other_global_id = cells[other_index]
                            while other_global_id >= 0: # To use tp>1: read tp particles ahead, and pick yours
                                if UtilizeNIII: # Could be done per cell basis...
                                    #flag = other_global_id < global_id
                                    TwodN = 2*(other_global_id - global_id)
                                    flag = other_global_id < num_part and (0 < TwodN <= num_part or TwodN < -num_part)
                                else:
                                    flag = other_global_id != global_id
                                if flag:
                                    dist_sq = dist_sq_function(vectors[r_id][other_global_id], vectors[r_id][global_id], sim_box)
                                    if dist_sq < cut_plus_skin*cut_plus_skin:
                                        not_excluded = True  # Check exclusion list. Do later ???
                                        for k in range(my_num_exclusions):
                                            if exclusions[global_id, k] ==  other_global_id:
                                                not_excluded = False
                                        if not_excluded:
                                            my_num_nbs += 1
                                            if my_num_nbs < max_nbs:                         
                                                nblist[global_id, my_num_nbs-1] = other_global_id     # Last entry is number of neighbors
                                other_global_id = next_particle_in_cell[other_global_id]
                nblist[global_id, -1] = my_num_nbs

                # Various house-keeping
                for k in range(D):    
                    r_ref[global_id, k] = vectors[r_id][global_id, k]   # Store positions for wich nblist was updated ( used in nblist_check() ) 
            #if local_id == 0 and my_t==0:
            #    cuda.atomic.add(nbflag, 0, -1)              # nbflag[0] = 0 by when all blocks are done
            if global_id == 0 and my_t==0:
                cuda.atomic.add(nbflag, 2, 1)               # Count how many updates are done in nbflag[2]
            if my_num_nbs >= max_nbs:                       # Overflow detected, nbflag[1] should be checked later, and then
                cuda.atomic.max(nbflag, 1, my_num_nbs)      # re-allocate larger nb-list, and redo computations from last safe state


        @cuda.jit(device=gridsync)
        def clear_cells(vectors, sim_box, nbflag, cells_per_dimension, cells, my_cell, next_particle_in_cell):
            """ Particles clears cell-list (only one (e.g. tail) actually needs to do this)
                Kernel configuration: [num_blocks, (pb, tp)]
            """

            global_id, my_t = cuda.grid(2)
            local_id = cuda.threadIdx.x

            if global_id < num_part and my_t == 0:
                for k in range(D):
                    if my_cell[global_id,k]<0:
                        print(global_id,k, my_cell[global_id,k],vectors[r_id][global_id,k])


            if global_id < num_part and my_t == 0: # Change to simple Nullify kernel?
                #cells[my_cell[global_id,:]] = -1 # Every body writes, but thats OK
                cells[my_cell[global_id,0], my_cell[global_id,1], my_cell[global_id,2]] = -1 # Every body writes, but thats OK
                next_particle_in_cell[global_id] = -1            

                
            if cuda.threadIdx.x == 0 and my_t==0: # One thread per threadblock decreases from numblocks
                cuda.atomic.add(nbflag, 0, -1)    # i.e., nbflag[0] = 0 by when all blocks are done
                                                  # (Not necessarry, after check moved to final kernel?)

        @cuda.jit(device=gridsync)
        def nblist_update(vectors, sim_box, cut_plus_skin, nbflag, nblist, r_ref, exclusions):
            """ N^2 Update neighbor-list using numba.cuda 
                Kernel configuration: [num_blocks, (pb, tp)]
            """

            global_id, my_t = cuda.grid(2)
            local_id = cuda.threadIdx.x 

            max_nbs = nblist.shape[1]-1 # Last index is used for storing number of neighbors

            if global_id < num_part and my_t==0:
                nblist[global_id, max_nbs] = 0  # Set number of neighbors (stored at index max_nbs) to zero
                
            cuda.syncthreads() # wait for nblist[global_id, max_nbs] to be ready
            
            if global_id < num_part:
                my_num_exclusions = exclusions[global_id,-1]
                for i in range(0, num_part, pb*tp): # Loop over blocks
                    for j in range(pb):             # Loop over particles the pb in block
                        other_global_id = j + i + my_t*pb   
                        if UtilizeNIII:
                            TwodN = 2*(other_global_id - global_id)
                            flag = other_global_id < num_part and (0 < TwodN <= num_part or TwodN < -num_part)
                        else:
                            flag = other_global_id != global_id and other_global_id < num_part
                        if flag:  
                            dist_sq = dist_sq_function(vectors[r_id][other_global_id], vectors[r_id][global_id], sim_box)
                            if dist_sq < cut_plus_skin*cut_plus_skin:
                                not_excluded = True  # Check exclusion list
                                for k in range(my_num_exclusions):
                                    if exclusions[global_id, k] ==  other_global_id:
                                        not_excluded = False
                                if not_excluded:
                                    my_num_nbs = cuda.atomic.add(nblist, (global_id, max_nbs), 1)   # Find next free index in nblist
                                    if my_num_nbs < max_nbs:                         
                                        nblist[global_id, my_num_nbs] = other_global_id     # Last entry is number of neighbors

            # Various house-keeping
            if global_id < num_part and my_t==0:
                for k in range(D):    
                    r_ref[global_id, k] = vectors[r_id][global_id, k]   # Store positions for wich nblist was updated ( used in nblist_check() ) 
            #if local_id == 0 and my_t==0:
            #    cuda.atomic.add(nbflag, 0, -1)              # nbflag[0] = 0 by when all blocks are done
            if global_id == 0 and my_t==0:
                cuda.atomic.add(nbflag, 2, 1)               # Count how many updates are done in nbflag[2]
            if my_num_nbs >= max_nbs:                       # Overflow detected, nbflag[1] should be checked later, and then
                cuda.atomic.max(nbflag, 1, my_num_nbs)      # re-allocate larger nb-list, and redo computations from last safe state

            return 
        
        if gridsync==True:
            # A device function, calling a number of device functions, using gridsync to syncronize
            @cuda.jit( device=gridsync )
            def check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters):
                max_cut, skin, nbflag, r_ref, exclusions, cells_per_dimension, cells, my_cell, next_particle_in_cell, simbox_last_rebuild = nblist_parameters
                nblist_check(vectors, sim_box, skin, r_ref, nbflag, simbox_last_rebuild, max_cut)
                grid.sync()
                if nbflag[0] > 0:
                    put_particles_in_cells(vectors, sim_box, nbflag, cells_per_dimension, cells, my_cell, next_particle_in_cell)
                    grid.sync()
                    nblist_update_from_linked_lists(vectors, sim_box, max_cut+skin, nbflag,  cells_per_dimension, cells, my_cell, next_particle_in_cell, nblist, r_ref, exclusions)           
                    #nblist_update(vectors, sim_box, max_cut+skin, nbflag, nblist, r_ref, exclusions)
                    grid.sync()
                    clear_cells(vectors, sim_box, nbflag,  cells_per_dimension, cells, my_cell, next_particle_in_cell)
                return
            return check_and_update
        
        else:
            # A python function, making several kernel calls to syncronize  
            def check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters):
                max_cut, skin, nbflag, r_ref, exclusions, cells_per_dimension, cells, my_cell, next_particle_in_cell, simbox_last_rebuild = nblist_parameters
                nblist_check[num_blocks, (pb, 1)](vectors, sim_box, skin, r_ref, nbflag, simbox_last_rebuild, max_cut)
                if nbflag[0] > 0:
                    put_particles_in_cells[num_blocks, (pb, 1)](vectors, sim_box, nbflag, cells_per_dimension, cells, my_cell, next_particle_in_cell)
                    nblist_update_from_linked_lists[num_blocks, (pb, 1)](vectors, sim_box, max_cut+skin, nbflag,  cells_per_dimension, cells, my_cell, next_particle_in_cell, nblist, r_ref, exclusions)           
                    #nblist_update[num_blocks, (pb, tp)](vectors, sim_box, max_cut+skin, nbflag, nblist, r_ref, exclusions) # Maybe tp = 1 ???
                    clear_cells[num_blocks, (pb, 1)](vectors, sim_box, nbflag,  cells_per_dimension, cells,  my_cell, next_particle_in_cell)
                return
            return check_and_update
