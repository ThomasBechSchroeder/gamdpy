import numpy as np
import numba
import math
import sys
from numba import cuda

# rumdpy
import gamdpy as rp

# For type annotation
from gamdpy.integrators import Integrator
from gamdpy.interactions import Interaction
from gamdpy.runtime_actions import RuntimeAction

# TODO: to remove import above you need to add a lot of lines are the following
#from ..simulation.get_default_compute_plan import get_default_compute_plan
#from ..configuration.Configuration import Configuration

# IO
import h5py


class Simulation():
    """ Class for running a simulation.

    Parameters
    ----------

    configuration : gamdpy.Configuration
        The configuration to simulate.

    interactions : list of interactions
        Interactions such as pair potentials, bonds, external fields, etc.

    integrator : an integrator
        The integrator to use for the simulation.

    runtime_actions : list of runtime actions
        Runtime actions such as ScalarSaver, ConfigurationSaver or MomentumReset

    num_steps : int
        Number of steps to run the simulation. If 0, num_timeblocks and steps_per_timeblock should be set.

    num_timeblocks : int
        Number of timeblocks to run the simulation. If not 0, then steps_per_timeblock should be set.

    steps_per_timeblock : int
        Number of steps per timeblock.

    compute_plan : dict
        A dictionary with the compute plan for the simulation. If None, a default compute plan is used.

    storage : str
        Storage for the simulation output. Can be 'memory' or a filename with extension '.h5'.

    compute_flags : dict
        For each scalar quantity, and stresses, specifies whether to be calculated

    verbose : bool
        If True, print verbose output.

    timing : bool
        If True, timing information is saved.


    See also
    --------

    :func:`gamdpy.get_default_sim`

    """

    def __init__(self, configuration: rp.Configuration, 
                 interactions: Interaction|list[Interaction], 
                 integrator: Integrator,
                 runtime_actions: list[RuntimeAction],
                 num_steps=0, num_timeblocks=0, steps_per_timeblock=0,
                 compute_plan=None, storage='output.h5',
                 #compute_flags=None, 
                 verbose=False, timing=True,
                 steps_in_kernel_test=1):

        self.configuration = configuration
        if compute_plan == None:
            self.compute_plan = rp.get_default_compute_plan(self.configuration)
        else:
            self.compute_plan = compute_plan

        # Integrator
        if type(interactions) == list:
            self.interactions = interactions
        else:
            self.interactions = [interactions, ]
        self.integrator = integrator
        self.dt = self.integrator.dt

        if num_timeblocks == 0:
            if num_steps == 0:
                raise ValueError("Either num_steps or num_timeblocks must be non-zero")
            num_timeblocks = 32
            steps_per_timeblock = 2 ** int(math.log2(math.ceil(num_steps / num_timeblocks)))
            num_timeblocks = math.ceil(num_steps / steps_per_timeblock)
            print('num_steps: ', num_steps)
            print('num_blocks: ', num_timeblocks)
            print('steps_per_block: ', steps_per_timeblock)
        elif steps_per_timeblock == 0:
            raise ValueError("If num_timeblocks is non-zero then steps_per_timeblock must be too (num_steps is ignored in this case)")
        # we do not use nsteps if num_timeblocks is non-zero, because it's
        # not guaranteed to be a multiple of the latter
        self.num_blocks = num_timeblocks
        self.current_block = -1

        self.steps_per_block = steps_per_timeblock
        self.storage = storage
        self.timing = timing
        self.steps_in_kernel_test = steps_in_kernel_test

        # Close output object if there
        # Check https://stackoverflow.com/questions/610883/how-to-check-if-an-object-has-an-attribute
        # Create output objects
        if self.storage == 'memory':
            # Creates a memory h5 file with named id(self).h5; id(self) is ensured to be unique
            self.memory = h5py.File(f"{id(self)}.h5", "w", driver='core', backing_store=False)
        elif self.storage[-3:] == '.h5':
            # The append is important for repeated istances of sim with same self.storage
            self.memory = h5py.File(self.storage, "w")
        else:
            print("Simulation data will not be saved")
        # Save setup info
        self.memory.attrs['dt'] = self.dt
        self.memory.attrs['simbox_initial'] = self.configuration.simbox.lengths
        if 'ptype' in self.memory.keys():
            del self.memory['ptype']
        self.memory.create_dataset("ptype", data=configuration.ptype, shape=(self.configuration.N), dtype=np.int32)
        if 'script_name' not in self.memory.keys():
            script_name = sys.argv[0]
            self.memory.attrs['script_name'] = script_name
            if isinstance(script_name,str) and script_name != '':
                with open(script_name, 'r') as file:
                    script_content = file.read()
                self.memory.attrs['script_content'] = script_content


        self.runtime_actions = runtime_actions

        compute_flags = None
        for runtime_action in self.runtime_actions:
            if runtime_action.get_compute_flags() is not None:
                if compute_flags is not None:
                    raise ValueError('Can not handle more than one compute_flags in runtime_actions')
                else:
                    compute_flags = runtime_action.get_compute_flags()

        self.compute_flags = rp.get_default_compute_flags() # configuration.compute_flags
        if compute_flags is not None:
            # only keys present in the default are processed
            for k in compute_flags:
                if k in self.compute_flags:
                    self.compute_flags[k] = compute_flags[k]
                else:
                    raise ValueError('Unknown key in compute_flags:%s' %k)
        for k in self.compute_flags:
            if self.compute_flags[k] and not configuration.compute_flags[k]:
                raise ValueError('compute_flags["%s]" set for Simulation but not in Configuration' % k)

        for runtime_action in self.runtime_actions:
            runtime_action.setup(configuration=self.configuration, num_timeblocks=num_timeblocks,
                                steps_per_timeblock=steps_per_timeblock, output=self.memory )

        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []

        self.JIT_and_test_kernel()
        for interaction in self.interactions: # Attempt to catch (eg.) nblist errors before actually doing any simulation
            interaction.check_datastructure_validity()


        if self.storage[-3:] == '.h5':
            self.memory.close()

    # __del__ is supposed to work also if __init__ fails. This means you can't use attributed defined in __init__
    # https://www.algorithm.co.il/programming/python-gotchas-1-__del__-is-not-the-opposite-of-__init__/

    def get_output(self, mode="r"):
        if self.storage[-3:] == '.h5':
            output = h5py.File(self.storage, mode)
        elif self.storage == 'memory':
            output = self.memory
        else:
            print("Warning: self.output can't recognize self.storage option, returning None")
            output = None
        return output

    # might be worth looking into cache_property https://www.reddit.com/r/learnpython/comments/184kqzp/in_class_properties_defined_in_functions/
    @property
    def output(self):
        return self.get_output()

    def JIT_and_test_kernel(self, adjust_compute_plan=True):
        while True:
            try:
                self.get_kernels_and_params()
                self.integrate = self.make_integrator(self.configuration, self.integrator_kernel, self.interactions_kernel,
                                                #self.output_calculator_kernel,
                                                self.runtime_actions_prestep_kernel,
                                                self.runtime_actions_poststep_kernel,
                                                self.compute_plan, True)
                self.configuration.copy_to_device() # By _not_ copying back to host later we dont change configuration
                self.integrate_self(0.0, self.steps_in_kernel_test)
                break
            except numba.cuda.cudadrv.driver.CudaAPIError as e:
                if not adjust_compute_plan:
                    self.compute_plan['tp'] = 0 # Signal failure to autotuner
                    break
                #print('Failed compute_plan : ', self.compute_plan)
                if self.compute_plan['tp'] > 1:             # Most common problem tp is too big
                    self.compute_plan['tp'] -= 1            # ... so we reduce it and try again
                elif self.compute_plan['gridsync'] == True: # Last resort: turn off gridsync
                    self.compute_plan['gridsync'] = False
                else:
                    print(f'FAILURE. Can not handle cuda error {e}')
                    exit()
                print('Trying adjusted compute_plan :', self.compute_plan)

    def get_kernels_and_params(self, verbose=False):
        # Interactions
        self.interactions_kernel, self.interactions_params = rp.add_interactions_list(self.configuration,
                                                                                      self.interactions,
                                                                                      compute_plan=self.compute_plan,
                                                                                      compute_flags=self.compute_flags)

        # Runtime actions
        if self.runtime_actions:
            self.runtime_actions_prestep_kernel, self.runtime_actions_poststep_kernel, self.runtime_actions_params = rp.add_runtime_actions_list(self.configuration,
                                                                                                    self.runtime_actions,
                                                                                                    compute_plan=self.compute_plan)
        else:
            self.runtime_actions_prestep_kernel = None
            self.runtime_actions_poststep_kernel = None
            self.runtime_actions_params = (0,)

        # Integrator
        self.integrator_params = self.integrator.get_params(self.configuration, self.interactions_params)
        self.integrator_kernel = self.integrator.get_kernel(self.configuration, self.compute_plan, self.compute_flags, self.interactions_kernel)

        return

    def update_params(self, verbose=False):
        # Interactions
        _, self.interactions_params = rp.add_interactions_list(self.configuration,
                                                                self.interactions,
                                                                compute_plan=self.compute_plan,
                                                                compute_flags=self.compute_flags)

        # Runtime actions
        if self.runtime_actions:
            _, _, self.runtime_actions_params = rp.add_runtime_actions_list(self.configuration,
                                                                                                    self.runtime_actions,
                                                                                                    compute_plan=self.compute_plan)
        else:
            #self.runtime_actions_kernel = None
            self.runtime_actions_params = (0,)

        # Integrator
        self.integrator_params = self.integrator.get_params(self.configuration, self.interactions_params, verbose)
        #self.integrator_kernel = self.integrator.get_kernel(self.configuration, self.compute_plan, verbose)

        return

    def integrate_self(self, time_zero, steps):
        self.integrate(self.configuration.d_vectors,
                           self.configuration.d_scalars,
                           self.configuration.d_ptype,
                           self.configuration.d_r_im,
                           self.configuration.simbox.d_data,
                           self.interactions_params,
                           self.integrator_params,
                           self.runtime_actions_params,
                           np.float32(time_zero),
                           steps)
        return


    def make_integrator(self, configuration, integration_step, compute_interactions,# output_calculator_kernel,
                        runtime_actions_prestep_kernel, runtime_actions_poststep_kernel, compute_plan, verbose=True):
        
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        num_blocks = (num_part - 1) // pb + 1

        get_integrator_setup = getattr(self.integrator, "get_setup_kernel", None)
        if get_integrator_setup is not None:
            integrator_setup = get_integrator_setup(self.configuration, self.compute_plan, self.interactions_kernel)
            self.configuration.copy_to_device()
            integrator_setup(
                self.configuration.d_vectors,
                self.configuration.d_scalars,
                self.configuration.d_r_im,
                self.configuration.simbox.d_data,
                self.integrator_params,
                self.configuration.d_ptype,
            )

        if gridsync:
            # Return a kernel that does 'steps' timesteps, using grid.sync to syncronize   
            @cuda.jit
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params,
                            runtime_actions_params, time_zero, steps):
                grid = cuda.cg.this_grid()
                for step in range(steps + 1): # make extra step without integration, so that interactions and run_time actions called for final configuration
                    time = time_zero + step * integrator_params[0]
                    compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_params)
                    grid.sync()
                    if runtime_actions_prestep_kernel != None:
                        runtime_actions_prestep_kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params)
                        grid.sync()
                    if step<steps:
                        integration_step(grid, vectors, scalars, r_im, sim_box, integrator_params, time, ptype)
                        grid.sync()
                        if runtime_actions_poststep_kernel != None:
                            runtime_actions_poststep_kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params)
                            grid.sync()

                return

            return integrator[num_blocks, (pb, tp)]

        else:

            # Return a Python function that does 'steps' timesteps, using kernel calls to syncronize  
            def integrator(vectors, scalars, ptype, r_im, sim_box, interaction_params, integrator_params,
                           runtime_actions_params, time_zero, steps):
                for step in range(steps + 1): # make extra step without integration, so that interactions and run_time actions called for final configuration
                    time = time_zero + step * integrator_params[0]
                    compute_interactions(0, vectors, scalars, ptype, sim_box, interaction_params)
                    if runtime_actions_prestep_kernel != None:
                        runtime_actions_prestep_kernel(0, vectors, scalars, r_im, sim_box, step, runtime_actions_params)
                    if step<steps:
                        integration_step(0, vectors, scalars, r_im, sim_box, integrator_params, time, ptype)
                        if runtime_actions_poststep_kernel != None:
                            runtime_actions_poststep_kernel(0, vectors, scalars, r_im, sim_box, step, runtime_actions_params)
                return

            return integrator
        return

        # simple run function

    def run(self, verbose=True):
        """ Run the simulation.

        See also
        --------

        :func:`gamdpy.Simulation.timeblocks`

        """
        for _ in self.run_timeblocks():
            if verbose:
                print(self.status(per_particle=True))
        if verbose:
            print(self.summary())

    # generator for running simulation one block at a time
    def run_timeblocks(self, num_timeblocks=-1):
        """ Generator for running the simulation one block at a time.

        Parameters
        ----------

        num_timeblocks : int
            Number of blocks to run. If -1, all blocks are run.

        Examples
        --------

        >>> import gamdpy as gp
        >>> sim = gp.get_default_sim()
        >>> for block in sim.run_timeblocks(num_timeblocks=3):
        ...     print(f'{block=}')  # Replace with code to analyze the current configuration
        block=0
        block=1
        block=2

        See also
        --------

        :func:`gamdpy.Simulation.run`

        """
        if num_timeblocks == -1:
            num_timeblocks = self.num_blocks
        self.last_num_blocks = num_timeblocks
        assert (num_timeblocks <= self.num_blocks)  # Could be made OK with more blocks

        self.configuration.copy_to_device()
        self.vectors_list = []
        self.scalars_list = []
        self.simbox_data_list = []
        self.scalars_t = []

        if self.timing:
            start = cuda.event()
            end = cuda.event()
            start_block = cuda.event()
            end_block = cuda.event()
            block_times = []
            start.record()

        zero = np.float32(0.0)

        for block in range(num_timeblocks):

            self.current_block = block
            for runtime_action in self.runtime_actions:
                runtime_action.initialize_before_timeblock()
            
            if self.timing: 
                start_block.record()

            self.integrate_self(np.float32(block * self.steps_per_block * self.dt),self.steps_per_block)

            if self.timing:
                end_block.record()
                end_block.synchronize()
                block_times.append(cuda.event_elapsed_time(start_block, end_block))

            self.configuration.copy_to_host()

            for interaction in self.interactions:
                interaction.check_datastructure_validity()

            for runtime_action in self.runtime_actions:
                runtime_action.update_at_end_of_timeblock(block, self.get_output(mode="a"))

            if self.storage[-3:] == '.h5':
                self.output.close()

            yield block

        # Finalizing run
        if self.timing:
            end.record()
            end.synchronize()

            self.timing_numba = cuda.event_elapsed_time(start, end)
            self.timing_numba_blocks = np.array(block_times)
        #self.nbflag = self.interactions[0].nblist.d_nbflag.copy_to_host()
        self.scalars_list = np.array(self.scalars_list)

    def status(self, per_particle=False) -> str:
        """ String with the current status
        Should be executed during the simulation run, see :func:`gamdpy.Simulation.timeblocks`

        Parameters
        ----------

        per_particle : bool
            If True, the values are divided by the number of particles in the configuration.

        Returns
        -------

        str
            A string with the current status of the simulation.

        """
        time = (self.current_block+1) * self.steps_per_block * self.dt
        st = f'timeblock= {self.current_block :<6}'
        st += f'{time= :<12.3f}'
        for name in self.configuration.sid:
            if name in self.configuration.compute_flags and self.configuration.compute_flags[name]:
                data = np.sum(self.configuration[name])
                if per_particle:
                    data /= self.configuration.N
                st += f'{name}= {data:<10.3f}'
        return st

    def summary(self) -> str:
        """ Returns a summary string of the simulation run.
         Should be called after the simulation has been run,
         see :func:`gamdpy.Simulation.timeblocks` or :func:`gamdpy.Simulation.run`
         """
        if self.timing:
            time_total = self.timing_numba / 1000
            tps_total = self.last_num_blocks * self.steps_per_block / time_total
            time_sim = np.sum(self.timing_numba_blocks) / 1000
            tps_sim = self.last_num_blocks * self.steps_per_block / time_sim

        st = f'Particles : {self.configuration.N} \n'
        st += f'Steps : {self.last_num_blocks} * {self.steps_per_block} = '
        st += f'{self.last_num_blocks * self.steps_per_block:_} \n'
        if self.timing:
            st += f'Total run time  (incl. time spent between timeblocks): {time_total:.2f} s '
            st += f'( TPS: {tps_total:.2e} )\n'
            st += f'Simulation time (excl. time spent between timeblocks): {time_sim:.2f} s '
            st += f'( TPS: {tps_sim:.2e} )\n'
        return st

    def autotune_bruteforce(self, pbs='auto', skins='auto', tps='auto', timesteps=0, repeats=1, verbose=False):
        if verbose:
            print('compute_plan :', self.compute_plan)
        if timesteps==0: 
            timesteps = self.steps_per_block
        assert timesteps<=self.steps_per_block
        
        pb = self.compute_plan['pb']
        if pbs=='auto':
            pbs = [pb//2, pb, pb*2]
        if pbs=='default':
            pbs = [pb,]
        if verbose:
            print('pbs :', pbs)

        tp = self.compute_plan['tp']
        if tps=='auto': 
            tps = [tp - 3, tp - 2, tp - 1, tp, tp + 1, tp + 2, tp + 3,]
        if tps=='default':
            tps = [tp,]
        if verbose:
            print('tps :', tps)

        skin = self.compute_plan['skin']
        if skins=='auto':
            skins = [skin - 0.6, skin - 0.4, skin - 0.2, skin, skin + 0.2, skin + 0.4, skin + 0.6, skin + 0.8,  skin + 1.0] 
        elif skins=='default':
            skins = [skin, ]
        if verbose:
            print('skins :', skins)

        nblists = []
        if self.configuration.N < 32000:
            nblists.append('N squared')
        if self.configuration.N > 2000:
            nblists.append('linked lists')
        if verbose:
            print('nblists:', nblists)

        gridsyncs = []
        if self.configuration.N < 200000:
            gridsyncs.append(True)
        if self.configuration.N > 10000:
            gridsyncs.append(False)
        if verbose:
            print('gridsyncs :', gridsyncs)
            
        UtilizeNIIIs = [False, True]
        if verbose:
            print('UtilizeNIIIs :', UtilizeNIIIs)
            
        flag = cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS
        cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = False
        
        skin_times = []
        total_min_time = 1e9
        optimal_compute_plan = rp.get_default_compute_plan(self.configuration) # Overwritten below

        for nblist in nblists:
            self.compute_plan['nblist'] = nblist
            for gridsync in gridsyncs:
                self.compute_plan['gridsync'] = gridsync
                for UtilizeNIII in UtilizeNIIIs:
                    self.compute_plan['UtilizeNIII'] = UtilizeNIII
                    print(f'\n {nblist}, {gridsync=}, {UtilizeNIII=}: ', end='')
                    local_min_time = 1e9
                    for pb in pbs:
                        if pb <= 512:
                            self.compute_plan['pb'] = pb
                            for tp in tps:
                                if tp>0:
                                    self.compute_plan['tp'] = tp
                                    gridsync = self.compute_plan['gridsync']
                                    self.JIT_and_test_kernel(adjust_compute_plan=False)
                                    # does kernel run without adjustment?
                                    if self.compute_plan['tp'] != tp or self.compute_plan['gridsync'] != gridsync: 
                                        break
                                    #print('Seems to work, so looping over skins...')
                                    total_min_time, local_min_time = self.autotune_scan_skin(self.compute_plan, skins, timesteps, repeats, total_min_time, local_min_time, optimal_compute_plan, verbose)

        self.compute_plan = optimal_compute_plan
        if verbose:
            print('\nFinal compute_plan :', self.compute_plan)
        else:
            print('')
            #print('\nFinal compute_plan :', self.compute_plan)
        self.JIT_and_test_kernel()
        
        cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = flag


    def autotune_scan_skin(self, compute_plan, skins, timesteps, repeats, total_min_time, local_min_time, optimal_compute_plan, verbose=False):
        min_time = 1e9
        skin_times = []
        pb = compute_plan['pb']
        tp = compute_plan['tp']
        for skin in skins:
            if 0 < skin < 1.2:
                self.compute_plan['skin'] = skin
                self.update_params()
                self.configuration.copy_to_device() # By _not_ copying back to host later we dont change configuration
                start = cuda.event()
                end = cuda.event()
                start.record()
                for i in range(repeats):
                    self.integrate_self(0.0, timesteps)
                end.record()
                end.synchronize()
                time_elapsed = cuda.event_elapsed_time(start, end)
                if time_elapsed < min_time:
                    min_time = time_elapsed
                    min_skin = skin
                skin_times.append(time_elapsed)
                if verbose:
                    print(f"({skin}, {skin_times[-1]:.3})", end=' ', flush=True)
        max_TPS = repeats * timesteps / min_time * 1000
        if verbose:
            print('\n', pb, tp, min_skin, min_time, max_TPS)
        else:
            print('.', end='', flush=True)
        if min_time < local_min_time:
            local_min_time = min_time
            print(f' ({pb}x{tp},{min_skin:.2f}):{max_TPS:.2f}', end=' ', flush=True)
        if min_time < total_min_time:
            total_min_time = min_time
            optimal_compute_plan['UtilizeNIII'] = self.compute_plan['UtilizeNIII']
            optimal_compute_plan['nblist'] = self.compute_plan['nblist']
            optimal_compute_plan['gridsync'] = self.compute_plan['gridsync']
            optimal_compute_plan['skin'] = min_skin
            optimal_compute_plan['pb'] = pb
            optimal_compute_plan['tp'] = tp
        return total_min_time, local_min_time


    def autotune(self):
        flag = cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS
        cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = False

        initial_compute_plan = self.compute_plan.copy()
        timesteps = self.steps_per_block
        repeats = 1

        # Binary choises: Take self.compute_plan as starting point, and add alternatives if appropiate 
        gridsyncs = [initial_compute_plan['gridsync'], ]
        if gridsyncs[0] != True:
            gridsyncs.append(True)
        if gridsyncs[0] != False and self.configuration.N > 10000:
            gridsyncs.append(False)

        nblists = [initial_compute_plan['nblist'], ]
        if nblists[0] != 'N squared' and self.configuration.N < 32000:
            nblists.append('N squared')
        if nblists[0] != 'linked lists' and self.configuration.N > 2000:
            nblists.append('linked lists')

        UtilizeNIIIs = [initial_compute_plan['UtilizeNIII'], ]
        if UtilizeNIIIs[0] != False:
            UtilizeNIIIs.append(False)
        if UtilizeNIIIs[0] != True:
            UtilizeNIIIs.append(True)


        pb = self.compute_plan['pb']
        pbs = [pb//2, pb, pb*2]

        optimal_compute_plan = initial_compute_plan.copy()
        results = []
        # Loop over binary parameters
        total_min_time = 1e9
        binary_min_time = 1e9
        for nblist in nblists:
            self.compute_plan['nblist'] = nblist
            for gridsync in gridsyncs:
                self.compute_plan['gridsync'] = gridsync
                for UtilizeNIII in UtilizeNIIIs:
                    self.compute_plan['UtilizeNIII'] = UtilizeNIII
                    
                    # Get time for default 'pb', 'tp' to check if its worth to proceede with these choises
                    self.compute_plan['pb'] = initial_compute_plan['pb']
                    self.compute_plan['tp'] = initial_compute_plan['tp']
                    print(f'\n {nblist+",":12}\t{gridsync=}\t{UtilizeNIII=}:\t', end='')
                    self.JIT_and_test_kernel(adjust_compute_plan=False)
                    if self.compute_plan['tp'] == 0 or self.compute_plan['gridsync'] != gridsync: 
                        continue
        
                    local_min_time = 1e9
                    total_min_time, local_min_time, min_time = self.autotune_skin(initial_compute_plan['skin'], 0.2, 
                                                                        timesteps, repeats, total_min_time, local_min_time, 
                                                                        optimal_compute_plan, verbose=False)
                    self.compute_plan['min_time'] = min_time
                    if min_time < binary_min_time:
                        binary_min_time = min_time
                    results.append(self.compute_plan.copy())
        
        for compute_plan in results:
            if compute_plan['min_time'] < 1.15 * binary_min_time:
                        
                local_min_time = 1e9
                self.compute_plan = compute_plan.copy()
                nblist = compute_plan['nblist']
                gridsync = compute_plan['gridsync']
                UtilizeNIII = compute_plan['UtilizeNIII']
                print(f'\n {nblist+",":12}\t{gridsync=}\t{UtilizeNIII=}:\t', end='')

                pb = compute_plan['pb']
                pb_min_time = 1e9
                initial_min_time = -1e9
                while 4 <= pb <= 1024:
                    self.compute_plan['pb'] = pb
                    print(f' {pb=} ', end='')
                    total_min_time, local_min_time, min_time = self.autotune_tp(compute_plan['tp'], 1, compute_plan, timesteps, repeats, optimal_compute_plan, total_min_time, local_min_time)
                    if initial_min_time<0:
                        initial_min_time = min_time
                    #print(f'[{min_time:.3}, {pb_min_time:.3}]')
                    if min_time > 1.01 * pb_min_time:
                        break
                    if min_time < pb_min_time:
                        pb_min_time = min_time
                    pb *= 2
                pb = compute_plan['pb'] // 2
                if initial_min_time < 1.01*pb_min_time: # Is it worth trying smaller pb?
                    while 4 <= pb <= 1024:
                        self.compute_plan['pb'] = pb
                        print(f' {pb=} ', end='')
                        total_min_time, local_min_time, min_time = self.autotune_tp(compute_plan['tp'], 1, compute_plan, timesteps, repeats, optimal_compute_plan, total_min_time, local_min_time)
                        #print(f'[{min_time:.3}, {pb_min_time:.3}]')
                        if min_time > 1.01 * pb_min_time:
                            break
                        if min_time < pb_min_time:
                            pb_min_time = min_time
                        pb = pb // 2

        self.compute_plan = optimal_compute_plan
        self.JIT_and_test_kernel()
        print()
        cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = flag


    def autotune_tp(self, initial_tp, delta_tp, initial_compute_plan, timesteps, repeats, optimal_compute_plan, total_min_time, local_min_time):
        tp = initial_tp
        tp_min_time = 1e9
        while 0 < tp <= 64:
            #print(self.compute_plan['pb'], tp, (self.compute_plan['pb']*tp) % 32)
            if (self.compute_plan['pb']*tp) % 32 == 0:
                self.compute_plan['tp'] = tp
                self.JIT_and_test_kernel(adjust_compute_plan=False)
                if self.compute_plan['tp'] != tp: #or self.compute_plan['gridsync'] != gridsync: 
                    break
                total_min_time, local_min_time, min_time = self.autotune_skin(initial_compute_plan['skin'], 0.2, 
                                                                    timesteps, repeats, total_min_time, local_min_time, 
                                                                    optimal_compute_plan, verbose=False)
                if min_time > 1.05 * tp_min_time:
                    break
                if min_time < tp_min_time:
                    tp_min_time = min_time
            tp += delta_tp
        tp = initial_tp - delta_tp
        while 0 < tp <= 64:
            if (self.compute_plan['pb']*tp) % 32 == 0:
                self.compute_plan['tp'] = tp
                self.JIT_and_test_kernel(adjust_compute_plan=False)
                if self.compute_plan['tp'] != tp: #or self.compute_plan['gridsync'] != gridsync: 
                    break
                total_min_time, local_min_time, min_time = self.autotune_skin(initial_compute_plan['skin'], 0.2, 
                                                                    timesteps, repeats, total_min_time, local_min_time, 
                                                                    optimal_compute_plan, verbose=False)
                if min_time > 1.05 * tp_min_time:
                    break
                if min_time < tp_min_time:
                    tp_min_time = min_time
            tp -= delta_tp
        return total_min_time, local_min_time, tp_min_time

 
    def autotune_skin(self, initial_skin, delta_skin, timesteps, repeats, total_min_time, local_min_time, optimal_compute_plan, verbose=False):
        min_time = 1e9
        min_skin = -0.1
        
        # Upscan
        min_time, min_skin = self.scan_skin(timesteps, repeats, initial_skin, delta_skin, min_time, min_skin)
        
        # Downscan
        min_time, min_skin = self.scan_skin(timesteps, repeats, initial_skin-delta_skin, -delta_skin, min_time, min_skin)
        
        max_TPS = repeats * timesteps / min_time * 1000
        if min_time < local_min_time:
            local_min_time = min_time
            print(f" ({self.compute_plan['pb']}x{self.compute_plan['tp']},{min_skin:.2f}):{max_TPS:.2f}", end=' ', flush=True)
        else:
            print('.', end='', flush=True)
        if min_time < total_min_time:
            total_min_time = min_time
            optimal_compute_plan['UtilizeNIII'] = self.compute_plan['UtilizeNIII']
            optimal_compute_plan['nblist'] = self.compute_plan['nblist']
            optimal_compute_plan['gridsync'] = self.compute_plan['gridsync']
            optimal_compute_plan['skin'] = min_skin
            optimal_compute_plan['pb'] = self.compute_plan['pb']
            optimal_compute_plan['tp'] = self.compute_plan['tp']
        return total_min_time, local_min_time, min_time

    def scan_skin(self, timesteps, repeats, skin, delta_skin, min_time, min_skin):
        while 0.0 < skin < 1.45: # Should keep on going until eg. linked lists throw an error
            self.compute_plan['skin'] = skin
            self.update_params()
            self.configuration.copy_to_device() # By _not_ copying back to host later we dont change configuration
            start = cuda.event()
            end = cuda.event()
            start.record()
            for i in range(repeats):
                self.integrate_self(0.0, timesteps)
            end.record()
            end.synchronize()
            time_elapsed = cuda.event_elapsed_time(start, end)
            if time_elapsed < min_time:
                min_time = time_elapsed
                min_skin = skin
            elif time_elapsed > 1.1*min_time:
                break
            skin += delta_skin
        return min_time, min_skin











