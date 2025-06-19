import numpy as np
import math


class TimeScheduler():
    """
    Class used to:
        - define steps to save configuration at;
        - get functions for the numba kernel to check whether to save.

    An instance of TimeScheduler must be passed to TrajectorySaver, either explicitly or implicitly
    (i.e. passing appropriate kewords to TrajectorySaver, that will create an instance of TimeScheduler).

    Example:

    ..code-block:: python
        import gamdpy as gp
        scheduler = gp.TimeScheduler(schedule='log', base=1.5)
        runtime_actions = [gp.TrajectorySaver(scheduler=scheduler),]

    alternatively

    ..code-block:: python
        import gamdpy as gp
        runtime_actions = [gp.TrajectorySaver(schedule='log', base=1.5),]

    See below for indications about kwargs for different schedules. If no keyword or scheduler instance
    is passed to TrajectorySaver, it falls back to a logarithmic schedule with base 2. The list
    `runtime_actions` must then be passed to a Simulation instance.
    """

    def __init__(self, schedule='log2', **kwargs):

        self.schedule = schedule
        self._kwargs = kwargs

    def setup(self, stepmax, ntimeblocks):
        # This is necessary aside from __init__ because in TrajectorySaver
        # `steps_per_timeblock` is initialised only in a `setup` method

        # `stepmax` is by construction the same as `steps_per_timeblock` in TrajectorySaver
        # it makes sense to keep it as an attribute since it may be needed in the future for other schedules
        self.stepmax = stepmax
        self.ntimeblocks = ntimeblocks

        self.steps = np.arange(0, self.stepmax, dtype=int)

        # no specific kwarg is required
        if self.schedule=='log2':
            self.base = 2
            self.steps = self._steps_log()

        # `base` kwarg is accepted but not required
        elif self.schedule=='log':
            self.base = self._kwargs.get('base', np.exp(1.0))
            self.steps = self._steps_log()
            # self.stepcheck_func = self._get_stepcheck_log()

        # `steps_between_output` or `npoints` kwarg is required
        elif self.schedule=='lin':
            deltastep = self._kwargs.get('steps_between_output', None)
            npoints = self._kwargs.get('npoints', None)
            if deltastep is not None:
                self.deltastep = deltastep
                self.steps = self._steps_lin(delta=True)
            elif npoints is not None:
                self.npoints = npoints
                self.steps = self._steps_lin(delta=False)
            else:
                raise TypeError("'steps_between_output' or 'npoints' is required for schedule 'lin'")
            # self.stepcheck_func = self._get_stepcheck_lin()

        # `npoints` kwarg is required
        elif self.schedule=='geom':
            npoints = self._kwargs.get('npoints', None)
            if npoints is not None:
                self.npoints = npoints
                self.steps = self._steps_geom()
                assert self.nsaves==self.npoints, 'Too many points, schedule distorsion; try fewer points'
            else:
                raise TypeError("'npoints' is required for schedule 'geom'")
                
            # self.stepcheck_func = self._get_stepcheck_geom()

        # TODO
        elif self.schedule=='custom':
            pass

        if self.schedule != 'log2':
            self.stepcheck_func = self._get_stepcheck()
        else:
            self.stepcheck_func = self._get_stepcheck_log2()


        # self.steps = self._compute_steps()
        print(len(self.steps), self.steps)
        # this is specific to 'geom' schedule
        # if self.npoints is not None:
        #     assert self.npoints==self.nsaves, 'Invalid npoints, try saving fewer points'
        # # TODO: deprecate this
        # self.stepsall = self._compute_stepsall()

    def _steps_log(self):
        steps = [0]
        step = 0
        exponent = 0
        while True:
            step = int(self.base**exponent)
            if step>self.stepmax:
                break
            steps.append(step)
            exponent += 1
        return np.unique(np.array(steps))
        
    def _steps_lin(self, delta):
        steps = [0]
        if delta:
            step = 0
            while True:
                step += self.deltastep
                if step>=self.stepmax:
                    break
                steps.append(step)
        else:
            delta = self.stepmax / self.npoints
            time = 0
            while True:
                time += delta
                if time>=self.stepmax:
                    break
                steps.append(int(time))
        return np.unique(np.array(steps))

    def _steps_geom(self):
        steps = [0]
        a = self.stepmax**(1.0/self.npoints)
        for i in range(1, self.npoints):
            step = int(a**(i+1) - 1)
            steps.append(step)
        return np.unique(np.array(steps))

    def _get_stepcheck(self):
        steps = np.array(self.steps)
        def stepcheck(step):
            for j in range(len(steps)):
                if step==steps[j]:
                    return True, j
            return False, -1
        return stepcheck
    
    def _get_stepcheck_log2(self):
        def stepcheck(step):
            # flag = False
            # save_index = -1 # this is for python calls of the function
            if step == 0:
                # flag = True
                # save_index = 0
                return True, 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step==c:
                    # flag = True
                    # save_index = b + 1
                    return True, b+1
            # return flag, save_index
            return False, -1
        return stepcheck

    @property
    def nsaves(self):
        return len(self.steps)

    '''
    def _get_stepcheck_log(self):
        base = self.base
        arr = self.steps
        ls = list(self.steps)
        tupl = tuple(self.steps)
        print('hello')
        def checkstep(step):
            print(tupl[step])
            # arr[0] +=  1
            # print(arr[0])
            if step==0 or step==1:
                return True, step
            prev_int = 1
            current = 1.0
            idx = 1
            while True:
                current *= base
                curr_int = int(current)
                if curr_int != prev_int:
                    idx += 1
                    if curr_int == step:
                        return True, idx
                    if curr_int > step:
                        return False, -1
                    prev_int = curr_int
        return checkstep

    def _get_stepcheck_lin(self):
        deltastep = self.deltastep
        def stepcheck(step):
            # flag = False
            # save_index = -1 # this is for python calls of the function
            if step%deltastep==0:
                # flag = True
                # save_index = step//deltastep
                return True, step//deltastep
            return False, -1
            # return flag, save_index
        return stepcheck

    def _get_stepcheck_geom(self):
        nsteps = self.stepmax
        npoints = self.npoints
        def stepcheck(step):
            if step==0:
                return True, 0
            xx = nsteps**(1.0/npoints)
            for idx in range(1, npoints):
                c = xx**(idx+1)-1
                if step==int(c):
                    return True, idx
            return False, -1
        return stepcheck
    '''

    # def _compute_steps(self):
    #     try:
    #         stepmax = self.stepmax
    #     except AttributeError:
    #         print('Error in steps computation; probably setup() has not been called yet')
    #     steps = []
    #     for step in range(stepmax+1):
    #         flag, _ = self.stepcheck_func(step)
    #         if flag: steps.append(step)
    #     # if stepmax not in steps: steps.append(stepmax)
    #     return steps

    # # TODO: deprecate this
    # def _compute_stepsall(self):
    #     try:
    #         stepmax = self.stepmax
    #         ntimeblocks = self.ntimeblocks
    #     except AttributeError:
    #         print('probably setup() has not been called yet')
    #     steps = []
    #     for step in range(stepmax+1):
    #         flag, _ = self.stepcheck_func(step)
    #         if flag: steps.append(step)
    #     overallsteps = []
    #     for i_block in range(ntimeblocks):
    #         for step in steps:
    #             overallstep = step+stepmax*i_block
    #             overallsteps.append(overallstep)
    #     if stepmax not in steps:
    #         overallsteps.append(stepmax*ntimeblocks)
    #     return overallsteps


    # @property
    # def nsavesall(self):
    #     try:
    #         return len(self.stepsall)
    #     except AttributeError:
    #         print('probably setup() has not been called yet')

