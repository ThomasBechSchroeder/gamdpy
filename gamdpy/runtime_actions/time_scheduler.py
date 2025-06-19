import numpy as np
import math


class TimeScheduler():
    """
    Class used to:
        - define steps to save configuration at;
        - get functions for the numba kernel to check whether to save.

    An instance of TimeScheduler must be passed to TrajectorySaver, either explicitly or implicitly
    (i.e. passing appropriate keywords to TrajectorySaver, that will create an instance of TimeScheduler).

    Example:

    ..code-block:: python
        import gamdpy as gp
        scheduler = gp.TimeScheduler(schedule='log', base=1.5)
        runtime_actions = [gp.TrajectorySaver(schedule=scheduler),]

    alternatively

    ..code-block:: python
        import gamdpy as gp
        runtime_actions = [gp.TrajectorySaver(schedule='log', base=1.5),]

    See below for indications about kwargs required by different schedules. 
    Default is 'log2', which does not require any arguments.
    The list `runtime_actions` must then be passed to a Simulation instance.
    """

    def __init__(self, schedule='log2', **kwargs):

        self.known_schedules = ['log2', 'log', 'lin', 'geom']
        # self.known_schedules = ['log2', 'log', 'lin', 'geom', 'custom']
        self.schedule = schedule
        self._kwargs = kwargs

    def setup(self, stepmax, ntimeblocks):
        # This is necessary aside from __init__ because in TrajectorySaver
        # `steps_per_timeblock` is initialised only in a `setup` method

        # `stepmax` is by construction the same as `steps_per_timeblock` in TrajectorySaver
        # it makes sense to keep it as an attribute since it may be needed in the future for other schedules
        self.stepmax = stepmax
        self.ntimeblocks = ntimeblocks

        # no specific kwarg is required
        if self.schedule=='log2':
            self.stepcheck_func = self._get_stepcheck_log2()

        # `base` kwarg is accepted but not required (default is Euler number)
        elif self.schedule=='log':
            self.base = self._kwargs.get('base', np.exp(1.0))
            self.stepcheck_func = self._get_stepcheck_log()

        # `steps_between_output` kwarg is required
        elif self.schedule=='lin':
            deltastep = self._kwargs.get('steps_between_output', None)
            npoints = self._kwargs.get('npoints', None)
            if deltastep is not None:
                self.deltastep = deltastep
                self.stepcheck_func = self._get_stepcheck_lin()
            elif npoints is not None:
                raise NotImplementedError("Passing 'npoints' to 'lin' schedule should be possible")
                #self.npoints = npoints
            else:
                raise TypeError("'steps_between_output' or 'npoints' is required for schedule 'lin'")

        # `npoints` kwarg is required
        elif self.schedule=='geom':
            self.npoints = self._kwargs.get('npoints', None)
            if npoints is not None:
                self.npoints = npoints
                self.stepcheck_func = self._get_stepcheck_geom()

        # TODO: using custom steps would require passing arrays to stepcheck functions, which is a potential issue
        # elif self.schedule=='custom':
        #     pass

        self.steps = self._compute_steps()
        # self.stepsall = self._compute_stepsall()
        if self.schedule=='geom':
            # the 'geom' schedule must return each save index only once; this must be after _compute_steps()
            assert self.nsaves==self.npoints, 'Too many points, schedule distorsion; try fewer points'

    def _get_stepcheck_log2(self):
        def stepcheck(step):
            flag = False
            idx = -1 # this is for python calls of the function
            if step == 0:
                flag = True
                idx = 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step == c:
                    flag = True
                    idx = b + 1
            return flag, idx
        return stepcheck

    def _get_stepcheck_log(self):
        base = self.base
        def stepcheck(step):
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
                        break
                    prev_int = curr_int            
            return False, -1
        return stepcheck

    def _get_stepcheck_lin(self):
        deltastep = self.deltastep
        def stepcheck(step):
            if step%deltastep==0:
                return True, step//deltastep
            return False, -1
        return stepcheck

    def _get_stepcheck_geom(self):
        stepmax = self.stepmax
        npoints = self.npoints
        def stepcheck(step):
            if step==0:
                return True, 0
            xx = stepmax**(1.0/npoints)
            for idx in range(1, npoints):
                c = xx**(idx+1)-1
                if step==int(c):
                    return True, idx
            return False, -1
        return stepcheck

    def _compute_steps(self):
        steps = []
        # we want to include the last step IFF it raises a True flag
        for step in range(self.stepmax+1):
            flag, _ = self.stepcheck_func(step)
            if flag: steps.append(step)
        # if stepmax not in steps: steps.append(stepmax)
        return np.unique(np.array(steps))

    @property
    def nsaves(self):
        return len(self.steps)

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

