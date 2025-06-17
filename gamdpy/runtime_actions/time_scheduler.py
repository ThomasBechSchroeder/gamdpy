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

        # no specific kwarg is required
        if self.schedule=='log2':
            self.stepcheck_func = self._get_stepcheck_log2()

        # `base` kwarg is required
        elif self.schedule=='log':
            self.base = self._kwargs.get('base', np.exp(1.0))
            self.stepcheck_func = self._get_stepcheck_log()

        # `steps_between_output` kwarg is required
        elif self.schedule=='lin':
            self.deltastep = self._kwargs.get('steps_between_output', None)
            assert type(self.deltastep)==int, 'Invalid number of points'
            self.stepcheck_func = self._get_stepcheck_lin()

        # TODO
        # `npoints` kwarg is required
        elif self.schedule=='geom':
            self.npoints = self._kwargs.get('npoints', None)
            assert type(self.npoints)==int, 'Invalid number of points'
            self.stepcheck_func = self._get_stepcheck_geom()

        # TODO: "... but never is probably better than RIGHT now"
        # elif self.schedule=='custom':
        #     pass

        self.steps = self._compute_steps()
        self.stepsall = self._compute_stepsall()

    def _get_stepcheck_log2(self):
        def stepcheck(step):
            Flag = False
            save_index = -1 # this is for python calls of the function
            if step == 0:
                Flag = True
                save_index = 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step == c:
                    Flag = True
                    save_index = b + 1
            return Flag, save_index
        return stepcheck

    def _get_stepcheck_log(self):
        base = self.base
        def stepcheck(step):
            Flag = False
            save_index = -1 # this is for python calls of the function
            if step == 0:
                Flag = True
                save_index = 0
            else:
                exponent = np.int32(math.log(np.float32(step)) / math.log(base))
                c = np.int32(base ** exponent)
                if abs(step-c)==1: # this seems to be working
                    Flag = True
                    save_index = exponent
            return Flag, save_index
        return stepcheck

    def _get_stepcheck_lin(self):
        deltastep = self.deltastep
        def stepcheck(step):
            Flag = False
            save_index = -1 # this is for python calls of the function
            if step%deltastep==0:
                Flag = True
                save_index = step//deltastep
            return Flag, save_index
        return stepcheck

    # TODO
    def _get_stepcheck_geom(self):
        # variables
        def stepcheck(step):
            if step == 0:
                Flag = True
                save_index = 0
            else:
                pass
            return Flag, save_index
        return stepcheck

    def _compute_steps(self):
        try:
            stepmax = self.stepmax
        except AttributeError:
            print('probably setup() has not been called yet')
        steps = []
        for step in range(stepmax+1):
            Flag, _ = self.stepcheck_func(step)
            if Flag: steps.append(step)
        if stepmax not in steps: steps.append(stepmax)
        return steps

    def _compute_stepsall(self):
        try:
            stepmax = self.stepmax
            ntimeblocks = self.ntimeblocks
        except AttributeError:
            print('probably setup() has not been called yet')
        steps = []
        for step in range(stepmax+1):
            Flag, _ = self.stepcheck_func(step)
            if Flag: steps.append(step)
        overallsteps = []
        for i_block in range(ntimeblocks):
            for step in steps:
                overallstep = step+stepmax*i_block
                overallsteps.append(overallstep)
        if stepmax not in steps: 
            overallsteps.append(stepmax*ntimeblocks)
        return overallsteps

    @property
    def nsaves(self):
        try:
            return len(self.steps)
        except AttributeError:
            print('probably setup() has not been called yet')

    @property
    def nsavesall(self):
        try:
            return len(self.stepsall)
        except AttributeError:
            print('probably setup() has not been called yet')

    # def stepcheck_old(self, step):
    #     Flag = False
    #     if step == 0:
    #         Flag = True
    #         save_index = 0
    #     else:
    #         b = np.int32(math.log2(np.float32(step)))
    #         c = 2 ** b
    #         if step == c:
    #             Flag = True
    #             save_index = b + 1
    #     return Flag, save_index
