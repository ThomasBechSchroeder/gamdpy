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

        self.known_schedules = ['log2', 'log', 'lin', 'geom', 'custom']
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
            self.base = 2
            self.steps = self._steps_log()

        # `base` kwarg is accepted but not required
        elif self.schedule=='log':
            self.base = self._kwargs.get('base', np.exp(1.0))
            self.steps = self._steps_log()

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

        # `npoints` kwarg is required
        elif self.schedule=='geom':
            npoints = self._kwargs.get('npoints', None)
            if npoints is not None:
                self.npoints = npoints
                self.steps = self._steps_geom()
                assert self.nsaves==self.npoints, 'Too many points, schedule distorsion; try fewer points'
            else:
                raise TypeError("'npoints' is required for schedule 'geom'")

        # `steps` kwarg is required
        elif self.schedule=='custom':
            steps = self._kwargs.get('steps', None)
            if steps is not None:
                assert isinstance(steps, (list, np.array)), "'steps' must be 'list' or 'numpy.array'"
                self.steps = np.unique(np.array(steps))
            else:
                raise TypeError("'steps' is required for schedule 'custom'")

        if self.schedule != 'log2':
            self.stepcheck_func = self._get_stepcheck()
        else:
            self.stepcheck_func = self._get_stepcheck_log2()

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
            if step == 0:
                return True, 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step==c:
                    return True, b+1
            return False, -1
        return stepcheck

    @property
    def nsaves(self):
        return len(self.steps)
