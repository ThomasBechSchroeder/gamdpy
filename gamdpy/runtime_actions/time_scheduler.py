import numpy as np
import math

"""
Time scheduler classes. They are used to:
    - define steps to save configuration at;
    - get functions for the numba kernel to check whether to save.

The BaseScheduler class defines common methods and attributes, while only
child classes instances can be passed to TrajectorySaver for it to be functional.

Example:

    ..code-block:: Python
        import gamdpy as gp
        runtime_actions = [gp.TrajectorySaver(scheduler=gp.Logarithmic2()),]

Then runtime_actions list can be passed to a Simulation instance.
"""


class BaseScheduler():
    """
    Time scheduler abstract base class. It mainly contains the setup()
    method that is called by TrajectorySaver.
    
    Child classed must implement the _get_stepcheck() method to return the 
    function to be compiled with numba.njit(). Such function must take only
    the current `step` as input and return a flag and the save index.
    Here is a template to define a new child class.

        ..code-block:: Python

            class MyScheduler(BaseScheduler):
            
                def __init__(self, mykeyword):
                    super().__init__()
                    self.mykeyword = mykeyword
                    self.kwargs = super().get_kwargs()

                def _get_stepcheck(self):
                    # here all attributes can be retrieved
                    def stepcheck(step):
                        # define steps to save at and their indexes
                        pass
                    return stepcheck
    """

    def __init__(self):
        self.known_schedules = ['log2', 'log', 'lin', 'geom']

    def setup(self, stepmax, ntimeblocks):
        # This is necessary aside from __init__ because in TrajectorySaver
        # `steps_per_timeblock` is initialised only in a `setup` method

        # `stepmax` is by construction the same as `steps_per_timeblock` in TrajectorySaver
        # it makes sense to keep it as an attribute since it may be needed in the future for other schedules
        self.stepmax = stepmax
        self.ntimeblocks = ntimeblocks # currently not used

        self.stepcheck_func = self._get_stepcheck()
        self.steps, self.indexes = self._compute_steps()

    def get_kwargs(self):
        import inspect
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        kwargs = {arg: values[arg] for arg in args if arg != 'self'}
        return kwargs

    def _get_stepcheck(self):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")

    def _compute_steps(self):
        steps = []
        indexes = []
        # we want to include the last step IFF it raises a True flag
        for step in range(self.stepmax+1):
            flag, idx = self.stepcheck_func(step)
            if flag: 
                steps.append(step)
                indexes.append(idx)
        return np.array(steps), np.array(indexes)

    @property
    def nsaves(self):
        return len(self.steps)


class Log2(BaseScheduler):

    def __init__(self):
        super().__init__()
        self.kwargs = super().get_kwargs()

    def _get_stepcheck(self):
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


class Log(BaseScheduler):
    
    # def __init__(self, base=np.exp(1.0)):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def _get_stepcheck(self):
        base = self.base
        def stepcheck(step):
            # determine flag
            flag = False
            if step==0 or step==1:
                flag = True
            else:
                virtual_index = int(np.log(step+1)/np.log(base))
                virtual_step = int(base**virtual_index)
                if virtual_step==step:
                    flag = True
            if not flag:
                return False, -1
            idx = 0
            # find the save index by counting previous true flags
            for i in range(1, step+1):
                if i==0 or i==1:
                    idx += 1
                else:
                    virtual_index = int(np.log(i+1)/np.log(base))
                    virtual_step = int(base**virtual_index)
                    if virtual_step==i:
                        idx += 1
            return True, idx
        return stepcheck


class Lin(BaseScheduler):

    def __init__(self, steps_between=None, npoints=None):
        super().__init__()
        self.steps_between = steps_between
        self.npoints = npoints

    def _get_stepcheck(self):
        # this must go here because the needed super() attributes are defined in setup(), not __init__()
        if self.steps_between is not None and self.npoints is None:
            self.deltastep = self.steps_between
        elif self.npoints is not None:
            # this needs testing
            self.deltastep = self.stepmax // self.npoints
        deltastep = self.deltastep
        def stepcheck(step):
            if step%deltastep==0:
                return True, step//deltastep
            return False, -1
        return stepcheck


class Geom(BaseScheduler):

    def __init__(self, npoints):
        super().__init__()
        self.npoints = npoints

    def _get_stepcheck(self):
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
