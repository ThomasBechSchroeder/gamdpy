# This is a library of predefined gamdpy classes which can be used in tests
__all__ = ["configuration_SC",
           "pairpot_LJ"]

from gamdpy import Configuration as _Configuration
# Set up a 3d configuration for a single component system
configuration_SC = _Configuration(D=3, compute_flags={'W':True})
configuration_SC.make_positions(N=1000, rho=0.754)
configuration_SC['m'] = 1.0  # Set all masses to 1.0
configuration_SC.randomize_velocities(temperature=2.0)

from gamdpy import apply_shifted_potential_cutoff as _apply_shifted_potential_cutoff
from gamdpy import LJ_12_6_sigma_epsilon as _LJ_12_6_sigma_epsilon
from gamdpy import PairPotential as _PairPotential
# Set up a LJ pair potential
_pairfunc = _apply_shifted_potential_cutoff(_LJ_12_6_sigma_epsilon)
_sig, _eps, _cut = 1.0, 1.0, 2.5
pairpot_LJ = _PairPotential(_pairfunc, params=[_sig, _eps, _cut], max_num_nbs=1000)

