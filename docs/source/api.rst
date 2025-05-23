API
###


The Simulation Class
********************

.. autoclass:: gamdpy.Simulation
   :members:

The Configuration Class
***********************

.. autoclass:: gamdpy.Configuration
   :members:

Integrators
***********

.. autoclass:: gamdpy.NVE
   :members:

.. autoclass:: gamdpy.NVT
   :members:

.. autoclass:: gamdpy.NVT_Langevin
   :members:

.. autoclass:: gamdpy.NPT_Atomic
   :members:

.. autoclass:: gamdpy.NPT_Langevin
   :members:

.. autoclass:: gamdpy.SLLOD
   :members:

.. autoclass:: gamdpy.NVU_RT
   :members:

Interactions
************

Pair potentials
===============

.. autoclass:: gamdpy.PairPotential
   :members:

Functions (pair potentials)
---------------------------

.. autofunction:: gamdpy.LJ_12_6

.. autofunction:: gamdpy.LJ_12_6_sigma_epsilon

.. autofunction:: gamdpy.harmonic_repulsion

.. autofunction:: gamdpy.hertzian

.. autofunction:: gamdpy.SAAP

Functions (bonds)
---------------------------

.. autofunction:: gamdpy.harmonic_bond_function

Generators
----------

Generators return a function that can be used to calculate the potential energy and the force between two particles.

.. autofunction:: gamdpy.make_LJ_m_n

.. autofunction:: gamdpy.make_IPL_n

.. autofunction:: gamdpy.make_potential_function_from_sympy

Modifies
--------

Modifies are typically used to smoothly truncate the potential at a certain distance.

.. autofunction:: gamdpy.apply_shifted_potential_cutoff

.. autofunction:: gamdpy.apply_shifted_force_cutoff

Fixed interactions
==================

Classes
-------

.. autoclass:: gamdpy.Bonds

.. autoclass:: gamdpy.Angles

.. autoclass:: gamdpy.Tether

.. autoclass:: gamdpy.Gravity

.. autoclass:: gamdpy.Relaxtemp

Generators
----------

.. autofunction:: gamdpy.make_planar_calculator

.. autofunction:: gamdpy.setup_planar_interactions

.. autofunction:: gamdpy.make_fixed_interactions

Runtime Actions
***************

.. autoclass:: gamdpy.TrajectorySaver

.. autoclass:: gamdpy.ScalarSaver

.. autoclass:: gamdpy.RestartSaver

.. autoclass:: gamdpy.MomentumReset

Calculators
***********

.. autoclass:: gamdpy.CalculatorRadialDistribution
   :members:

.. autoclass:: gamdpy.CalculatorStructureFactor
   :members:

.. autoclass:: gamdpy.CalculatorWidomInsertion
   :members:

.. autoclass:: gamdpy.CalculatorHydrodynamicCorrelations
   :members:

.. autoclass:: gamdpy.CalculatorHydrodynamicProfile
   :members:

Tools and helper functions
**************************

Input and Output
================

The TrajectoryIO class
----------------------

.. autoclass:: gamdpy.tools.TrajectoryIO
   :members:

IO functions
------------

.. autofunction:: gamdpy.tools.save_configuration

.. autofunction:: gamdpy.configuration_to_hdf5

.. autofunction:: gamdpy.configuration_from_hdf5

.. autofunction:: gamdpy.configuration_to_rumd3

.. autofunction:: gamdpy.configuration_from_rumd3

.. autofunction:: gamdpy.configuration_to_lammps


Mathematical functions
======================

The below returns functions that can be executed fast in a GPU kernel.
As an example, they can be used to set a time-dependent target temperature.

.. autofunction:: gamdpy.make_function_constant

.. autofunction:: gamdpy.make_function_ramp

.. autofunction:: gamdpy.make_function_sin

Extract data
============

.. autofunction:: gamdpy.extract_scalars

Miscellaneous
*************

.. autofunction:: gamdpy.get_default_sim

.. autofunction:: gamdpy.get_default_compute_plan

.. autofunction:: gamdpy.get_default_compute_flags

.. autofunction:: gamdpy.plot_molecule
