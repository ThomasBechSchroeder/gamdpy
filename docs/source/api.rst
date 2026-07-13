===
API
===

.. toctree::
   :maxdepth: 1

The Simulation Class
--------------------

.. autoclass:: gamdpy.Simulation
   :members:

The Configuration Class
-----------------------

.. autoclass:: gamdpy.Configuration
   :members:

Simulation boxes
~~~~~~~~~~~~~~~~

An simulation box object is attached to an configuration object.

.. autoclass:: gamdpy.Orthorhombic
   :members:

.. autoclass:: gamdpy.LeesEdwards
   :members:


.. _integrators:

Integrators
-----------

One of the below integrators should be given as a parameter to the :class:`~gamdpy.Simulation` class.

Constant energy and volume
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gamdpy.NVE
   :members:
   :exclude-members: get_kernel, get_params

Constant temperature and volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gamdpy.NVT
   :members:
   :exclude-members: get_kernel, get_params

.. autoclass:: gamdpy.NVT_Langevin
   :members:
   :exclude-members: get_kernel, get_params

.. autoclass:: gamdpy.Brownian
   :members:
   :exclude-members: get_kernel, get_params

Constant temperature and pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gamdpy.NPT_Atomic
   :members:
   :exclude-members: get_kernel, get_params

.. autoclass:: gamdpy.NPT_Langevin
   :members:
   :exclude-members: get_kernel, get_params

Other integrators
~~~~~~~~~~~~~~~~~

.. autoclass:: gamdpy.SLLOD
   :members:
   :exclude-members: get_kernel, get_params

.. autoclass:: gamdpy.GradientDescent
   :members:
   :exclude-members: get_kernel, get_params

.. autoclass:: gamdpy.ActiveOUP
   :members:
   :exclude-members: get_kernel, get_params

.. _interactions:

Interactions
------------

Interactions are passed in a list to the :class:`~gamdpy.Simulation` class.
This will typically include a *pair potential* and *fix interactions* like gravity and walls.

Pair potential
~~~~~~~~~~~~~~

Evaluating pair potentials is typically computationally expensive.
For efficiency, instances of the class below only compute forces between two particles
when their separation is less than a specified cutoff distance.

.. autoclass:: gamdpy.PairPotential
   :members:
   :exclude-members: check_datastructure_validity, get_kernel, get_params

Pair potential functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gamdpy.LJ_12_6

.. autofunction:: gamdpy.LJ_12_6_sigma_epsilon

.. autofunction:: gamdpy.yukawa

.. autofunction:: gamdpy.gaussian_core_model

.. autofunction:: gamdpy.exponential_repulsion

.. autofunction:: gamdpy.harmonic_repulsion

.. autofunction:: gamdpy.hertzian

.. autofunction:: gamdpy.SAAP

.. autofunction:: gamdpy.universal_zbl_potential

Generators
^^^^^^^^^^

Generators return a function that can be used to calculate the potential energy and the force between two particles.

.. autofunction:: gamdpy.make_LJ_m_n

.. autofunction:: gamdpy.make_IPL_n

.. autofunction:: gamdpy.add_potential_functions

.. autofunction:: gamdpy.make_potential_function_from_sympy

Modifies
^^^^^^^^

Modifies are typically used to smoothly truncate the potential at a certain distance.

.. autofunction:: gamdpy.apply_shifted_potential_cutoff

.. autofunction:: gamdpy.apply_shifted_force_cutoff

.. autofunction:: gamdpy.apply_cubic_spline_cutoff

.. autofunction:: gamdpy.apply_gromacs_cutoff

Fixed interactions
~~~~~~~~~~~~~~~~~~

Fixed interactions (covalent bonds, angles, gravitational forces, walls, or tethers to an anchor points) are evaluated at every time step.
This is contrast to non-bonded pair interactions, which are only computed for particle pairs within the non-bonded cutoff distance.

Classes
^^^^^^^

.. autoclass:: gamdpy.Bonds

.. autoclass:: gamdpy.Angles

.. autoclass:: gamdpy.Tether

.. autoclass:: gamdpy.Relaxtemp

.. autoclass:: gamdpy.Planar

Generators
^^^^^^^^^^

.. autofunction:: gamdpy.make_planar_calculator

.. autofunction:: gamdpy.setup_planar_interactions

.. autofunction:: gamdpy.make_fixed_interactions


Bond functions
^^^^^^^^^^^^^^

A *bond potential* function is needed for the :class:`~gamdpy.Bonds` class.

.. autofunction:: gamdpy.harmonic_bond_function

Angle functions
^^^^^^^^^^^^^^^

An *angle potential* function is needed for the :class:`~gamdpy.Angles` class.

.. autofunction:: gamdpy.harmonic_angle_function

.. autofunction:: gamdpy.cos_angle_function



.. _runtime_actions:

Runtime Actions
---------------

A list of runtime actions are passed as an argument to the :class:`~gamdpy.Simulation` class.

.. autoclass:: gamdpy.TrajectorySaver
   :members:
   :exclude-members: get_params, get_poststep_kernel, get_prestep_kernel, initialize_before_timeblock, update_at_end_of_timeblock

.. autoclass:: gamdpy.ScalarSaver
   :members:
   :exclude-members: get_params, get_poststep_kernel, get_prestep_kernel, initialize_before_timeblock, update_at_end_of_timeblock

.. autoclass:: gamdpy.RestartSaver
   :members:
   :exclude-members: get_params, get_poststep_kernel, get_prestep_kernel, initialize_before_timeblock, update_at_end_of_timeblock

.. autoclass:: gamdpy.MomentumReset
   :members:
   :exclude-members: get_params, get_poststep_kernel, get_prestep_kernel, initialize_before_timeblock, update_at_end_of_timeblock

.. autoclass:: gamdpy.StressSaver
   :members:
   :exclude-members: get_params, get_poststep_kernel, get_prestep_kernel, initialize_before_timeblock, update_at_end_of_timeblock, make_zero_kernel_3

Calculators
-----------

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
--------------------------

Input and Output
~~~~~~~~~~~~~~~~

The TrajectoryIO class
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gamdpy.tools.TrajectoryIO
   :members:

IO functions
^^^^^^^^^^^^

.. autofunction:: gamdpy.configuration_from_rumd3

.. autofunction:: gamdpy.configuration_to_rumd3

.. autofunction:: gamdpy.configuration_to_lammps

Post-analysis tools
-------------------

.. autofunction:: gamdpy.tools.thermodynamics_NpT

.. autofunction:: gamdpy.tools.thermodynamics_NVT

.. autofunction:: gamdpy.tools.calc_dynamics

.. autofunction:: gamdpy.tools.calculate_molecular_center_of_masses

.. autofunction:: gamdpy.tools.calculate_molecular_velocities

.. autofunction:: gamdpy.tools.calculate_molecular_dipoles

Mathematical functions
----------------------

The below returns functions that can be executed fast in a GPU kernel.
As an example, they can be used to set a time-dependent target temperature.

.. autofunction:: gamdpy.make_function_constant

.. autofunction:: gamdpy.make_function_ramp

.. autofunction:: gamdpy.make_function_sin

Miscellaneous
-------------

.. autofunction:: gamdpy.select_gpu

.. autofunction:: gamdpy.get_default_sim

.. autofunction:: gamdpy.get_default_compute_plan

.. autofunction:: gamdpy.get_default_compute_flags

.. autofunction:: gamdpy.conversion_factors

.. autofunction:: gamdpy.plot_molecule

.. autofunction:: gamdpy.tools.print_h5_structure

.. autofunction:: gamdpy.tools.print_h5_attributes
