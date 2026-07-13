# Change log for `gamdpy`

## Version 0.8.4dev

### Bug fixes

### New features


## Version 0.8.3, May 18, 2026

### Bug fixes

* Fixed bug in make_lattice that gave wrong density for D!=3 and rho!=None.
* Fixed bug in calc_dynmamics when using Lees-Edwards boundary conditions.

### New features

* New examples: 
   - **water.py** : Simulation of the SPC/Fw model
   - **active_ornstein_uhlenbeck.py** : Purely repulsive active particles forming dense clusters (MIPS).
   - **zbl.py** : Simulation of Cu using universal Ziegler–Biersack–Littmark (ZBL) potential.
   - **lennard_jones_argon.py** : Lennard-Jones conversion from Argon units to real units
   - **lennard_jones_argon_real_units.py** : Simulation of Lennard-Jones Argon using real units
   - **LJchain_continue.py** : Continuation of LJ chains simulation.
   - **isomorph_forcemethod.py** : An isomorph is traced out using the force method
   - **analyze_NVT_thermodynamics.py** : Calculates thermodynamical properties of a constant NVT simulation.
   - **analyze_NpT_thermodynamics.py** Calculates thermodynamical properties of a constant NpT simulation.
* New Pair-potentials: 
   - LJ_coulomb_sf
   - Universal ZBL Potential 
   - Gaussian-core model
   - Yukawa, 
   - Exponential repulsion 
   - harmonic repulsion
   - Hertzian repulsion
* Interactions
   - Embedded atom methhod (EAM) potential with parameters for 16 elements, based on Zhou, Johnsonm, Wadley, Phys. Rev. B 69, 144113  (2004) 
   - Smooth Cubic spline truncations (gamdpy.apply_cubic_spline_cutoff)
   - Smooth Gromacs style truncations (gamdpy.apply_gromacs_cutoff)
   - Harmonic bond angle potential
   - Possibility to partially exclude pairs interactions as for example for atoms 1,4 of dihedral interactions.
   - Neighbor List for big systems, Linked-List, now works in 2 and 3 dimensions
* Integrator for Active Ornstein-Uhlenbeck Particle (gamdpy.ActiveOUP)
* Mechanism for integators to save their internal state for the purpose of correctly restarting an interrupted simulation  (for NVT and NPT_Langevin so far). 
* Compression of simulation box implemented as a method in class Simulation ( Simulation.compress() )
* Tools for unit conversions (gp.conversion_factors)
* Tools for analyzing thermodynamics fluctuations

## Version 0.8.2, Aug 3, 2025

### Bug fixes

* Incorrect application of shifted-force cutoff solved

### New features

* Integrator for Brownian dynamics.
* Integrator for gradient descent.
* Integrator for NVU dynamics.
* Variable strain rate for SLLOD
* calc_dynamics can handle Lees-Edwards boundary conditions.
* Tabulated pair potentials.
* extract_scalars superseesed by ScalarSaver.extract(), see examples/read_scalar_data_from_h5.py
* TrajectorySaver enhanced to allow saving of velocities and forces.
* TimeScheduler's implemented to control when output is done (for now only in TrajectorySaver).
* examples/visualize.py for 3D visualization using ovito.
* examples/plot_pkls.py for plotting data from several simulations.

### Other

* Updates to output h5 format
* Update the format of the dictionary returned by CalculatorRadialDistribution.read()
  
## Version 0.8.1, Jun 12, 2025
First release of the package on pypi.
