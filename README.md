#  **gamdpy [gam-dee-pai]** 
## GPU Accelerated Molecular Dynamics in Python

Gamdpy implements molecular dynamics on GPU's in Python, relying heavily on the numba package ([numba.org](https://numba.pydata.org/)) which does JIT (Just-In-Time) compilation both to CPU and GPU (cuda). 
The gamdpy package being pure Python (letting numba do the heavy lifting of generating fast code) results in an extremely extendable package: simply by interjecting Python functions in the right places, 
the (experienced) user can extend most aspect of the code, including new integrators, new pair-potentials, new properties to be calculated during simulation, new particle properties, ...  

[The Users Guide (gamdpy.readthedocs.io)](https://gamdpy.readthedocs.io)

[Installation](installation.md)

[Examples](examples/README.md)

[Benchmarks (preliminary)](benchmarks/README.md)

[Info for developers](info_for_developers.md)

## Overall structure of the package

### 1. Configuration
A class containing all relevant information about a configuration, including the simulation box (class sim_box). 
- Vectors (r, v, f, etc): (N,D) float array storing D-dimensional vector for each particle 
- Scalars (mass, kinetic energy, etc.): (N,) float array storing scalar for each particle 
- sim_box (data describing box + functions implementing how to calculate distances and how to implement BC). For now:orthorombic (default) and lees_edwards 

### 2. Integrators
Classes implementing a simulation algorithm. Currently implemented: 
- class NVE
- class NVT : Nose-Hoover thermostat 
- class NVT_Langevin
- class NPT_Atomic
- class NPT_Langevin
- class NVU_RT (Experimental)

Temperature/Pressure can be controlled by a user-supplied function, see examples/kablj.py

### 3. Interactions
Classes implementing interactions that can be applied to particles in the system:  
- class PairPotential (stores potential parameters and the neighbour list to use (class NbList)
- fixed interactions (interactions known beforehand): 
  - bonds, angles, and dihedrals
  - planar interactions: smooth walls, gravity, electric fields, ...
  - point interactions, e.g., tethering

An interaction is responsible for keeping any internal datastructures up to date (in particular: class PairPotential is responsible for keeping its neighbor-list (class NbList up to date). 

### 4. Runtime actions
Classes implementing actions on the configuration which are not related to the interactions or the integration of the equation of motion.
These classes include momentum reset and savers.
- class TrajectorySaver
- class ScalarSaver
- class MomentumReset

### 5. Simulation
This class takes a Configuration, an Integrator, a (list of) Interaction(s) and a list of Runtime actions and sets up a simulation. 
Performing simulation is done by a method of this class.

### 6. Evaluator
Takes a Configuration and a (list of) Interaction(s), and evaluates properties and assign them to the appropiate fields in the configuration.

