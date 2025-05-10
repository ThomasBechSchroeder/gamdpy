#  **gamdpy [gam-dee-pai]** 
## GPU Accelerated Molecular Dynamics in Python

Gamdpy implements molecular dynamics on GPU's in Python, relying heavily on the numba package ([numba.org](https://numba.pydata.org/)) which does JIT (Just-In-Time) compilation both to CPU and GPU (cuda). 
The gamdpy package being pure Python (letting numba do the heavy lifting of generating fast code) results in an extremely extendable package: simply by interjecting Python functions in the right places, 
the (experienced) user can extend most aspect of the code, including new integrators, new pair-potentials, new properties to be calculated during simulation, new particle properties, ...  

[The Users Guide (gamdpy.readthedocs.io)](https://gamdpy.readthedocs.io)

[Installation](installation.md)

[Examples](examples/README.md)

[Benchmarks (preliminary)](benchmarks/README.md)

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
- class ConfigurationSaver
- class ScalarSaver
- class MomentumReset

### 5. Simulation
This class takes a Configuration, an Integrator, a (list of) Interaction(s) and a list of Runtime actions and sets up a simulation. 
Performing simulation is done by a method of this class.

### 6. Evaluator
Takes a Configuration and a (list of) Interaction(s), and evaluates properties and assign them to the appropiate fields in the configuration.

# Info for developers

## Implementing on GPU using numba.cuda

- Inherited from rumd3: pb (particles per block), tp (threads per particle)
- Hoping to avoid from rumd3: sorting (gets too complicated).

Synchronization is of the utmost importance for correctness. For example, all forces need to be calculated before the integrator starts moving the particles. 
Traditionally (and in rumd3) this is done by kernel-calls (a kernel is a function running on the GPU, called from the CPU): it is guaranteed that one kernel finishes before the next begins (unless you explicitly ask otherwise). 

Unfortunately, kernel calls are slow, especially in numba.cuda (as compared to c++.cuda). 
A rough estimate is that the maximum number of time steps per second (TPS) that can be achieved using kernel calls for synchronization is about 5000 - a far cry from the ~100.000 TPS that can be achieved for small systems using "grid synchronization": Calling 'grid.sync()' inside a kernel ensures all threads in the grid get synchronised (i.e., no threads proceed beyond this statement before all threads have reached this statement). 

There is a limit to how many thread blocks can be used with grid synchronization, which makes it inefficient at large system sizes, so we need to be able to choose between the two ways of synchronization. 
A good place to see how this is done without implementing all functions twice is in 'integrators.py'
 
## TODO, short term
- [x] Break a single file into several files/modules 
- [x] Start using GIT
- [x] Make it into a python package that can be installed locally by pip
- [x] cut = 2.5 hardcoded - change that! -> 'max_cut' now part of interaction parameters for pair-potential 
- [x] Implement springs as an example of 'fixed interactions' (needs testing for gridsync==False). 
- [x] Implement (fixed) planar interactions, e.g., smooth walls, gravity, and electric fields.
- [x] Implement exclusion list 
- [x] upload to GitLab
- [x] Use 'colarray' for vectors in Configuration
- [x] Move r_ref from Configuration to nblist

## TODO, before summer interns arrive
- [X] SLLOD (stress, LEBC), Nick
- [X] Bonds interface
- [X] Implement other fixed interactions: point interactions (tethered particles). Jesper
- [X] Finish Atomic interface (runtime actions...) Thomas
- [X] Momentum resetting (remove default) Nick
- [X] Read rumd3 & others configurations Nick
- [X] Testing (Framework, doctest), Ulf & Thomas
- [X] Testing using gitlab CI, Lorenzo
- [X] Include scalar column names in output, Lorenzo
- [X] Include vector column names in output, Lorenzo
- [X] Documentation/Tutorials/Best practices
- [X] Generalize make_configuration to different lattices, Ulf
- [X] Read configurations from file (Lorenzo: added function load_output in tools)
- [X] Runtime actions to include conf_saver and scalar_output, Thomas
- [X] Per particles thermostat using interaction
- [X] Post analysis, RDF and Sq 
- [X] Post analysis for multicomponents, Lorenzo/Danqui


## TODO or decide not necessary, before paper/'going public'
- [X] Molecules (angles, dihedrals) Jesper
- [ ] Molecules (Interface) Jesper, Ulf
- [ ] Molecular stress, Jesper/Nick
- [X] Stress calculation for bonds. Perhaps warning is not included for angles, dihedrals, Nick/Jesper
- [X] Implement O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [X] Test O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [X] Allow more flexible/dynamical changing which data to be stored in Configuration, Nick
- [ ] make GitLab/Hub address users, not ourselves (remove dev-state of page)
- [ ] Reserve name on pypi, conda? Thomas
- [ ] make installable by pip for all, by uploading to pypi, Thomas
- [ ] Use 'colarray' for scalars in Configuration (needs switching of dimensions)
- [ ] Configuration: include r_im in vectors
- [ ] Requirements/dependencies, especially to use grid-sync, ADD LINK NUMBA DOC 
- [X] Auto-tuner, TBS
- [X] "grid to large for gridsync" should be handled ( CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE )
- [ ] structure inside h5py: static info + a group for each runtime action (Lorenzo)
- [X] Test neighborlist integrity before and during simulations (after each timeblock)
- [X] Automatic reallocate larger neighborlist when needed, and redo simulation of the last timeblock
- [X] Benchmarking
- [ ] Charge (Water, SPCflexible), Jesper et al.
- [X] Remove NVE_Toxvaerd Nick
- [ ] Decide status of tools.save_configuration.py (is it necessary? move to Configuration.py ?) Lorenzo
- [ ] Include support for different types in CalculatorStructureFactor, Ulf
- [X] More robust procedure for zeroing the forces (right now done by a neighbor list and requires that there be exactly one pair potential present), Thomas
- [X] Remove imports of rumdpy inside rumdpy modules, Lorenzo
- [ ] Decide status of gravity interaction, should it be included in planar_interactions, Thomas
- [ ] NVU integrator (tests missing), Mark

## TODO, long term:
- [ ] Constraints
- [ ] EAM metallic potentials, Nick
- [ ] Use sympy to differentiate pair-potentials. Was implemented but a factor of 2 slower, are float64's sneaking in?
- [ ] Add CPU support (can it be done as a decorator?)
- [ ] Thermostat on subsets of particles

## Various tools/strategies we will use
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- Git ( https://git-scm.com/doc, https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell ).
- Sphinx ( https://www.sphinx-doc.org/ ) for documentation, 
- ... to be hosted on readthedocs ( https://about.readthedocs.com/ ). Model: https://numba.readthedocs.io.
- Hypothesis (property-based testing, https://hypothesis.readthedocs.io ).
- doctest (no more failing examples in examples/docs!, see colarray.py for example).
- Jupyter notebooks for tutorials. Testing: nbmake?, testbook?
- Automatic testing upon uploading (CI). How to get access to GPU's?.
- Systematic benchmarking. Substantial degradation in performance will be considered a bug.

## Checklist for developing a new feature
- Copy code that resembles what you want to do and modify it to your needs.
- Write tests in a file placed in tests (run pytest to check that it works).
- Write an example and place it in examples, add it to the examples/README.md
- Write documentation in the docstrings of the code (run doctests to check that it works).
- Include the new feature in the documentation, e.g., you may need to edit docs/source/api.rst

## Some git cmd which might be useful

Getting hash of your master (Head)
```sh
git log --pretty=format:'%h' -n 1
```

Creating a public branch (on the repository) starting from the current master / branch
```sh
git checkout -b your_branch
git push -u origin your_branch
```

Difference in a single file between branches. Can use hash instead of master / branch
```sh
git diff origin branch -- rumdpy/Simulation.py
git diff hash1 hash2 -- rumdpy/Simulation.py
```
List the files that are different in two branches
```sh
git diff --name-only origin branch 
```
Show version of a file in another branch
```sh
git show branch:file
```

Reset the last commit. It will not delete any file but will go back removing the last commit and the adding related to that commit
```sh
git reset HEAD~
```

Reset all changes.
```sh
git reset HEAD --hard
```

## How to test the code
Running `pytest` in root (gamdpy) directory will run all tests.
This will use the settings in the file `pytest.ini`.

Install needed packages:

```sh
pip install pytest hypothesis scipy ipywidgets
```

Running pytest:

```sh
python3 -m pytest
```

Running all tests typically takes several minutes.
Slow tests can be skipped by running (test functions decorated with `@pytest.mark.slow`):

```sh
python3 -m pytest -m "not slow"
```

Running pytest with an -x option makes pytest stop after the first failure
```sh
pytest -x
```

Running pytest starting from the last failed test
```sh
pytest --lf
```

### Test of specific features

Test scripts are located in the `tests` directory. Most can be executed (in a verbose mode) as a script:

```bash
python3 tests/test_examples.py
```

Running doctest of a single file:

```bash
python3 -m doctest -v gamdpy/calculators/calculator_radial_distribution.py
```

### Coverage of tests

To see what part of the code is covered:

```sh
pip install coverage
coverage run -m pytest
```

After the tests are finished, do:

```sh
coverage report -m
```

or `coverage html`.

## Building documentation

To building the documentation using sphinx, https://www.sphinx-doc.org
(needs `pip install myst_nb pydata_sphinx_theme`)

Install the necessary packages:

```sh
pip install sphinx myst_nb pydata_sphinx_theme
```

Build documentation webpage:

```sh
cd docs
make html
```

Open a webpage with firefox (or your favorite browsers):

```sh
firefox build/html/index.html
```

Clean the build directory (optional):

```sh
make clean
```


