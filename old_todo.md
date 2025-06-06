Open Todo's have been transfered to issues after developer meeting 4/6-25. For reference old todo-list is listed below:

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
- [X] Molecules (Interface) Jesper, Ulf
- [ ] Read topology from file. Jesper, Ulf 
- [ ] Molecular stress, Jesper/Nick
- [X] Stress calculation for bonds. Perhaps warning is not included for angles, dihedrals, Nick/Jesper
- [X] Implement O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [X] Test O($N$) nblist update and mechanism for choosing between this and O($N^2$)
- [X] Allow more flexible/dynamical changing which data to be stored in Configuration, Nick
- [X] make GitLab/Hub address users, not ourselves (remove dev-state of page)
- [X] Reserve name on pypi, conda? Thomas
- [X] make installable by pip for all, by uploading to pypi, Thomas
- [ ] Use 'colarray' for scalars in Configuration (needs switching of dimensions)
- [ ] Configuration: include r_im in vectors?
- [ ] Requirements/dependencies, especially to use grid-sync, ADD LINK NUMBA DOC 
- [X] Auto-tuner, TBS
- [X] "grid to large for gridsync" should be handled ( CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE )
- [X] structure inside h5py: static info + a group for each runtime action (Lorenzo)
- [X] Test neighborlist integrity before and during simulations (after each timeblock)
- [X] Automatic reallocate larger neighborlist when needed, and redo simulation of the last timeblock
- [X] Benchmarking
- [ ] Charge (Water, SPCflexible), Jesper et al.
- [X] Remove NVE_Toxvaerd Nick
- [ ] Decide status of tools.save_configuration.py (is it necessary? move to Configuration.py ?) Lorenzo
- [X] Include support for different types in CalculatorStructureFactor, Ulf
- [X] More robust procedure for zeroing the forces (right now done by a neighbor list and requires that there be exactly one pair potential present), Thomas
- [X] Remove imports of rumdpy inside rumdpy modules, Lorenzo
- [ ] Decide status of gravity interaction, should it be included in planar_interactions, Thomas
- [ ] NVU integrator (tests missing), Mark

## TODO, long term:

- [ ] EAM metallic potentials, Nick
- [ ] Use sympy to differentiate pair-potentials. Was implemented but a factor of 2 slower, are float64's sneaking in?
- [ ] Add CPU support? (can it be done as a decorator?)
- [ ] Add AMD support?
- [ ] Thermostat on subsets of particles
- [ ] Constraints ???
