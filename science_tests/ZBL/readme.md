# Investigation of ZBL potential

A comparison between a LAMMPS simulation and a gamdpy simulation of Cobber (Z=29)
using the universal ZBL potential.
The initial structure is an FCC lattice with lattice constant of 4.2 Å.
An NVT simulation is done at T = 3500 K, where the crystal melts.
The figures below show results of liquid state points 
(discarding the first part of the simulation).


## Results

### LAMMPS

![lmp_u](./Data_lammps/potential_energy.png)

![lmp_p](./Data_lammps/pressure.png)

### Gamdpy

![gp_u](./energy.png)

![gp_p](./pressure.png)

![gp_uw](./UW.png)

![gp_rdf](./rdf.png)

![gp_msd](./msd.png)

## Run simulations and analysis

### LAMMPS simulation

Input files for LAMMPS are located in `Data_lammps`

    cd Data_lammps

Run the simulation with (single core)

    lmp -in in.lammps -log log.lammps

or (multi-core)

    mpirun -np 8 lmp -in in.lammps -log log.lammps

Inspect data and generate figures with

    python inspect_data.py

### Run simulation and analysis

    python zbl.py | tee zbl.log
    python analyse.py | tee analyse.log

### Clean

    rm zbl.h5

