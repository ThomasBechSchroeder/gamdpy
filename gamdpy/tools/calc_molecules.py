import numpy
import numpy as np

import gamdpy
import gamdpy as gp 
from numba import jit
import math
import cmath

# Work horses
@jit(nopython=True)
def __calc_molcm__(rmols, mmols, atomindices, nuau, ratoms, matoms, images, lbox, nmols):

    for i in range(nmols):
        rmols[i,:] = 0.0
        mmols[i] = 0.0
        for n in range(nuau):
            aidx = atomindices[i,n] 
            rmols[i,:] += matoms[aidx]*( ratoms[aidx,:] + images[aidx,:]*lbox[:] )
            mmols[i] += matoms[aidx]

        rmols[i,:] = rmols[i,:]/mmols[i] # This is not translated into the simbox!


@jit(nopython=True)
def __calc_molvcm__(vmols, atomindices, nuau, vatoms, matoms, nmols):

    for i in range(nmols):
        vmols[i,:] = 0.0
        mass = 0.0
        for n in range(nuau):
            aidx = atomindices[i,n] 
            vmols[i,:] += matoms[aidx]*vatoms[aidx,:] 
            mass += matoms[aidx]
        
        vmols[i,:] = vmols[i,:]/mass


@jit(nopython=True)
def __calc_moldipole__(dmols, rmols, atomindices, nuau, ratoms, qatoms, images, lbox, nmols):

    for i in range(nmols):
        dmols[i,:] = 0.0
        
        for n in range(nuau):
            aidx = atomindices[i,n] 
            ratomtrue = ratoms[aidx,:] + images[aidx,:]*lbox[:]

            dmols[i,:] += qatoms[n]*(ratomtrue - rmols[i,:])
           

# Wrappers
def calculate_molecular_center_of_masses(configuration: gamdpy.Configuration, molecule: str):
    """ Compute molecular center-of-mass positions and masses.

    Parameters
    ----------
    configuration : Configuration
        Simulation configuration instance containing topology, atomic positions
        array (`conf['r']`), atomic masses array (`conf['m']`), image flags
        (`conf.r_im`), and simulation box lengths.
    molecule : str
        Name of the molecule type to process.

    Returns
    -------
    positions : ndarray, shape (n_molecules, 3)
        Center-of-mass coordinates of each molecule.
    masses : ndarray, shape (n_molecules,)
        Total mass of each molecule.

    Notes
    -----
    The returned positions are *not* wrapped according to periodic boundary conditions.
    """

    atom_idxs = np.array(configuration.topology.molecules[molecule], dtype=np.uint64)

    nmols, nuau = atom_idxs.shape[0], atom_idxs.shape[1]

    rmols = np.zeros( (nmols, 3) )  # URP: It looks like 3D is hard-coded. Can it be generalize?
    mmols = np.zeros( nmols )

    __calc_molcm__(rmols, mmols, atom_idxs, nuau, configuration['r'], configuration['m'], configuration.r_im, configuration.simbox.get_lengths(), nmols)

    return rmols, mmols


def calculate_molecular_velocities(configuration: gamdpy.Configuration, molecule: str):
    """ Compute molecular center-of-mass velocities.

    Parameters
    ----------
    configuration : Configuration
        Configuration instance containing topology, atomic velocities, and atomic masses.
    molecule : str
        Name of the molecule type to process.

    Returns
    -------
    velocities : ndarray, shape (n_molecules, 3)
        Center-of-mass velocity vectors for each molecule.

    Notes
    -----
    Velocities are computed by combining atomic velocities and masses for each molecule.
    """

    atom_idxs = np.array(configuration.topology.molecules[molecule], dtype=np.uint64)

    nmols, nuau = atom_idxs.shape[0], atom_idxs.shape[1]

    vmols = np.zeros( (nmols, 3) )  # URP. It Looks like 3D is hard-coded. Can it be generalized?
    __calc_molvcm__(vmols, atom_idxs, nuau, configuration['v'], configuration['m'], nmols)

    return vmols


def calculate_molecular_dipoles(configuration: gamdpy.Configuration, atom_charges: numpy.ndarray, molecule: str):
    r""" Compute molecular dipole moments, centers of mass, and masses.

    Parameters
    ----------
    configuration : Configuration
        A Configuration instance, containing topology, coordinates.
    atom_charges : array_like of float, shape (n_atoms,)
        Partial charges for each atom in the specified molecule type.
    molecule : str
        Name of the molecule type to process.

    Returns
    -------
    dipoles : ndarray, shape (n_molecules, 3)
        Dipole moment vectors for each molecule.
    positions : ndarray, shape (n_molecules, 3)
        Center-of-mass coordinates of each molecule. These are not wrapped
        according to periodic boundary conditions.
    masses : ndarray, shape (n_molecules,)
        Total mass of each molecule.

    Notes
    -----
    The returned positions are *not* wrapped according to periodic boundary conditions.
    """
    # https://numba.readthedocs.io/en/stable/reference/deprecation.html
    # LC: it seems soon numba would only accept numba.typed.List and not regular python lists
    from numba.typed import List

    atom_idxs = np.array(configuration.topology.molecules[molecule], dtype=np.uint64)
    nmols, nuau = atom_idxs.shape[0], atom_idxs.shape[1]

    dmols = np.zeros( (nmols, 3) )   # URP: It looks like 3D have been hard-coded, can it be generalized?
    rmols, mmols = calculate_molecular_center_of_masses(configuration, molecule)

    __calc_moldipole__(dmols, rmols, atom_idxs, nuau, configuration['r'], List(atom_charges), configuration.r_im, configuration.simbox.get_lengths(), nmols)

    return dmols, rmols, mmols 





