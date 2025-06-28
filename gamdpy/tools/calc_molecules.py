import numpy as np 
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
def calculateMolCenterMass(conf, molname: str):
    '''Compute molecular center of mass and molecule mass

    Parameters
    ----------
        1: Configuration instance
        2: Molecule type name 

    Output
    ------
        1: Molecular positions 
        2: Molecular masses

    Note: The positions are *not* wrapped in accordance with periodic boundary conditions 

    '''

    atom_idxs = np.array(conf.topology.molecules[molname], dtype=np.uint64)

    nmols, nuau = atom_idxs.shape[0], atom_idxs.shape[1]

    rmols = np.zeros( (nmols, 3) )
    mmols = np.zeros( nmols )

    __calc_molcm__(rmols, mmols, atom_idxs, nuau, conf['r'], conf['m'], conf.r_im, conf.simbox.get_lengths(), nmols) 

    return rmols, mmols


def calculateMolVelocity(conf, molname: str):
    '''Compute molecular center of mass velocity 

    Parameters
    ----------
        1: Configuration instance
        2: Molecule type name 

    Output
    ------
        1: Molecular center of mass velocity
    '''

    atom_idxs = np.array(conf.topology.molecules[molname], dtype=np.uint64)

    nmols, nuau = atom_idxs.shape[0], atom_idxs.shape[1]

    vmols = np.zeros( (nmols, 3) )
    __calc_molvcm__(vmols, atom_idxs, nuau, conf['v'], conf['m'], nmols)

    return vmols


def calculateMolDipole(conf, qatoms, molname: str):
    '''Compute molecular dipoles, molecular center of mass and molecule mass

    Parameters
    ----------
        1: Configuration instance
        2: Atoms charges for this molecule type (array with length of number of atoms)
        2: Molecule type name 

    Output
    ------
        1: Molecular dipoles
        2: Molecular positions 
        3: Molecular masses

    Note: The positions are *not* wrapped in accordance with periodic boundary conditions 

    '''
    # https://numba.readthedocs.io/en/stable/reference/deprecation.html
    # LC: it seems soon numba would only accept numba.typed.List and not regular python lists
    from numba.typed import List

    atom_idxs = np.array(conf.topology.molecules[molname], dtype=np.uint64)
    nmols, nuau = atom_idxs.shape[0], atom_idxs.shape[1]

    dmols = np.zeros( (nmols, 3) )
    rmols, mmols = calculateMolCenterMass(conf, molname)

    __calc_moldipole__(dmols, rmols, atom_idxs, nuau, conf['r'], List(qatoms), conf.r_im, conf.simbox.get_lengths(), nmols)

    return dmols, rmols, mmols 





