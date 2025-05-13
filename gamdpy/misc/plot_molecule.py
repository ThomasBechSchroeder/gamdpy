import matplotlib.pyplot as plt
import numpy as np

def plot_molecule(top, positions, particle_types, filename="molecule.pdf", block=False):
    '''
    This function write a pdf file with a drawing of the molecule 

    Parameters
    ----------
    top : gamdpy topology object

    positions : list or numpy array with positions of all atoms

    particle_types : types of the molecule

    filename :  name of the output pdf file, default is molecule.pdf

    block: boolean, default False. If True shows plot and blocks script until display window is closed

    '''
    fig = plt.figure()
    fig.suptitle("This is the molecule you're going to simulate", fontsize=16)
    ax = fig.add_subplot(projection='3d')

    pos = np.array(positions)
    num_of_type = np.array([np.size(np.nonzero(particle_types==value)) for value in np.unique(particle_types)])

    D = pos.shape[1]

    if D>2:
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=particle_types)
    else:
        ax.scatter(pos[:,0], pos[:,1], pos[:,1]*0, c=particle_types)

    cmap = plt.cm.plasma
    num_bond = np.size(np.unique(np.array(top.bonds)[:,2]))
    if num_bond == 0: 
        print("There are no bonds in top.bonds")
        exit()
    for bond in top.bonds:
        part1 = pos[bond[0],:]
        part2 = pos[bond[1],:]
        bondt = bond[2]
        if D>2:
            ax.plot3D([part1[0], part2[0]], [part1[1], part2[1]], [part1[2], part2[2]], color=cmap(bondt/num_bond))
        else:
            ax.plot3D([part1[0], part2[0]], [part1[1], part2[1]], [0, 0], color=cmap(bondt/num_bond))

    if block: plt.show(block=block)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
