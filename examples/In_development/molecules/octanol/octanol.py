''' TraPPE octanol
Force-field parameters for octanol is downloaded at, http://trappe.oit.umn.edu/

PDB file is from https://github.com/wesbarnett/OPLS-molecules/blob/master/pdb/alcohols/octanol.pdb
wget https://raw.githubusercontent.com/wesbarnett/OPLS-molecules/master/pdb/alcohols/octanol.pdb

It is assumed that atoms are listed in the same order as in the two files.
'''

from pprint import pprint
import numpy as np
import csv


def load_pdb(filename: str) -> dict:
    """ Load a PDB file and return a lists of Atom name, Residue name and coordinates
    Ignore hydrogen atoms attached to a carbon atom """
    atom_name = []
    coordinates = []
    delete_if_hydrogen = False
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM"):
                a_name: str = line[12:17].strip()
                if a_name.startswith("H") and delete_if_hydrogen:
                    continue
                if a_name.startswith("C"):
                    delete_if_hydrogen = True
                else:
                    delete_if_hydrogen = False
                atom_name.append(a_name)
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {
        "atom_name": atom_name,
        "coordinates": coordinates
    }


def plot_molecule(molecule: dict):
    import matplotlib.pyplot as plt

    xyx = np.array(molecule["coordinates"])
    plt.figure()
    plt.plot(xyx[:, 0], xyx[:, 1], '--o')
    for i, txt in enumerate(molecule["atom_name"]):
        plt.annotate(f'{i}: {txt}', (xyx[i, 0], xyx[i, 1]))
    plt.show()


def read_parameters(file_path: str) -> dict:
    # This dictionary will store all data with sub-dictionaries for each type of parameter
    data = {
        'atoms': [],
        'stretches': [],
        'bends': [],
        'torsions': []
    }

    # Define current section to manage where to put the data
    current_section = None
    header = None

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if not row:
                continue  # skip empty rows
            if row[0].startswith('#'):  # New section
                if 'atom' in row[1]:
                    current_section = 'atoms'
                elif 'stretch' in row[1]:
                    current_section = 'stretches'
                elif 'bend' in row[1]:
                    current_section = 'bends'
                elif 'torsion' in row[1]:
                    current_section = 'torsions'
                header = row
            else:
                # It's a data row
                if header and current_section:
                    # Create a dictionary for the row using the latest headers
                    item = {header[i]: row[i] for i in range(len(row))}
                    data[current_section].append(item)

    return data


def main():
    octanol = load_pdb("octanol.pdb")
    pprint(octanol)
    plot_molecule(octanol)
    ff_parameters = read_parameters("trappe_parameters_35.csv")
    pprint(ff_parameters)


if __name__ == "__main__":
    main()
