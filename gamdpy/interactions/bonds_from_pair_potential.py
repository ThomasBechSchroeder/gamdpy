import numba
from .bonds import Bonds

def Bonds_from_PairPotential(pair_pot, pair_list, full_type_list, strength_factor, ok_if_not_excluded=False):

    # 1. Get the pair_function from the pair_potential and scale the output by strength_factor
    pair_func = numba.njit(pair_pot.pairpotential_function)
    def modified_strength_pairfunc(dist, params):
        u, s, umm = pair_func(dist, params)
        return strength_factor * u, strength_factor*s, strength_factor*umm



    pair_type_bond_type = {}
    bonds_dihedral14_list = [] # the new bond-list consisting of dihedral 1-4 pairs
    bond14_params = [] # the Lennard-Jones parameters for a given pair type that appears in the 1-4 list

    # Get the parameters from the pair potential object. These will get passed to the pair function as is.
    pp_params, pp_max_cut = pair_pot.convert_user_params()

    exclusion_list = pair_pot.exclusions
    for atom0, atom1 in pair_list:
        if not ok_if_not_excluded:
            found_1_in_list_0 = atom1 in exclusion_list[atom0][:exclusion_list[atom0][-1]]
            found_0_in_list_1 = atom0 in exclusion_list[atom1][:exclusion_list[atom1][-1]]
            if not (found_1_in_list_0 and found_0_in_list_1):
                raise ValueError("Pair %d, %d not in exclusion list" % (atom0, atom1))

        type0, type1 = sorted((full_type_list[atom0], full_type_list[atom1])) # so type0 <= type 1
        if (type0, type1) not in pair_type_bond_type:
            pair_type_bond_type[(type0, type1)] = len(pair_type_bond_type) # so they get mapped to a bond type which 
            # is just the order in which the types first appear in the dihhedral list0
            
            # the data type giving the parameters for a given type-combination here gives problems in bonds.py (line 44)
            # when trying to convert to a simple array of float32.
            bond14_params.append(list(pp_params[type0, type1]))
        bond_type = pair_type_bond_type[(type0, type1)]
        bonds_dihedral14_list.append([atom0, atom1, bond_type])
    bonds = Bonds(modified_strength_pairfunc, bonds_dihedral14_list, bond14_params)
    return bonds
