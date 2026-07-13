""" Plot data of LAMMPS simulation """
from sympy.physics.units import energy


def read_lammps_log(filename):
    """
    Read thermo data from a LAMMPS log.lammps file.
    Returns: dict {column_name: [values, ...]}
    """

    with open(filename, "r") as f:
        lines = f.readlines()

    data = {}
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # Look for a thermo header line starting with "Step"
        if line.startswith("Step"):
            columns = line.split()
            for col in columns:
                data.setdefault(col, [])

            i += 1  # move to the first data line
            while i < n:
                line = lines[i].strip()
                if not line:
                    break

                parts = line.split()

                # stop if the first column is not a number
                try:
                    float(parts[0])
                except ValueError:
                    break

                # stop if a line doesn't match the number of columns
                if len(parts) != len(columns):
                    break

                for col, val in zip(columns, parts):
                    data[col].append(float(val))

                i += 1

            # re-check the current line (non-data) in the outer loop
            continue

        i += 1

    return data

def save_data(min_step=5000):
    data = read_lammps_log("log.lammps")
    to_disk = ''
    step = np.array(data['Step'])

    for key in data.keys():
        arr = np.array(data[key])
        to_disk += f'{key} = {np.mean(arr[step>min_step])}\n'

    print(f'Average values for step > {min_step}:')
    print(to_disk)
    print(to_disk, file=open('data.txt', 'w'))

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Plot pressure
    data = read_lammps_log("log.lammps")
    # print(data.keys())

    # Convert from bar to Gpa
    p_in_Gpa = np.array(data["Press"]) * 1e5/1e9

    steps = np.array(data["Step"])
    mask = steps > 5000
    time_step = 0.004  # picoseconds
    time = steps*time_step

    plt.figure(figsize=(6, 4))
    p_mean = np.mean(p_in_Gpa[mask])
    plt.title(f"LAMMPS simulation: <p> = {p_mean:.2f} GPa")
    plt.plot(time[mask], p_in_Gpa[mask])
    plt.axhline(y=p_mean, color='k', linestyle='--')
    plt.xlabel("Time, $t$ [ps]")
    plt.ylabel("Pressure. $p$ [GPa]")
    plt.savefig("pressure.png")
    plt.show()

    # Plot potential energy
    plt.figure(figsize=(6, 4))
    atoms_per_cell = 4
    num_cells = 10**3
    num_atoms = atoms_per_cell * num_cells
    energy_per_atom = np.array(data['PotEng'])/num_atoms
    u_mean = np.mean(energy_per_atom[mask])
    plt.title(f"LAMMPS simulation: <u> = {u_mean:.3f} eV/atom")
    plt.plot(time[mask], energy_per_atom[mask], label=r"$Y$")
    plt.axhline(y=u_mean, color='k', linestyle='--')
    plt.xlabel("Time, $t$ [ps]")
    plt.ylabel("potential energy per particle, $U/N$ [eV]")
    plt.savefig("potential_energy.png")
    plt.show()


    # Save data to disk
    save_data()