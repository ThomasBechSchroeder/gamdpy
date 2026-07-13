""" Test that ZBL potential is giving same results as LAMMPS """
import sys

def test_zbl():
    import numpy as np
    import gamdpy as gp
    import pytest
    from zbl import conversion_factors_zbl

    cf = conversion_factors_zbl(Z=29, molar_mass=63.546)

    filename = './zbl.h5'
    output = gp.tools.TrajectoryIO(filename).get_h5()
    nblocks, nconfs, N, D = output['trajectory/positions'].shape
    U, W, K = gp.ScalarSaver.extract(output, ['U', 'W', 'K'], per_particle=False, first_block=1)

    # Energy in eV/particle
    u_lammps = 3.580  # eV/particle
    u_gamdpy = float(np.mean(U/N*cf['in_eV']))
    print(f"u_lammps={u_lammps:.3f}, u_gamdpy={u_gamdpy:.3f}")
    assert u_lammps == pytest.approx(u_gamdpy, rel=1e-2)

    # Pressure in Gpa
    simbox = output['initial_configuration'].attrs['simbox_data']
    volume = np.prod(simbox)
    rho = N / volume
    dof = D * N - D  # degrees of freedom
    T_kin = 2 * K / dof
    P = rho * T_kin + W / volume

    p_lammps = 71.03  # GPa
    p_gamdpy = float(np.mean(P*cf['in_GPa']))
    print(f"p_lammps={p_lammps:.3f}, p_gamdpy={p_gamdpy:.3f}")
    assert p_lammps == pytest.approx(p_gamdpy, rel=1e-2)

mode = "ci" # default
if "nightly" in sys.argv:
    mode = "nightly"
if "interactive" in sys.argv:
    mode = "interactive" # if both present then go with "interactive"
print("Mode=%s" % mode)

# Run simulation (produce zbl.h5 file)
from zbl import main as setup
if mode == "ci" or "setup" in sys.argv:
    setup()

# Run tests
test_zbl()

# Generate figures
if mode == "interactive":
    from generate_figures import main as generate_figures
    generate_figures()
