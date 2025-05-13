""" Consistency check for NPT integrators.

Simulation of a Lennard-Jones liquid in the NPT ensemble.
It's possible to switch between Langevin and Atomic NPT integrators.
This script verifies that several thermodynamics quantity are properly produced.

"""

import numpy as np

import gamdpy as gp

# Here you can decide to use "NPT_Atomic" or "NPT_Langevin"
flag = "Atomic"
my_T, my_rho, my_p = 2.0, 0.754289412611, 4.7     # Pressure should be P=4.7 for T=2.0 at this density

# Choose integrator
if flag=="Atomic":
    integrator = gp.integrators.NPT_Atomic  (temperature=my_T, tau=0.4, pressure=my_p, tau_p=20, dt=0.001)
elif flag=="Langevin":
    gp.integrators.NPT_Langevin(temperature=my_T, pressure=my_p, alpha=0.1, alpha_baro=0.0001, mass_baro=0.0001,
                                                             volume_velocity=0.0, barostatModeISO=True, boxFlucCoord=2, dt=0.001, seed=2023)

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3, compute_flags={'Vol':True})
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=my_rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=my_T, seed=0)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# NVT equilibration for calculation of c_V and Thermal pressure coefficient
integratorNVT = gp.integrators.NVT(temperature=my_T, tau=0.2, dt=0.001)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(32),
                   gp.MomentumReset(100)]

sim = gp.Simulation(configuration, pair_pot, integratorNVT, runtime_actions, 
                    num_timeblocks=8, steps_per_timeblock=16384,
                    storage='memory')
# Equilibration run
print("Running NVT simulation at (\\rho, T) = ({my_rho},{my_T})")
print("Equilibration")
sim.run()
# Data run
print("Data run")
sim.run()
U, W, K, Vol = gp.extract_scalars(sim.output, ['U', 'W', 'K', 'Vol'], first_block=1)
# Full c_V (not excess)
c_V = np.std(U+K)**2/my_T**2/configuration.N
dU = U - np.mean(U)
dW = W - np.mean(W)
# Thermal pressure coefficient https://en.wikipedia.org/wiki/Thermal_pressure
# from Eq. 89 in http://glass.ruc.dk/pdf/articles/2008_J_Chem_Phys_129_184508.pdf
P_th = my_rho + np.mean(dW*dU)/my_T**2/np.mean(Vol)
print(f"NVT simulation at T={my_T} and \\rho={my_rho}")
print(f"Mean values  of U, W and T_kin: {np.mean(U)/configuration.N} {np.mean(W)/configuration.N} {2*np.mean(K)/3/configuration.N}")
print(f"Standard dev of U, W and T_kin: {np.std(U)/configuration.N} {np.std(W)/configuration.N} {2*np.std(K)/3/configuration.N}")
print(f"Pressure (mean): {my_rho*(2*np.mean(K)/3+np.mean(W))/configuration.N}")
print(f"Thermal pressure coefficient: {P_th}")
print(f"Specific heat at constant volume: {c_V}")
print()

# NPT Simulation 
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=8, steps_per_timeblock=16384,
                    storage='memory')
sim.run()
sim.run()
U, W, K, Vol = gp.extract_scalars(sim.output, ['U', 'W', 'K', 'Vol'], first_block=1)
H = K + U + my_p * Vol      # enthalpy H = Etot + PV  
dH = H - np.mean(H)
dV = Vol - np.mean(Vol)
# Full c_P (not excess)
c_P = np.std(H)**2/my_T**2/configuration.N
print(f"NPT simulation at T={my_T} and p={my_p}")
print(f"Mean values  of U, W and T_kin: {np.mean(U)/configuration.N} {np.mean(W)/configuration.N} {2*np.mean(K)/3/configuration.N}")
print(f"Standard dev of U, W and T_kin: {np.std(U)/configuration.N} {np.std(W)/configuration.N} {2*np.std(K)/3/configuration.N}")
print(f"Mean and std of Volume: {np.mean(Vol)} {np.std(Vol)}")
print(f"Pressure (mean): {(2*np.mean(K)/3+np.mean(W))/np.mean(Vol)}") 
print(f"Density should be {my_rho} and is {configuration.N/np.mean(Vol)}")
K_T = my_T*np.mean(Vol)/np.std(Vol)**2
print(f"Isothermal bulk modulus K_T: {K_T}")
print()
print(f"Check 1: Thermal expansion coefficient")
beta_P = P_th/K_T           # from https://en.wikipedia.org/wiki/Thermal_pressure (it's named \alpha on wiki)
print(f"Thermal expansion coefficient from NVT fluctuations \\beta_P: {beta_P}")
print(f"Thermal expansion coefficient form NPT fluctuations \\beta_P: {np.mean(dH*dV)/my_T**2/np.mean(Vol)}") # 1/V dV/dT|_P Allen-Tildesley Eq. 2.87 pg 53
print()
print(f"Check 2: Consistency relations in Appendix B of http://dx.doi.org/10.1063/1.467468")
print(f"Equation B3 <P_int>   = P_ext + T<V^-1> : {np.mean((2*K/3+W)/Vol)} ?= {my_p + my_T*np.mean(1/Vol)}")
print(f"Equation B4 <P_int V> = P_exp <V>       : {np.mean( 2*K/3+W     )/configuration.N} ?= {my_p * np.mean(Vol)/configuration.N}")
print()
print(f"Check 3: Specific heat at constant pressure (this needs a longer run, use num_timeblocks=128)")
print(f"c_P from enthalpy fluctuations in the NPT ensamble : {c_P}")
# Verify that C_P = C_V + T V \beta_P**2 * K_T from problem 5.14 part c of Daniel V. Schroeder - An Introduction to Thermal Physics (1999)
print(f"c_P obtained from c_P = c_V + T V \\beta_P**2 * K_T: {c_V + my_T * np.mean(Vol)/configuration.N * beta_P**2 * K_T}" )
print()

