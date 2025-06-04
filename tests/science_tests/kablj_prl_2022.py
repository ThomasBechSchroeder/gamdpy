""" Example of a binary LJ simulation using gamdpy.

NVT simulation of the Kob-Andersen mixture, and compare results with Rumd3 (rumd.org)
"""

import gamdpy as gp

import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

argv = sys.argv.copy()

# Specify statepoint
num_part = 10000
rho = 1.200
temperature = 0.45
filename = 'kablj_prl_2022'

# Read equilibrated configuration
configuration = gp.configuration_from_rumd3('ReferenceData/'+filename+'_end.xyz.gz')

# Setup pair potential: Binary Kob-Andersen LJ mixture.
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.80],
       [0.80, 0.88]]
eps = [[1.00, 1.50],
       [1.50, 0.50]]
cut = np.array(sig)*2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

if 'analyze_saved' not in argv:

    # Setup integrator. 
    dt = 0.004  # timestep
    num_timeblocks = 64           # Do simulation in this many 'blocks'. 
    #steps_per_timeblock = 16*8*1024  # ... each of this many steps'
    steps_per_timeblock = 1*1024  # ... each of this many steps'

    integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

    runtime_actions = [gp.MomentumReset(100),
                       gp.RestartSaver(),
                    gp.TrajectorySaver(),
                    gp.ScalarSaver() ]


    sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                        num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                        storage='Data/'+filename+'.h5')

    # Setup on-the-fly calculation of Radial Distribution Function
    calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)

    for block in sim.run_timeblocks():
        print(f'{sim.status(per_particle=True)}')
        calc_rdf.update()
    print(sim.summary())

    del sim

# The following could be done by using sim.output, but we use the saved h5 file instead:
with h5py.File('Data/'+filename+'.h5', 'r') as output:
    dynamics = gp.tools.calc_dynamics(output, 0, qvalues=[7.25, 5.5])
    U, W = gp.ScalarSaver.extract(output, columns=['U', 'W'], first_block=0)

    configuration = gp.configuration_from_hdf5_group(output, 'restarts/restart0000')
    configuration.copy_to_device()
    calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)
    for block in range(64):
        configuration2 = gp.configuration_from_hdf5_group(output, f'restarts/restart{block:04d}')
        configuration['r'] = configuration2['r'] # SHOULD NOT BE NECESARRY!!!
        configuration.copy_to_device()
        calc_rdf.update()

rdf = calc_rdf.read()
rdf['distances'] += rdf['distances'][1]/2 # NOTE: distances refers to bginning of bin!!!


#fig = plt.figure(figsize=(9, 14), layout='constrained')
fig = plt.figure(figsize=(8, 14))
axs = fig.subplot_mosaic([["msd", "msd"],
                          ["Fs", "Fs"],
                          ["rdf", "rdf"],
                          ["UW", "F1F2"]])

title = f' Science test ({filename}.py): SUCCESS\n'
title += 'Kob & Andersen Binary LJ, N=10000, rho=1.200, T=0.450 \n'
title += 'Reference: PRL v129 245501 (2022), doi.org/10.1103/PhysRevLett.129.245501'
axs['msd'].set_title(title)

axs['msd'].set_ylabel('log10( MSD )')
axs['msd'].set_xlabel('log10( Time )')
msd_ref = np.loadtxt('ReferenceData/'+filename+'_msd.dat')

lmsd_ref = np.log10(msd_ref[:25,1])
lmsd = np.log10(dynamics['msd'][:,0])
rmse = np.sqrt(np.mean( (lmsd-lmsd_ref[:len(lmsd)])**2 ))
axs['msd'].plot(np.log10(msd_ref[:25,0]), lmsd_ref, '--', label='A, ref')
axs['msd'].plot(np.log10(dynamics['times']), lmsd, 'o--', label=f'A, {rmse=:.1e}')

lmsd_ref = np.log10(msd_ref[:25,2])
lmsd = np.log10(dynamics['msd'][:,1])
rmse = np.sqrt(np.mean( (lmsd-lmsd_ref[:len(lmsd)])**2 ))
axs['msd'].plot(np.log10(msd_ref[:25,0]), lmsd_ref, '--', label='B, ref')
axs['msd'].plot(np.log10(dynamics['times']), lmsd, 'o--', label=f'B, {rmse=:.1e}')

axs['msd'].grid(linestyle='--', alpha=0.5)
axs['msd'].legend()


axs['Fs'].set_ylabel('Intermediate scattering function')
axs['Fs'].set_xlabel('log10( Time )')
Fs_ref = np.loadtxt('ReferenceData/'+filename+'_Fs.dat')

fs_ref = Fs_ref[:25,1]
fs = dynamics['Fs'][:,0]
rmse = np.sqrt(np.mean( (fs-fs_ref[:len(fs)])**2 ))
axs['Fs'].plot(np.log10(Fs_ref[:25,0]), fs_ref, '--', label='A, ref')
axs['Fs'].plot(np.log10(dynamics['times']), fs, 'o--', label=f'A, {rmse=:.1e}')

fs_ref = Fs_ref[:25,2]
fs = dynamics['Fs'][:,1]
rmse = np.sqrt(np.mean( (fs-fs_ref[:len(fs)])**2 ))
axs['Fs'].plot(np.log10(Fs_ref[:25,0]), fs_ref, '--', label='B, ref')
axs['Fs'].plot(np.log10(dynamics['times']), fs, 'o--', label=f'B, {rmse=:.1e}')

axs['Fs'].grid(linestyle='--', alpha=0.5)
axs['Fs'].legend()

axs['rdf'].set_ylabel('Radial Distribution Function')
axs['rdf'].set_xlabel('Distance')
rdf_ref = np.loadtxt('ReferenceData/'+filename+'_rdf.dat')
step = 4
axs['rdf'].plot(rdf_ref[::step,0], rdf_ref[::step,1], 'k.', label='AA, ref', alpha=0.75)
axs['rdf'].plot(rdf_ref[::step,0], rdf_ref[::step,2], 'b.', label='AB, ref', alpha=0.75)
axs['rdf'].plot(rdf_ref[::step,0], rdf_ref[::step,4], 'r.', label='BB, ref', alpha=0.75)
axs['rdf'].plot(rdf['distances'], np.mean(rdf['rdf_ptype'][:,0,0,:], axis=0), 'k', label='AA, rmse=xxx')
axs['rdf'].plot(rdf['distances'], np.mean(rdf['rdf_ptype'][:,0,1,:], axis=0), 'b', label='AB, rmse=xxx')
axs['rdf'].plot(rdf['distances'], np.mean(rdf['rdf_ptype'][:,1,1,:], axis=0), 'r', label='BB, rmse=xxx')
axs['rdf'].set_xlim([0.5, 3])
axs['rdf'].grid(linestyle='--', alpha=0.5)
axs['rdf'].legend()

axs['UW'].set_ylabel('W, Virial per particle')
axs['UW'].set_xlabel('U, Potential energy per particle')
step = max(int(len(U)/5000), 1) # Find step to plot at most 5000 points
NRhoTUWgammaRCvex_ref = np.loadtxt('ReferenceData/'+filename+'_NRhoTUWgammaRCvex.dat')

axs['UW'].plot(U[::step], W[::step], '.', alpha=0.5)
axs['UW'].plot(NRhoTUWgammaRCvex_ref[3], NRhoTUWgammaRCvex_ref[4], 'ko')
axs['UW'].text(0.05, 0.9, f'<U>={np.mean(U):.3f}, ({NRhoTUWgammaRCvex_ref[3]:.3f})', transform=axs['UW'].transAxes)
axs['UW'].text(0.05, 0.8, f'<W>={np.mean(W):.3f}, ({NRhoTUWgammaRCvex_ref[4]:.3f})', transform=axs['UW'].transAxes)
axs['UW'].text(0.05, 0.7, f'Cv,ex={np.var(U)*num_part/temperature**2:.1f}, ({NRhoTUWgammaRCvex_ref[7]:.1f})', transform=axs['UW'].transAxes)
cov = np.cov(W,U)
R =  cov[0,1]/(cov[0,0]*cov[1,1])**.5
gamma = cov[0,1]/cov[1,1]
axs['UW'].text(0.5, 0.2, f'R={R:.2f}, ({NRhoTUWgammaRCvex_ref[6]:.2f})', transform=axs['UW'].transAxes)
axs['UW'].text(0.5, 0.1, f'Slope={gamma:.2f}, ({NRhoTUWgammaRCvex_ref[5]:.2f})', transform=axs['UW'].transAxes)

F1 = configuration2['f']
configuration2.atomic_scale(density=1.44)
evaluator = gp.Evaluator(configuration2, pair_pot)
evaluator.evaluate(configuration2)
F2 = configuration2['f']
axs['F1F2'].set_ylabel('Fx(rho=1.44)')
axs['F1F2'].set_xlabel('Fx(rho=1.20)')
axs['F1F2'].plot(F1[:,0], F2[:,0], '.', alpha=0.5)
axs['F1F2'].plot(0, 0, 'ko')
cov = np.cov(F2[:,0], F1[:,0])
Rx =  cov[0,1]/(cov[0,0]*cov[1,1])**.5
cov = np.cov(F2[:,1], F1[:,1])
Ry =  cov[0,1]/(cov[0,0]*cov[1,1])**.5
cov = np.cov(F2[:,2], F1[:,2])
Rz =  cov[0,1]/(cov[0,0]*cov[1,1])**.5
axs['F1F2'].text(0.5, 0.3, f'Rx={Rx:.3f} (0.988)', transform=axs['F1F2'].transAxes)
axs['F1F2'].text(0.5, 0.2, f'Ry={Ry:.3f}', transform=axs['F1F2'].transAxes)
axs['F1F2'].text(0.5, 0.1, f'Rz={Rz:.3f}', transform=axs['F1F2'].transAxes)

fig.savefig('Data/'+filename+'.pdf')
print(f"Wrote: {'Data/'+filename+'.pdf'}")

fig.savefig('Data/'+filename+'.png')
print(f"Wrote: {'Data/'+filename+'.png'}")

if __name__ == "__main__":
    plt.show(block=True)

