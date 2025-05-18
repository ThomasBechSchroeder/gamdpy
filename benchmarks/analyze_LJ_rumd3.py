import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

# List of stored benchmarks to compare with
#benchmarks = ['h100', 'RTX_4090', 'RTX_4070_Laptop', 'RTX_3070_Laptop', 'Quadro_P2000_Mobile']
benchmarks = ['RTX_2060_Super_AT', 
              'RTX_2080_Ti_bead65_AT', 
              #'RTX_3070_Laptop_AT',
              'RTX_4070_AT', 
              'RTX_4090_AT',]
style = ['ro', 'bo', 'go', 'ko']

# Print benchmarks in markdown
print('## Preliminary benchmarks.')
print()
print('![Fig](./Data/benchmark_LJ_tps.png)')
for index, benchmark in enumerate(benchmarks):
    print('\n' + benchmark+':')
    with open('Data/benchmark_LJ_' + benchmark + '.pkl', 'rb') as file:
        data = pickle.load(file)
    print('|        N  |   TPS   |  MATS |  pb | tp | skin | gridsync |  nblist      |  NIII  |')
    print('| --------: | ------: | ----: | --: | --:| ---: | :------: | :----------: | :----: |')
    for n, tps, cp in zip(data['N'], data['TPS_AT'], data['compute_plans_at']):
        print(f"|{n:10} |{tps:8.1f} |{tps*n/1e6:6.1f} |{cp['pb']:4} |{cp['tp']:3} |{cp['skin']:5.2f} |   {cp['gridsync']!s:6} | {cp['nblist']:12} | {cp['UtilizeNIII']!s:5}  |")

plt.figure()
plt.title('LJ benchmark, NVE, rho=0.8442')
for index, benchmark in enumerate(benchmarks):
    bdf = pd.read_csv('Data/benchmark_LJ_' + benchmark + '.csv')
    plt.loglog(bdf['N'], bdf['TPS_AT'],  style[index]+'-', label=benchmark)
N = np.array((512, 2e6))
plt.loglog(N, 500 * 1e6 / N, '-.', label='MATS=500')
plt.legend()
plt.xlim((400, 1.5e6))
plt.ylim((50, 1.e6))
plt.xlabel('N')
plt.ylabel('TPS')
plt.savefig('Data/benchmark_LJ_tps.pdf')
plt.savefig('Data/benchmark_LJ_tps.png')
plt.show(block=False)
 


fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
axs[0].set_ylabel('TPS')
axs[1].set_ylabel('TPS')
axs[2].set_ylabel('TPS')
axs[2].set_xlabel('N')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)
axs[0].set_title('LJ benchmark, NVE, rho=0.8442, single prec.')
colors = ['red', 'blue', 'green']

bdf = pd.read_csv('Data/benchmark_LJ_RTX_2080_Ti_bead65_AT.csv')
axs[0].loglog(bdf['N'], bdf['TPS_AT'], 'o-', color=colors[0], label='gamdpy')
bdf = pd.read_csv('Data/benchmark_LJ_RTX_2080_Ti_i43_AT.csv')
axs[0].loglog(bdf['N'], bdf['TPS_AT'], 'o--', color=colors[0], alpha=0.6)
rumd36 = np.loadtxt('Data/Rumd36_LJ_RTX_2080_Ti.dat')
axs[0].loglog(rumd36[:,0], rumd36[:, 1], 'o-', color=colors[1], label='Rumd3.6')
lammps = np.loadtxt('Data/Lammps_Rtx2080Ti_gpu_cuda_single_mpi.dat')
axs[0].loglog(lammps[:,0],lammps[:, 1], 'o--', color=colors[2], alpha=0.6)
lammps = np.loadtxt('Data/Lammps_Rtx2080Ti_bead65_gpu_cuda_single_mpi.dat')
axs[0].loglog(lammps[:,0],lammps[:, 1], 'o-', color=colors[2], label='Lammps')
N = np.array((512, 2e6))
PLOTMATS = 2*rumd36[-1,0]*rumd36[-1, 1]/1e6
axs[0].loglog(N, PLOTMATS * 1e6 / N, 'k-.', label=f'MATS={int(PLOTMATS)}')
axs[0].legend()
axs[0].set_xlim((400, 1.5e6))
axs[0].set_ylim((50, 3e5))
txt =  'Rtx 2080 Ti: 11.75 TFlops, 616 GB/s\n'
txt += 'bead65 (lines): TR 2920X 12-Core 3.5 GHz\n'
txt += 'i43 (dashed lines): FX-8370 8-Core 4.0 GHz '
axs[0].text(1e3, 1e2, txt, fontsize=11)


bdf = pd.read_csv('Data/benchmark_LJ_RTX_4070_AT.csv')
axs[1].loglog(bdf['N'], bdf['TPS_AT'], 'o-', color=colors[0], label='gamdpy')
rumd36 = np.loadtxt('Data/Rumd36_LJ_RTX_4070.dat')
axs[1].loglog(rumd36[:,0], rumd36[:, 1], 'o-', color=colors[1], label='Rumd3.6')
lammps = np.loadtxt('Data/Lammps_Rtx4070_gpu_cuda_single_mpi.dat')
axs[1].loglog(lammps[:,0],lammps[:, 1], 'o-', color=colors[2], label='Lammps')
#lammpsV100 = np.loadtxt('Data/MATS_Lammps_LJ_V100.dat')
#axs[1].loglog(lammpsV100[:,0],lammpsV100[:, 1]/lammpsV100[:,0]*1e6, 'o--', color=colors[2], alpha=0.6, label='Lammps V100')
N = np.array((512, 2e6))
PLOTMATS = 2*rumd36[-1,0]*rumd36[-1, 1]/1e6
axs[1].loglog(N, PLOTMATS * 1e6 / N, 'k-.', label=f'MATS={int(PLOTMATS)}')
axs[1].legend()
axs[1].set_xlim((400, 1.5e6))
axs[1].set_ylim((50, 3e5))
txt =  'Rtx 4070: 22.61 TFlops, 504 GB/s\n'
txt += 'desktop: i7-14700 8-core 5.3 GHz'
#txt += 'V100 (double prec!) '
axs[1].text(1e3, 1e2, txt, fontsize=11)
for i in range(6, lammps.shape[0]):
    axs[1].text(lammps[i, 0]/1.2, lammps[i, 1]/1.5, int(lammps[i, 2]), fontsize=10)


bdf = pd.read_csv('Data/benchmark_LJ_RTX_4090_AT.csv')
axs[2].loglog(bdf['N'], bdf['TPS'], 'o-', color=colors[0], label='gamdpy')
rumd36 = np.loadtxt('Data/Rumd36_LJ_RTX_4090.dat')
axs[2].loglog(rumd36[:,0], rumd36[:, 1], 'o-', color=colors[1], label='Rumd3.6')
lammps = np.loadtxt('Data/Lammps_Rtx4090_gpu_cuda_single_mpi.dat')
axs[2].loglog(lammps[:,0],lammps[:, 1], 'o-', color=colors[2], label='Lammps')
N = np.array((512, 2e6))
PLOTMATS = 2*rumd36[-1,0]*rumd36[-1, 1]/1e6
axs[2].loglog(N, PLOTMATS * 1e6 / N, 'k-.', label=f'MATS={int(PLOTMATS)}')
axs[2].legend()
axs[2].set_xlim((400, 1.5e6))
axs[2].set_ylim((50, 3e5))
txt =  'Rtx 4090: 73.07 TFlops, 1008 GB/s\n'
txt += 'bead67: Ryzen 9 7900 12-Core 3.7 GHz'
axs[2].text(1e3, 1e2, txt, fontsize=11)
for i in range(6, lammps.shape[0]):
    axs[2].text(lammps[i, 0]/1.2, lammps[i, 1]/1.5, int(lammps[i, 2]), fontsize=10)


plt.savefig('Data/benchmark_LJ_compare_rumd3_tps.pdf')
plt.savefig('Data/benchmark_LJ_compare_rumd3_tps.png')
plt.show(block=False)