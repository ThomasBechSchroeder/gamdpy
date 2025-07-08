""" Example of simulation for a LJ system using NVU integrator """

import gamdpy as gp
import pandas as pd
import numpy as np
from numba import config
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) # Removes "RuntimeWarning: More than 20 figures have been opened."
config.CUDA_LOW_OCCUPANCY_WARNINGS = False

dl = 0.03
temperature = 0.800



# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
configuration.make_positions(N=5*5*5*4, rho=1.2)
configuration['m'] = 1.0 # Specify all masses to unity
configuration.randomize_velocities(temperature=2.0) # Initial high temperature for randomizing
configuration.ptype[::5] = 1 # Every fifth particle set to type 1 (4:1 mixture)


# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)

sig = [[1.00, 0.80],
        [0.80, 0.88]]
eps = [[1.00, 1.50],
        [1.50, 0.50]]
cut = np.array(sig)*2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)


Ttarget_function = gp.make_function_ramp(value0=2.000,       x0=1024*64*0.004*(1/8),
                                          value1=temperature, x1=1024*64*0.004*(1/4))
print()
print("Step 1/3: Running a NVT simulation with a temperature ramp from T=2.0",
      "to 0.8")
for i in range(2): print()
# Setup of NVT integrator and simulation, in order to find the average value
# of the potential energy to be used by the NVU integrator.
NVT_integrator = gp.integrators.NVT(temperature = Ttarget_function, tau = 0.2, dt = 0.004)
#runtime_actions = [gp.MomentumReset(100), ]
runtime_actions = [gp.MomentumReset(100),
                   #gp.ScalarSaver(2),
                   gp.TrajectorySaver() ,]
NVT_sim = gp.Simulation(configuration, pair_pot, NVT_integrator, runtime_actions,
                    num_timeblocks=64, steps_per_timeblock=1024,
                    storage="memory")


#Running the NVT simulation
NVT_sim.run()

for i in range(2): print()
print("Step 2/3: Continuing the NVT simulation with T = 0.8 in order to find the",
      "average value of the potential energy to be used by the NVU integrator")
for i in range(2): print()

runtime_actions = [gp.MomentumReset(100),
                   gp.TrajectorySaver(),
                    gp.ScalarSaver(2, {'Fsq':True, 'lapU':True}), ]


NVT_integrator = gp.integrators.NVT(temperature = temperature, tau = 0.2, dt = 0.004)
NVT_sim = gp.Simulation(configuration, pair_pot, NVT_integrator, runtime_actions,
                        num_timeblocks = 64, steps_per_timeblock = 1024,
                        storage = 'memory')


#Running the NVT simulation
NVT_sim.run()


#
#Finding the average potential energy (= U_0) of the run.
columns = ['U']
data = np.array(gp.extract_scalars(NVT_sim.output, columns, first_block=8))
df = pd.DataFrame(data.T, columns=columns)
U_0 = np.mean(df['U'])/configuration.N


for i in range(2): print()

print("Step 3/3: Running the NVU simulation using the final configuration",
      "from the previous NVT simulation and the average PE as the", 
      f"constant-potential energy: U_0 = {np.round(U_0,3)} (pr particle)")
for i in range(2): print()
#Setting up the NVU integrator and simulation. Note, that dt = dl.
NVU_integrator = gp.integrators.NVU(U_0 = U_0, dl = dl)

runtime_actions = [gp.MomentumReset(100),
                    gp.TrajectorySaver(),
                    gp.ScalarSaver(4*128, {'Fsq':True, 'lapU':True}), ]

NVU_sim = gp.Simulation(configuration, pair_pot, NVU_integrator, runtime_actions,
                        num_timeblocks = 32, steps_per_timeblock = 8*1024, #num_timeblocks = 32
                        storage = 'memory')


#Running the NVU simulation
NVU_sim.run()



#Calculating dynamics
NVU_dynamics = gp.tools.calc_dynamics(NVU_sim.output, 16, qvalues=[7.5, 5.5])
NVT_dynamics = gp.tools.calc_dynamics(NVT_sim.output, 16, qvalues=[7.5, 5.5])

plt.loglog(0,0,'k',label = "NVT simulation")
plt.loglog(0,0,'k+',label = "NVU simulation")
plt.loglog(NVT_dynamics['times'],NVT_dynamics['msd'][:,0])
plt.loglog(NVT_dynamics['times'],NVT_dynamics['msd'][:,1])
plt.loglog(NVU_dynamics['times']*0.028,NVU_dynamics['msd'][:,0],"C0+",markersize=10)
plt.loglog(NVU_dynamics['times']*0.028,NVU_dynamics['msd'][:,1],"C1+",markersize=10)
plt.loglog((0,NVU_dynamics['times'][-1]),(0,NVU_dynamics['msd'][-1,0]),'k--',linewidth=.5, label = "Slope = 1")
plt.legend()
plt.title("Comparing NVT and NVU dynamics using the mean squared\ndisplacement for A and B particles in KABLJ system at T=0.8.")
plt.xlabel(r"$t\cdot some\ constant$")
plt.ylabel(r"$<(\Delta \mathbf{r})^2>$")
plt.xlim(0.001)
plt.ylim(0.1**5)
if __name__ == "__main__":
    plt.show(block=False)

#Calculating the configurational temperature
columns = ['U', 'lapU', 'Fsq', 'W']
data = np.array(gp.extract_scalars(NVU_sim.output, columns, first_block=16))
df = pd.DataFrame(data.T, columns=columns)
df['Tconf'] = df['Fsq']/df['lapU']
Tconf = np.mean(df['Tconf'],axis=0)

times = len(df['Tconf'])*128*4*dl


plt.figure(figsize=(10,4))
plt.plot(np.arange(len(df['Tconf']))*128*4*dl,np.round(df['Tconf'],2),label = r"$T_{conf}$")
plt.plot((0,times),(temperature,temperature), label = f"Set temperature (T = {temperature})")
plt.ylabel("Temperature")
plt.xlabel("t")
plt.legend()
if __name__ == "__main__":
    plt.show(block=False)

plt.figure(figsize=(10,4))
plt.plot(np.arange(len(df['U']))*128*4*dl,df['U']/configuration.N, label = "U(t)")
plt.plot((0,times),(U_0,U_0), label = f"U_0 = {np.round(U_0,3)}")
plt.ylabel("Potential energy")
plt.xlabel("t")
plt.legend()

if __name__ == "__main__":
    plt.show(block=True)
