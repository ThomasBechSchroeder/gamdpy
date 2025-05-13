"""
Application of the NVU Ray Tracing integrator.
Comparision with NVE inspired by article:
`NVU dynamics. II. Comparing to four other dynamics.`
"""
import math
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import sys
import gamdpy as gp


RHO = 1.2
TEMPERATURE = 0.7
SEED = None

DO_NVE_EQ = True
NVE_EQ_STEPS = 2**20
NVE_EQ_STEPS_PER_TIMEBLOCK = 2**15
NVE_EQ_OUTPUT = "examples/Data/NVU_RT_NVE_EQ_OUTPUT.h5"
NVE_EQ_CONF_OUTPUT = "examples/Data/NVU_RT_NVE_EQ_CONF_OUTPUT.npz"
NVE_DT = .005

DO_NVE_PROD = True
NVE_PROD_STEPS = NVE_EQ_STEPS
NVE_PROD_STEPS_PER_TIMEBLOCK = NVE_EQ_STEPS_PER_TIMEBLOCK
NVE_PROD_OUTPUT = "examples/Data/NVU_RT_NVE_PROD_OUTPUT.h5"
NVE_PROD_CONF_OUTPUT = "examples/Data/NVU_RT_NVE_PROD_CONF_OUTPUT.npz"

DO_NVU_PROD = True
NVU_PROD_STEPS = NVE_PROD_STEPS + NVE_EQ_STEPS
NVU_PROD_STEPS_PER_TIMEBLOCK = NVE_PROD_STEPS_PER_TIMEBLOCK
NVU_EQ_BLOCKS = NVU_PROD_STEPS//NVU_PROD_STEPS_PER_TIMEBLOCK//2
NVU_SCALAR_OUTPUT = NVU_PROD_STEPS//2**12
NVU_PROD_OUTPUT = "examples/Data/NVU_RT_NVU_PROD_OUTPUT.h5"


def run_simulations():
    # Setup configuration: FCC crystal
    #configuration['r'][27,2] += 0.01 # Perturb z-coordinate of particle 27

    # Setup pair potential: Binary Kob-Andersen LJ mixture.
    pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig = [[1.00, 0.80],
           [0.80, 0.88]]
    eps = [[1.00, 1.50],
           [1.50, 0.50]]
    # sig, eps = 1, 1
    cut = np.array(sig)*2.5
    pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    nve_integrator = gp.integrators.NVE(dt=NVE_DT)
    if DO_NVE_EQ:
        conf = gp.Configuration(D=3, compute_flags={'Fsq':True})
        conf.make_lattice(gp.unit_cells.FCC, cells=[4, 4, 4], rho=RHO)
        conf['m'] = 1.0
        conf.randomize_velocities(temperature=TEMPERATURE)
        conf.ptype[::5] = 1     # Every fifth particle set to type 1 (4:1 mixture)
        print(f"========== NVE EQ ({NVE_EQ_STEPS//NVE_EQ_STEPS_PER_TIMEBLOCK} blocks) ==========")

        sim = gp.Simulation(
            conf, pair_pot, nve_integrator,
            num_timeblocks=NVE_EQ_STEPS//NVE_EQ_STEPS_PER_TIMEBLOCK, 
            steps_per_timeblock=NVE_EQ_STEPS_PER_TIMEBLOCK,
            storage=NVE_EQ_OUTPUT)

        for block in sim.run_timeblocks():
            if block % 5 == 0:
                print(f'{block=:4}  {sim.status(per_particle=True)}')
        print(sim.summary())
        save_conf_to_npz(NVE_EQ_CONF_OUTPUT, conf, 0)

    if DO_NVE_PROD:
        print(f"========== NVE PROD ({NVE_PROD_STEPS//NVE_PROD_STEPS_PER_TIMEBLOCK} blocks) ==========")
        conf, _ = load_conf_from_npz(NVE_EQ_CONF_OUTPUT)

        sim = gp.Simulation(
            conf, pair_pot, nve_integrator,
            num_timeblocks=NVE_PROD_STEPS//NVE_PROD_STEPS_PER_TIMEBLOCK, 
            steps_per_timeblock=NVE_PROD_STEPS_PER_TIMEBLOCK,
            storage=NVE_PROD_OUTPUT)

        for block in sim.run_timeblocks():
            if block % 5 == 0:
                print(f'{block=:4}  {sim.status(per_particle=True)}')
        print(sim.summary())
        
        u, = gp.extract_scalars(sim.output, ['U'], first_block=0, D=conf.D)
        target_u = np.mean(u[len(u)*3//4:])
        for block in sim.run_timeblocks():
            ev = gp.Evaluator(conf, pair_pot)
            ev.evaluate()
            conf_u = np.sum(conf['U'])
            if conf_u <= target_u:
                save_conf_to_npz(NVE_PROD_CONF_OUTPUT, conf, target_u)
                break
        else:
            save_conf_to_npz(NVE_PROD_CONF_OUTPUT, conf, target_u)
            print("ERROR: Could not find a suitable configuration for NVU", file=sys.stderr)

    conf, target_u = load_conf_from_npz(NVE_PROD_CONF_OUTPUT)

    nvu_integrator = gp.integrators.NVU_RT(
        target_u=target_u,
        max_abs_val=2,
        threshold=1e-5,
        eps=5e-6,
        max_steps=10,
        max_initial_step_corrections=20,
        initial_step=0.5/RHO**(1/3),
        initial_step_if_high=0.01/RHO**(1/3),
        step=1,
        raytracing_method="parabola-newton",
    )

    if DO_NVU_PROD:
        print(f"========== NVU PROD ({NVU_PROD_STEPS//NVU_PROD_STEPS_PER_TIMEBLOCK} blocks) ==========")
        sim = gp.Simulation(
            conf, pair_pot, nvu_integrator,
            num_timeblocks=NVU_PROD_STEPS//NVU_PROD_STEPS_PER_TIMEBLOCK, 
            steps_per_timeblock=NVU_PROD_STEPS_PER_TIMEBLOCK,
            compute_flags={'Fsq':True},
            storage=NVU_PROD_OUTPUT,
            scalar_output=NVU_SCALAR_OUTPUT,
        )

        for block in sim.run_timeblocks():
            if block % 5 == 0:
                print(f'{block=:4}  {sim.status(per_particle=True)}')
        print(sim.summary())


def save_conf_to_npz(path: str, conf: gp.Configuration, target_u: float) -> None:
    np.savez(path, r=conf["r"], v=conf["v"], ptype=conf.ptype, 
             target_u=target_u, simbox_initial=conf.simbox.lengths, 
             m=conf["m"])
    

def load_conf_from_npz(path: str) -> Tuple[gp.Configuration, float]:
    conf_data = np.load(path)
    n, d = conf_data["r"].shape
    conf = gp.Configuration(N=n, D=d, compute_flags={'Fsq':True})
    conf["r"] = conf_data["r"]
    conf["m"] = conf_data.get("m", 1)
    conf["v"] = conf_data["v"]
    conf.ptype = conf_data["ptype"]
    conf.simbox = gp.Orthorhombic(D=d, lengths=conf_data["simbox_initial"])
    return conf, float(conf_data["target_u"])

def get_rdf(conf, positions, first_block, conf_per_block):
    _, nconf, _n, _d = positions.shape
    cal_rdf = gp.CalculatorRadialDistribution(conf, bins=500)
    for i in range(positions.shape[0]):
        if i < first_block:
            continue
        for j in range(conf_per_block):
            k = math.floor(j * nconf / conf_per_block)
            pos = positions[i, k, :, :]
            conf["r"] = pos
            conf.copy_to_device()
            cal_rdf.update()

    rdf = cal_rdf.read()
    return rdf

def plot_output():
    nve_prod_output = gp.tools.TrajectoryIO(NVE_PROD_OUTPUT).get_h5()
    conf, target_u = load_conf_from_npz(NVE_PROD_CONF_OUTPUT)
    nvu_prod_output = gp.tools.TrajectoryIO(NVU_PROD_OUTPUT).get_h5()

    conf_per_block = 1024 // (NVE_PROD_STEPS // NVE_PROD_STEPS_PER_TIMEBLOCK)
    nve_rdf = get_rdf(conf, nve_prod_output["block"][:, :, 0, :, :], 0, conf_per_block)
    nvu_rdf = get_rdf(conf, nvu_prod_output["block"][:, :, 0, :, :], NVU_EQ_BLOCKS, conf_per_block)

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(rf"$g(r)$")
    ax = fig.add_subplot()
    ax.plot(nve_rdf['distances'], np.mean(nve_rdf['rdf'], axis=0), linewidth=1, color="black", label="NVE")
    ax.plot(nvu_rdf['distances'], np.mean(nvu_rdf['rdf'], axis=0), "o", linewidth=0, markersize=5, 
            markeredgecolor="black", markeredgewidth=1, alpha=.5,
            label=f"NVU")
    for i in range(nve_rdf["rdf_ptype"].shape[1]):
        for j in range(nve_rdf["rdf_ptype"].shape[1]):
            if i > j:
                continue
            e_rdf = np.mean(nve_rdf['rdf_ptype'][:, i, j, :], axis=0)
            u_rdf = np.mean(nvu_rdf['rdf_ptype'][:, i, j, :], axis=0)
            ax.plot(nve_rdf['distances'], e_rdf, linewidth=1, color="black")
            ax.plot(nvu_rdf['distances'], u_rdf, "o", linewidth=0, markersize=5, 
                    markeredgecolor="black", markeredgewidth=1, alpha=.5,
                    label=f"NVU  {['A', 'B'][i]}-{['A', 'B'][j]}")
    ax.legend()
    ax.grid()
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")
    fig.savefig("rdf.svg")

    nve_msd = gp.tools.calc_dynamics(nve_prod_output, first_block=0)["msd"]
    nvu_msd = gp.tools.calc_dynamics(nvu_prod_output, first_block=NVU_EQ_BLOCKS)["msd"]

    n_msd, n_ptype = nvu_msd.shape
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    fig.suptitle("MSD")
    for i in range(n_ptype):
        other_time = nve_prod_output.attrs['dt'] * 2 ** np.arange(n_msd)
        ln, = ax.loglog(other_time, nve_msd[:, i], linewidth=1, color="black", alpha=.8)
        if i == 0:
            ln.set_label("NVE")
        kb = 1
        mass = 1
        prod_beta = np.sqrt(mass * nvu_msd[0, i] / (3 * kb * TEMPERATURE))
        beta_nvu_time = prod_beta * 2 ** np.arange(n_msd)
        ax.loglog(beta_nvu_time, nvu_msd[:, i], linewidth=0, marker='o', 
                  markeredgecolor="black", markersize=5, markeredgewidth=1, alpha=.8, label=f"NVU {['A', 'B'][i]}")
    ax.legend()
    ax.grid()
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$MSD$")
    fig.savefig("msd.svg")


if __name__ == "__main__":
    run_simulations()
    plot_output()
    # plt.show()
