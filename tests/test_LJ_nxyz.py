import sys
import numpy as np
import gamdpy as gp
from numba import cuda, config
import pandas as pd
import pytest

from hypothesis import given, strategies as st, settings, Verbosity, example

def LJ(nx, ny, nz, rho=0.8442, pb=None, tp=None, skin=None, gridsync=None, UtilizeNIII=None, cut=2.5, integrator='NVE', verbose=True):
    
    # Generate configuration with a FCC lattice
    configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[nx, ny, nz], rho=rho)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=1.44, seed=0)
    assert configuration.N==nx*ny*nz*4, f'Wrong number particles (FCC), {configuration.N} <> {nx*ny*nz*4}'
    assert configuration.D==3, f'Wrong dimension (FCC), {configuration.D} <> {3}'

    # Allow for overwritting of the default compute_plan
    compute_plan = gp.get_default_compute_plan(configuration)
    if pb!=None:
        compute_plan['pb'] = pb
    if tp!=None:
        compute_plan['tp'] = tp
    if skin!=None:
        compute_plan['skin'] = np.float32(skin)
    if gridsync!=None:
        compute_plan['gridsync'] = gridsync
    if UtilizeNIII!=None:
        compute_plan['UtilizeNb'] = UtilizeNIII
    if verbose:
        print('simbox lengths:', configuration.simbox.lengths)
        print('compute_plan: ', compute_plan)
   
    # Make the pair potential.
    pairfunc = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)  

    # Setup the integrator
    dt = 0.005

    if integrator=='NVE':
        integrator = gp.integrators.NVE(dt=dt)

    if integrator=='NVT':
        integrator = gp.integrators.NVT(temperature=0.70, tau=0.2, dt=dt)
        
    if integrator=='NVT_Langevin':
        integrator = gp.integrators.NVT_Langevin(temperature=0.70, alpha=0.1, dt=dt, seed=213)

    runtime_actions = [gp.MomentumReset(100), 
                   gp.ConfigurationSaver(), 
                   gp.ScalarSaver(8, {'Fsq':True, 'lapU':True, 'stresses':False}), ]


    # Setup the Simulation
    sim = gp.Simulation(configuration, pairpot, integrator, runtime_actions,
                        num_timeblocks=2, steps_per_timeblock=1024 * 4,
                        storage='memory', verbose=False)

    # Run simulation one block at a time
    for block in sim.run_timeblocks():
        pass 

    # Make conversion to dataframe a method at some point...
    columns = ['U', 'W', 'lapU', 'Fsq', 'K']
    data = np.array(gp.extract_scalars(sim.output, columns, first_block=1))
    df = pd.DataFrame(data.T, columns=columns)
    return df

def get_results_from_df(df, N, D):
    df['E'] = df['U'] + df['K'] # Total energy
    df['Tkin'] =2*df['K']/D/(N-1)
    df['Tconf'] = df['Fsq']/df['lapU']
    df['dU'] = df['U'] - np.mean(df['U'])
    df['dE'] = df['E'] - np.mean(df['E'])
    df['dW'] = df['W'] - np.mean(df['W'])

    df2 = df.drop(range(len(df)//2))

    df2['dU'] = df2['U'] - np.mean(df2['U'])
    df2['dE'] = df2['E'] - np.mean(df2['E'])
    df2['dW'] = df2['W'] - np.mean(df2['W'])

    var_e = np.var(df['E'])/N
    Tkin = np.mean(df2['Tkin'])
    Tconf = np.mean(df2['Tconf'])        
    R = np.dot(df2['dW'], df2['dU'])/(np.dot(df2['dW'], df2['dW'])*np.dot(df2['dU'], df2['dU']))**0.5
    Gamma = np.dot(df2['dW'], df2['dU'])/(np.dot(df2['dU'], df2['dU']))

    return var_e, Tkin, Tconf, R, Gamma

@pytest.mark.slow
@settings(deadline=200_000, max_examples = 8)
@given(nx=st.integers(min_value=4, max_value=16), ny=st.integers(min_value=4, max_value=16), nz=st.integers(min_value=4, max_value=16))
@example(nx=4,  ny=4,  nz=4)
@example(nx=16, ny=16, nz=32)
@example(nx=5, ny=5, nz=13)
def test_nve(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    df = LJ(nx, ny, nz, integrator='NVE', cut=2.5, verbose=False)
    var_e, Tkin, Tconf, R, Gamma = get_results_from_df(df, N, D)
    print(N, '\t', nx, '\t', ny, '\t', nz, '\t', var_e, '\t', Tkin, '\t',Tconf, '\t',R, '\t',Gamma)
    assert var_e < 1e-4
    assert 0.67 < Tkin  < 0.72, print(f'{Tkin=}')
    assert 0.67 < Tconf < 0.72, print(f'{Tkin=}')
    assert 0.86 <   R   < 0.99, print(f'{R=}')
    assert 5.1  < Gamma < 6.6,  print(f'{Gamma=}')
    
    return
@pytest.mark.slow
@settings(deadline=200_000, max_examples = 8)
@given(nx=st.integers(min_value=4, max_value=16), ny=st.integers(min_value=4, max_value=16), nz=st.integers(min_value=4, max_value=16))
@example(nx=4,  ny=4,  nz=4)
@example(nx=16, ny=16, nz=32)
@example(nx=10, ny=13, nz=15)
def test_nvt(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    df = LJ(nx, ny, nz, integrator='NVT', verbose=False)
    var_e, Tkin, Tconf, R, Gamma = get_results_from_df(df, N, D)
    print(N, '\t', nx, '\t', ny, '\t', nz, '\t', var_e, '\t', Tkin, '\t',Tconf, '\t',R, '\t',Gamma)
    # assert var_e < 0.001
    assert 0.67 < Tkin  < 0.73, print(f'{Tkin=}')
    assert 0.67 < Tconf < 0.73, print(f'{Tkin=}')
    assert 0.91 <   R   < 1.00, print(f'{R=}')
    assert 4.9  < Gamma < 7.2,  print(f'{Gamma=}')
    
    return


@pytest.mark.slow
@settings(deadline=200_000, max_examples = 8)
@given(nx=st.integers(min_value=4, max_value=16), ny=st.integers(min_value=4, max_value=16), nz=st.integers(min_value=4, max_value=16))
@example(nx=4,  ny=4,  nz=4)
@example(nx=16, ny=16, nz=32)
def test_nvt_langevin(nx, ny, nz):
    N = nx*ny*nz*4
    D = 3
    df = LJ(nx, ny, nz, integrator='NVT_Langevin', verbose=False)
    var_e, Tkin, Tconf, R, Gamma = get_results_from_df(df, N, D)
    print(N, '\t', nx, '\t', ny, '\t', nz, '\t', var_e, '\t', Tkin, '\t',Tconf, '\t',R, '\t',Gamma)
    # assert var_e < 0.001
    assert 0.62 < Tkin  < 0.78, print(f'{Tkin=}')
    assert 0.62 < Tconf < 0.78, print(f'{Tconf=}')
    assert 0.85 <   R   < 1.00, print(f'{R=}')
    assert 4.8  < Gamma < 6.8,  print(f'{Gamma=}')
    
    return
    
if __name__ == "__main__":
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    if len(sys.argv)==1 or 'NVE' in sys.argv:
        print('Testing LJ NVE:')
        test_nve()
        print('Passed: LJ NVE!')
    if len(sys.argv)==1 or 'NVT' in sys.argv:
        print('Testing LJ NVT:')
        test_nvt()
        print('Passed: LJ NVT!')
    if len(sys.argv)==1 or 'NVT_Langevin' in sys.argv:
        print('Testing LJ NVT Langevin:')
        test_nvt_langevin()
        print('Passed: LJ NVT Langevin!')

