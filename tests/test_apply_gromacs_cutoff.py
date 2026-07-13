
def test_apply_gromacs_cutoff(plot=False):
    import numpy as np
    import gamdpy as gp

    pair_pot_bare = gp.LJ_12_6_sigma_epsilon
    pair_pot = gp.apply_gromacs_cutoff(pair_pot_bare)

    #        σ    ε    r₁   r𝒸
    params = 1.0, 1.0, 1.6, 2.0

    r = np.linspace(1.4, 2.2, 801)

    u, s, umm = [], [], []
    for this_r in r:
        this_u, this_s, this_umm = pair_pot(this_r, params)
        u.append(this_u)
        s.append(this_s)
        umm.append(this_umm)
    if plot:

        # Energy
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(r, u, label=r'$u_m=u_{LJ}+S$')
        plt.plot(r, 4*(r**-12-r**-6), label=r'$u_{LJ}=4(r^{-12}-r^{-6})$')
        plt.plot(r, 4 * (r ** -12 - r ** -6)-u, label='$u_{LJ}-u_m$')
        plt.plot([params[-2], params[-1]],[pair_pot(params[-2], params)[0],0], 'o', label='cutoffs', markersize=8)
        plt.xlabel(r'Pair distance, $r$')
        plt.ylabel('Pair Energy, $u$')
        plt.legend()
        plt.show()

        # Force multipliers, s=-u'/r
        plt.figure()
        plt.plot(r, s, label='From gamdpy implementation')
        plt.plot(r, -np.gradient(u, r) / r, '--', label='Numerical from u')
        plt.plot([params[-2], params[-1]], [pair_pot(params[-2], params)[1], 0], 'o', label='cutoffs', markersize=8)
        # plt.plot(r, 4 * (12*r ** -12 - 6*r ** -6), label='Bare Lennard-Jones')
        plt.xlabel(r'Pair distance, $r$')
        plt.ylabel('Force multiplier, $s$')
        plt.legend()
        plt.show()

        # Curvature
        plt.figure()
        plt.plot(r, umm, label='From gamdpy implementation')
        plt.plot(r[2:-2], np.gradient(np.gradient(u, r), r)[2:-2], '--', label='Numerical from u')
        plt.plot([params[-2], params[-1]], [pair_pot(params[-2], params)[2], 0], 'o', label='cutoffs', markersize=8)
        plt.plot(r, 4 * (12*13*r ** -14 - 6*7*r ** -8), ':', label='Bare Lennard-Jones')
        plt.xlabel(r'Pair distance, $r$')
        plt.ylabel(r'Curvature, $\frac{d^2u}{dr^2}$')
        plt.legend()
        plt.show()

    test_dists = 1.4, 1.6, 1.8, 1.99, 2.0
    reference_data = {
        1.4: (-0.35459317532331097, -1.1942834996714082, -5.768193433122846),
        1.6: (-0.11811397750352626, -0.4921801632917777, -3.0455808541773877),
        1.8: (-0.01744051389029623, -0.13735330361158876, -2.1937169086509116),
        #1.99: (-2.541031881131417e-06, -0.0003817580316710645, -0.15115156157313392), # Change needed after removing jit'ing from cut-off function
        1.99: (np.float32(-2.5480986e-06), np.float32(-0.00038173795), np.float32(-0.15115207)),
        2.0: (0.0, 0.0, 0.0)
    }

    for dr in test_dists:
        # print(f'{dr}: {pair_pot(dr, params)},') # Generate data (assume code is trusted, confirm visually by plots)
        pp = pair_pot(dr, params)
        assert np.allclose(pp, reference_data[dr]), f'Gromacs switching for dr={dr} is incorrect: {pp} <-> {reference_data[dr]}'

if __name__ == '__main__':  # pragma: no cover
    test_apply_gromacs_cutoff(plot=True)
