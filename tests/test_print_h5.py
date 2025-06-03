import gamdpy as gp

def test_print_h5():
    sim = gp.get_default_sim()
    for _ in sim.run_timeblocks():
        pass
    gp.tools.print_h5_structure(sim.output)
    gp.tools.print_h5_attributes(sim.output)

if __name__ == '__main__':  # pragma: no cover
    test_print_h5()
