import pytest
import gamdpy

@pytest.fixture
def sim():
    """Fixture to initialize a default simulation object."""
    # steps_per_block is usually set during initialization in gamdpy
    return gamdpy.get_default_sim()

def test_compress_reaches_target_density(sim):
    """Verify the simulation reaches the desired density within a small tolerance."""
    initial_rho = sim.configuration.N / sim.configuration.get_volume()
    target_rho = initial_rho * 1.1  # 10% increase
    
    # Parameters for the compression
    steps_per_rescale = 10
    relative_change = 0.02
    
    sim.compress(
        desired_rho=target_rho, 
        steps_per_rescale=steps_per_rescale, 
        relative_change=relative_change
    )
    
    final_rho = sim.configuration.N / sim.configuration.get_volume()
    
    # Use pytest.approx for floating point comparisons
    assert final_rho == pytest.approx(target_rho, rel=1e-5)

def test_compress_invalid_steps_raises_error(sim):
    """Verify that a ValueError is raised if steps_per_rescale > steps_per_block."""
    too_many_steps = sim.steps_per_block + 1
    target_rho = (sim.configuration.N / sim.configuration.get_volume()) * 1.05
    
    with pytest.raises(ValueError, match="must not be larger than"):
        sim.compress(
            desired_rho=target_rho, 
            steps_per_rescale=too_many_steps, 
            relative_change=0.01
        )

def test_compress_exact_hit_on_last_step(sim):
    """Ensure the logic handles the 'else: done = True' branch to hit density exactly."""
    initial_rho = sim.configuration.N / sim.configuration.get_volume()
    # A very small change that should be completed in a single iteration
    target_rho = initial_rho * 1.001 
    
    sim.compress(
        desired_rho=target_rho, 
        steps_per_rescale=1, 
        relative_change=0.05
    )
    
    final_rho = sim.configuration.N / sim.configuration.get_volume()
    assert final_rho == pytest.approx(target_rho)

    