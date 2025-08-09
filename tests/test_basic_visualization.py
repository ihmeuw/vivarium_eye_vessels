"""Test basic particle visualization for 5 seconds to demonstrate the system."""
import sys
import time
from pathlib import Path

import pygame
import pytest
from vivarium import InteractiveContext

# Add the src directory to Python path for local development
current_dir = Path(__file__).parent.parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def test_basic_particle_visualization():
    """
    Test that runs a basic particle simulation with visualization for 5 seconds.
    
    This demonstrates 100 particles moving around in 3D space within an ellipsoid
    boundary, providing a visual test of the core particle system components.
    """
    # Initialize pygame to check if display is available
    pygame.init()
    
    # Skip test if no display is available (e.g., on CI servers)
    try:
        pygame.display.set_mode((1, 1))
        pygame.display.quit()
    except pygame.error:
        pytest.skip("No display available for visual test")
    
    # Path to the basic particle model specification
    model_spec_path = (
        current_dir 
        / "src" 
        / "vivarium_eye_vessels" 
        / "model_specifications" 
        / "basic_particles.yaml"
    )
    
    assert model_spec_path.exists(), f"Model spec not found at {model_spec_path}"
    
    # Create simulation context
    sim = InteractiveContext(str(model_spec_path))
    
    # Run simulation with visualization for approximately 5 seconds
    # With step_size=0.1 days and fps=60, we need about 50 time steps
    # to get roughly 5 seconds of real-time visualization
    start_time = time.time()
    target_duration = 5.0  # seconds
    
    try:
        while (time.time() - start_time) < target_duration:
            sim.step()
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
    except pygame.error as e:
        # If visualization fails, that's still useful information
        pytest.fail(f"Visualization failed with error: {e}")
    except KeyboardInterrupt:
        # Allow manual termination during development
        pass
    finally:
        # Clean up pygame
        pygame.quit()
    
    # Verify simulation ran and particles exist
    population = sim.get_population()
    assert len(population) == 100, f"Expected 100 particles, got {len(population)}"
    
    # Verify particles have expected columns
    expected_columns = ["x", "y", "z", "vx", "vy", "vz", "frozen"]
    for col in expected_columns:
        assert col in population.columns, f"Missing column: {col}"
    
    # Verify particles are within reasonable bounds (ellipsoid constraint)
    positions = population[["x", "y", "z"]].values
    max_distance = ((positions**2).sum(axis=1)**0.5).max()
    assert max_distance <= 3.0, f"Particles exceeded expected boundary: {max_distance}"
    
    print(f"✓ Visualization test completed successfully after {time.time() - start_time:.1f} seconds")
    print(f"✓ Final particle count: {len(population)}")
    print(f"✓ Particles contained within boundary (max distance: {max_distance:.2f})")


if __name__ == "__main__":
    # Allow running the test directly for manual visualization
    test_basic_particle_visualization()