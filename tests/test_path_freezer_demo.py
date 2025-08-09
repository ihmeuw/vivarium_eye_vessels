"""Test PathFreezer component for vessel formation demonstration."""
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


def test_path_freezer_vessel_formation():
    """
    Test PathFreezer component by running a longer simulation that demonstrates
    vessel formation through particle freezing. Shows how freeze_interval affects
    vessel continuity and growth patterns.
    """
    # Initialize pygame to check if display is available
    pygame.init()

    # Skip test if no display is available (e.g., on CI servers)
    try:
        pygame.display.set_mode((1, 1))
        pygame.display.quit()
    except pygame.error:
        pytest.skip("No display available for visual test")

    # Path to the PathFreezer demo model specification
    model_spec_path = (
        current_dir
        / "src"
        / "vivarium_eye_vessels"
        / "model_specifications"
        / "path_freezer_demo.yaml"
    )

    assert model_spec_path.exists(), f"Model spec not found at {model_spec_path}"

    # Create simulation context
    sim = InteractiveContext(str(model_spec_path))

    # Run simulation for about 10 seconds to see vessel formation
    start_time = time.time()
    target_duration = 10.0  # seconds
    step_count = 0

    try:
        while (time.time() - start_time) < target_duration:
            sim.step()
            step_count += 1
            # Small delay to allow visualization
            time.sleep(0.02)

    except pygame.error as e:
        pytest.fail(f"Visualization failed with error: {e}")
    except KeyboardInterrupt:
        # Allow manual termination during development
        pass
    finally:
        # Clean up pygame
        pygame.quit()

    # Verify simulation ran and created vessels
    population = sim.get_population()
    assert len(population) >= 200, f"Expected at least 200 particles, got {len(population)}"

    # Verify we have both active and frozen particles (vessel formation)
    active_particles = population[~population.frozen]
    frozen_particles = population[population.frozen]

    assert len(frozen_particles) > 0, "No vessel segments were created (no frozen particles)"
    assert len(active_particles) > 0, "No active growth points remaining"

    # Verify vessel tree structure is forming
    vessel_particles = population[population.path_id >= 0]
    assert len(vessel_particles) > 0, "No particles have valid path IDs"

    # Check that parent-child relationships exist
    has_parents = population[population.parent_id >= 0]
    assert len(has_parents) > 0, "No parent-child relationships formed"

    # Verify particles are contained within ellipsoid boundary
    positions = population[["x", "y", "z"]].values
    ellipsoid_distances = (
        (positions[:, 0] / 1.8) ** 2
        + (positions[:, 1] / 1.5) ** 2
        + (positions[:, 2] / 0.3) ** 2
    )
    max_ellipsoid_distance = ellipsoid_distances.max()
    assert (
        max_ellipsoid_distance <= 1.2
    ), f"Particles exceeded ellipsoid boundary: {max_ellipsoid_distance:.2f}"

    print(f"✓ PathFreezer simulation completed after {time.time() - start_time:.1f} seconds")
    print(f"✓ Total particles: {len(population)}")
    print(f"✓ Active particles (growth points): {len(active_particles)}")
    print(f"✓ Frozen particles (vessel segments): {len(frozen_particles)}")
    print(f"✓ Particles with path IDs: {len(vessel_particles)}")
    print(f"✓ Particles with parents: {len(has_parents)}")
    print(f"✓ Simulation steps completed: {step_count}")

    # Calculate vessel formation efficiency
    if len(population) > 0:
        vessel_coverage = len(frozen_particles) / len(population) * 100
        print(f"✓ Vessel formation coverage: {vessel_coverage:.1f}%")


if __name__ == "__main__":
    # Allow running the test directly for manual visualization
    test_path_freezer_vessel_formation()
