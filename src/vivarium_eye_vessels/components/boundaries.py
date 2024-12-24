from typing import Any, Dict, List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class EllipsoidContainment(Component):
    """Component that keeps particles within an ellipsoid boundary using Hooke's law.

    This version uses vectorized operations for improved performance and stores forces.
    """

    CONFIGURATION_DEFAULTS = {
        "ellipsoid_containment": {
            "a": 1.0,  # Semi-major axis in x direction
            "b": 1.0,  # Semi-major axis in y direction
            "c": 1.0,  # Semi-major axis in z direction
            "spring_constant": 0.1,  # Spring constant for Hooke's law (force/distance)
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "frozen"]

    def setup(self, builder: Builder) -> None:
        """Setup the component."""
        self.config = builder.configuration.ellipsoid_containment

        # Get semi-major axes from config
        self.a = float(self.config.a)
        self.b = float(self.config.b)
        self.c = float(self.config.c)
        self.spring_constant = float(self.config.spring_constant)

        # Pre-compute squared denominators for gradient calculation
        self.a2 = self.a * self.a
        self.b2 = self.b * self.b
        self.c2 = self.c * self.c

    def calculate_forces_vectorized(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized calculation of forces for all particles.

        Parameters
        ----------
        positions : np.ndarray
            Nx3 array of particle positions (x,y,z)

        Returns
        -------
        np.ndarray
            Nx3 array of force vectors
        """
        # Calculate normalized radial distances
        normalized_pos = positions * np.array([1 / self.a, 1 / self.b, 1 / self.c])
        d = np.sqrt(np.sum(normalized_pos * normalized_pos, axis=1))

        # Handle particles at origin
        mask_origin = d < 1e-10
        d[mask_origin] = 1e-10  # Avoid division by zero

        # Calculate surface distances (positive means outside)
        surface_distances = d - 1.0

        # Calculate direction vectors (proportional to gradient)
        directions = -positions * np.array([1 / self.a2, 1 / self.b2, 1 / self.c2])
        directions = directions / d[:, np.newaxis]  # Normalize by radial distance

        # Normalize direction vectors
        direction_magnitudes = np.linalg.norm(directions, axis=1)
        mask_nonzero = direction_magnitudes > 0
        directions[mask_nonzero] = directions[mask_nonzero] / direction_magnitudes[mask_nonzero, np.newaxis]

        # Only apply forces to particles outside ellipsoid
        mask_outside = surface_distances > 0

        # Calculate force magnitudes using Hooke's law
        force_magnitudes = np.zeros_like(surface_distances)
        force_magnitudes[mask_outside] = self.spring_constant * surface_distances[mask_outside]

        # Calculate final force vectors
        forces = directions * force_magnitudes[:, np.newaxis]

        # Zero out forces for particles at origin
        forces[mask_origin] = 0

        return forces

    def on_time_step(self, event: Event) -> None:
        """Apply and store restoring forces on each time step using vectorized operations."""
        # Get current state of all unfrozen particles
        pop = self.population_view.get(event.index, query="frozen == False")
        if pop.empty:
            return

        # Extract positions as numpy array
        positions = pop[["x", "y", "z"]].to_numpy()

        # Calculate forces for all particles at once
        forces = self.calculate_forces_vectorized(positions)

        # Store forces in population
        pop.loc[:, ["fx", "fy", "fz"]] += forces

        # Update velocities based on forces
        dt = event.step_size / pd.Timedelta(days=1)
        pop[["vx", "vy", "vz"]] += forces * dt

        # Update population
        self.population_view.update(pop)


class CylinderExclusion(Component):
    """Component that repels particles from inside a cylindrical exclusion zone using Hooke's law.
    Only considers radial distance from cylinder axis. This version uses vectorized operations and stores forces.
    """

    CONFIGURATION_DEFAULTS = {
        "cylinder_exclusion": {
            "radius": 1.0,  # Radius of the cylinder
            "center": [0.0, 0.0, 0.0],  # Center of the cylinder
            "direction": [0.0, 0.0, 1.0],  # Direction vector of cylinder (default along z-axis)
            "spring_constant": 0.1,  # Spring constant for Hooke's law (force/distance)
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "frozen"]

    def setup(self, builder: Builder) -> None:
        """Setup the component."""
        self.config = builder.configuration.cylinder_exclusion

        # Get cylinder parameters from config
        self.radius = float(self.config.radius)
        self.center = np.array(self.config.center, dtype=float)
        self.direction = np.array(self.config.direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)  # Ensure direction is a unit vector
        self.spring_constant = float(self.config.spring_constant)

        # Pre-compute random perpendicular vector for axis cases
        random_perpendicular = np.array([1, 0, 0]) if abs(self.direction[0]) < 0.9 else np.array([0, 1, 0])
        self.default_outward = np.cross(self.direction, random_perpendicular)
        self.default_outward /= np.linalg.norm(self.default_outward)

    def calculate_forces_vectorized(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized calculation of forces for all particles.

        Parameters
        ----------
        positions : np.ndarray
            Nx3 array of particle positions (x,y,z)

        Returns
        -------
        np.ndarray
            Nx3 array of force vectors
        """
        # Calculate vectors from center to each point
        rel_positions = positions - self.center

        # Calculate axial components for all points at once
        axial_dots = np.dot(rel_positions, self.direction)
        axial_components = axial_dots[:, np.newaxis] * self.direction

        # Calculate radial vectors
        radial_vectors = rel_positions - axial_components
        radial_distances = np.linalg.norm(radial_vectors, axis=1)

        # Calculate penetration depths (positive inside cylinder)
        penetrations = self.radius - radial_distances

        # Handle points on or very close to axis
        mask_on_axis = radial_distances < 1e-10
        outward_directions = np.zeros_like(positions)

        # For points not on axis, calculate actual outward direction
        mask_off_axis = ~mask_on_axis
        outward_directions[mask_off_axis] = radial_vectors[mask_off_axis] / radial_distances[mask_off_axis, np.newaxis]

        # For points on axis, use pre-computed default outward direction
        outward_directions[mask_on_axis] = self.default_outward

        # Only apply forces to particles inside cylinder
        mask_inside = penetrations > 0

        # Calculate force magnitudes using Hooke's law
        force_magnitudes = np.zeros_like(radial_distances)
        force_magnitudes[mask_inside] = self.spring_constant * penetrations[mask_inside]

        # Calculate final force vectors
        forces = outward_directions * force_magnitudes[:, np.newaxis]

        return forces

    def on_time_step(self, event: Event) -> None:
        """Apply and store repulsion forces on each time step using vectorized operations."""
        # Get current state of all unfrozen particles
        pop = self.population_view.get(event.index, query="frozen == False")
        if pop.empty:
            return

        # Extract positions as numpy array
        positions = pop[["x", "y", "z"]].to_numpy()

        # Calculate forces for all particles at once
        forces = self.calculate_forces_vectorized(positions)

        # Store forces in population
        pop.loc[:, ["fx", "fy", "fz"]] += forces

        # Update velocities based on forces
        dt = event.step_size / pd.Timedelta(days=1)
        pop[["vx", "vy", "vz"]] += forces * dt

        # Update population
        self.population_view.update(pop)


class MagneticRepulsion(Component):
    """Component that creates a point-based magnetic repulsion for active particles and stores forces."""

    CONFIGURATION_DEFAULTS = {
        "magnetic_repulsion": {
            "position": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            },
            "strength": 0.05,  # Magnetic field strength
            "min_distance": 0.1,  # Minimum distance before force capping
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "frozen", "path_id"]

    def setup(self, builder: Builder) -> None:
        """Setup the magnetic component."""
        self.config = builder.configuration.magnetic_repulsion
        
        # Get position parameters
        self.position = np.array([
            float(self.config.position.x),
            float(self.config.position.y),
            float(self.config.position.z)
        ])
        
        # Get other parameters
        self.strength = float(self.config.strength)
        self.min_distance = float(self.config.min_distance)

    def calculate_magnetic_forces(self, positions: np.ndarray) -> np.ndarray:
        """Calculate magnetic forces on particles based on their positions.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of shape (n_particles, 3) containing particle positions
            
        Returns
        -------
        np.ndarray
            Array of shape (n_particles, 3) containing force vectors
        """
        # Calculate displacement vectors from magnetic source to particles
        displacements = self.position - positions
        
        # Calculate distances
        distances = np.sqrt(np.sum(displacements**2, axis=1))
        
        # Apply minimum distance to prevent excessive forces
        distances = np.maximum(distances, self.min_distance)
        
        # Calculate force magnitudes (inverse square law)
        force_magnitudes = self.strength / (distances ** 2)
        
        # Calculate normalized direction vectors
        with np.errstate(invalid='ignore', divide='ignore'):
            direction_vectors = displacements / distances[:, np.newaxis]
        direction_vectors = np.nan_to_num(direction_vectors)
        
        # Calculate force vectors (negative for repulsion)
        forces = -direction_vectors * force_magnitudes[:, np.newaxis]
        
        return forces

    def on_time_step(self, event: Event) -> None:
        """Apply and store magnetic forces on each time step."""
        # Get current state of all particles
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        
        # Get active particles (not frozen and with a path)
        active_mask = (~pop.frozen) & (pop.path_id.notna())
        active = pop[active_mask]
        
        if len(active) == 0:
            return
            
        # Get positions as numpy array
        positions = active[["x", "y", "z"]].values
        
        # Calculate forces
        forces = self.calculate_magnetic_forces(positions)
        
        # Store forces in population
        active.loc[:, ["fx", "fy", "fz"]] += forces
        
        # Update velocities based on forces
        dt = event.step_size / pd.Timedelta(days=1)
        active.loc[:, ["vx", "vy", "vz"]] += forces * dt
        
        # Update population
        self.population_view.update(active)