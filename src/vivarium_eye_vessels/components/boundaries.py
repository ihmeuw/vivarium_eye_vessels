from typing import Any, Dict, List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class EllipsoidContainment(Component):
    """Component that keeps particles within an ellipsoid boundary using Hooke's law.

    This component applies an inward spring force proportional to how far particles
    have moved beyond the ellipsoid surface, pulling them back into the containment field.
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
        return ["x", "y", "z", "vx", "vy", "vz", "frozen"]

    def setup(self, builder: Builder) -> None:
        """Setup the component."""
        self.config = builder.configuration.ellipsoid_containment

        # Get semi-major axes from config
        self.a = float(self.config.a)
        self.b = float(self.config.b)
        self.c = float(self.config.c)
        self.spring_constant = float(self.config.spring_constant)

        # Register with time stepping system
        builder.event.register_listener("time_step", self.on_time_step)

    def calculate_ellipsoid_metrics(self, x: float, y: float, z: float) -> tuple[float, np.ndarray]:
        """Calculate distance beyond ellipsoid surface and direction of restoring force.
        
        Returns
        -------
        tuple[float, np.ndarray]
            - Distance beyond surface (positive means outside ellipsoid)
            - Unit vector pointing inward toward ellipsoid
        """
        # Calculate normalized radial distance
        d = np.sqrt((x / self.a) ** 2 + (y / self.b) ** 2 + (z / self.c) ** 2)
        
        if d < 1e-10:  # Handle point at origin
            return 0.0, np.zeros(3)
            
        # Calculate how far beyond surface (positive means outside)
        surface_distance = (d - 1.0)  
        
        # Calculate direction for force (pointing INWARD)
        # Note: These are proportional to the gradient, but we negate to point inward
        direction = np.array([
            -x / (self.a**2 * d),
            -y / (self.b**2 * d),
            -z / (self.c**2 * d)
        ])
        
        # Normalize the direction vector
        direction_magnitude = np.linalg.norm(direction)
        if direction_magnitude > 0:
            direction = direction / direction_magnitude
            
        return surface_distance, direction

    def calculate_repulsion_force(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """Calculate the spring-like restoring force vector using Hooke's law."""
        surface_distance, inward_direction = self.calculate_ellipsoid_metrics(x, y, z)
        
        # Only apply force if outside ellipsoid (positive surface_distance)
        if surface_distance <= 0:
            return 0.0, 0.0, 0.0

        # Apply Hooke's law: F = kx where:
        # - k is spring constant
        # - x is distance beyond surface
        # - direction is already pointing inward
        force_magnitude = self.spring_constant * surface_distance
        force_vector = force_magnitude * inward_direction
        
        return force_vector[0], force_vector[1], force_vector[2]

    def on_time_step(self, event: Event) -> None:
        """Apply restoring forces on each time step."""
        # Get current state of all unfrozen particles
        pop = self.population_view.get(event.index, query="frozen == False")
        if pop.empty:
            return

        # Calculate and apply forces for each particle
        forces = np.array(
            [
                self.calculate_repulsion_force(row.x, row.y, row.z)
                for idx, row in pop.iterrows()
            ]
        )

        # Update velocities based on forces
        dt = event.step_size / pd.Timedelta(days=1)
        pop["vx"] += forces[:, 0] * dt
        pop["vy"] += forces[:, 1] * dt
        pop["vz"] += forces[:, 2] * dt

        # Update population
        self.population_view.update(pop)


class CylinderExclusion(Component):
    """Component that repels particles from inside a cylindrical exclusion zone using Hooke's law.
    Only considers radial distance from cylinder axis.
    """

    CONFIGURATION_DEFAULTS = {
        "cylinder_exclusion": {
            "radius": 1.0,  # Radius of the cylinder
            "center": [0.0, 0.0, 0.0],  # Center of the cylinder
            "direction": [0.0, 0.0, 1.0],  # Direction vector of the cylinder (default along z-axis)
            "spring_constant": 0.1,  # Spring constant for Hooke's law (force/distance)
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "frozen"]

    def setup(self, builder: Builder) -> None:
        """Setup the component."""
        self.config = builder.configuration.cylinder_exclusion

        # Get cylinder parameters from config
        self.radius = float(self.config.radius)
        self.center = np.array(self.config.center, dtype=float)
        self.direction = np.array(self.config.direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)  # Ensure direction is a unit vector
        self.spring_constant = float(self.config.spring_constant)

        # Register with time stepping system
        builder.event.register_listener("time_step", self.on_time_step)

    def calculate_radial_penetration(self, x: float, y: float, z: float) -> tuple[float, np.ndarray]:
        """Calculate radial penetration into cylinder and direction of force.
        
        Returns:
        --------
        tuple[float, np.ndarray]
            - Penetration depth (positive means inside cylinder)
            - Unit vector pointing radially outward
        """
        # Get vector from center to point
        position = np.array([x, y, z], dtype=float) - self.center
        
        # Project onto cylinder axis
        axial_component = np.dot(position, self.direction) * self.direction
        
        # Get radial vector (perpendicular to axis)
        radial_vector = position - axial_component
        radial_distance = np.linalg.norm(radial_vector)

        # Calculate penetration (positive means inside cylinder)
        penetration = self.radius - radial_distance
        
        # Handle point exactly on axis with random outward direction
        if radial_distance < 1e-10:
            random_perpendicular = np.array([1, 0, 0]) if abs(self.direction[0]) < 0.9 else np.array([0, 1, 0])
            outward_direction = np.cross(self.direction, random_perpendicular)
            outward_direction /= np.linalg.norm(outward_direction)
        else:
            outward_direction = radial_vector / radial_distance

        return penetration, outward_direction

    def calculate_repulsion_force(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """Calculate the spring-like repulsive force vector using Hooke's law."""
        penetration, outward_direction = self.calculate_radial_penetration(x, y, z)
        
        # Only apply force if inside cylinder (positive penetration)
        if penetration <= 0:
            return 0.0, 0.0, 0.0

        # Apply Hooke's law: F = kx where x is penetration depth
        # Force points outward when penetration is positive
        force_magnitude = self.spring_constant * penetration
        force_vector = force_magnitude * outward_direction
        
        return force_vector[0], force_vector[1], force_vector[2]

    def on_time_step(self, event: Event) -> None:
        """Apply repulsion forces on each time step."""
        # Get current state of all unfrozen particles
        pop = self.population_view.get(event.index, query="frozen == False")
        if pop.empty:
            return

        # Calculate and apply forces for each particle
        forces = np.array(
            [
                self.calculate_repulsion_force(row.x, row.y, row.z)
                for idx, row in pop.iterrows()
            ]
        )

        # Update velocities based on forces
        dt = event.step_size / pd.Timedelta(days=1)
        pop["vx"] += forces[:, 0] * dt
        pop["vy"] += forces[:, 1] * dt
        pop["vz"] += forces[:, 2] * dt

        # Update population
        self.population_view.update(pop)
