from typing import Any, Dict, List

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

class EllipsoidContainment(Component):
    """Component that keeps particles within an ellipsoid boundary using magnetic repulsion.
    
    The force follows an inverse square law relationship, similar to magnetic repulsion,
    rather than the linear relationship of Hooke's law. This creates a stronger repulsion
    as particles get closer to the boundary.
    """

    CONFIGURATION_DEFAULTS = {
        "ellipsoid_containment": {
            "a": 1.0,  # Semi-major axis in x direction
            "b": 1.0,  # Semi-major axis in y direction
            "c": 1.0,  # Semi-major axis in z direction
            "magnetic_strength": 0.1,  # Strength coefficient for magnetic repulsion
            "min_distance": 0.01,  # Minimum distance to prevent singularity
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "frozen"]

    def setup(self, builder: Builder) -> None:
        """Setup the component and register force modifiers."""
        self.config = builder.configuration.ellipsoid_containment

        # Get semi-major axes from config
        self.a = float(self.config.a)
        self.b = float(self.config.b)
        self.c = float(self.config.c)
        self.magnetic_strength = float(self.config.magnetic_strength)
        self.min_distance = float(self.config.min_distance)

        # Pre-compute squared denominators for gradient calculation
        self.a2 = self.a * self.a
        self.b2 = self.b * self.b
        self.c2 = self.c * self.c

        # Initialize cache
        self.force_cache = {}
        self.clock = builder.time.clock()

        # Register force modifiers for each component
        builder.value.register_value_modifier(
            "particle.force.x",
            modifier=self.magnetic_force_x,
            requires_columns=["x", "y", "z", "frozen"]
        )
        builder.value.register_value_modifier(
            "particle.force.y",
            modifier=self.magnetic_force_y,
            requires_columns=["x", "y", "z", "frozen"]
        )
        builder.value.register_value_modifier(
            "particle.force.z",
            modifier=self.magnetic_force_z,
            requires_columns=["x", "y", "z", "frozen"]
        )

    def calculate_forces_vectorized(self, positions: np.ndarray) -> np.ndarray:
        """Calculate magnetic repulsion forces for all particles using vectorized operations."""
        # Calculate normalized position coordinates
        x_norm = positions[:, 0] / self.a
        y_norm = positions[:, 1] / self.b
        z_norm = positions[:, 2] / self.c
        
        # Calculate ellipsoid equation value at each point
        ellipsoid_val = x_norm**2 + y_norm**2 + z_norm**2
        
        # Calculate gradient components (normal to the ellipsoid surface)
        dx = 2 * x_norm / self.a
        dy = 2 * y_norm / self.b
        dz = 2 * z_norm / self.c
        
        # Stack gradient components
        gradient = np.column_stack([dx, dy, dz])
        
        # Calculate force only for points outside ellipsoid (ellipsoid_val > 1)
        outside_mask = ellipsoid_val > 1
        
        # Initialize forces array
        forces = np.zeros_like(positions)
        
        if np.any(outside_mask):
            # For points outside, calculate repulsive force
            gradient_outside = gradient[outside_mask]
            
            # Normalize the gradient vectors
            gradient_norms = np.linalg.norm(gradient_outside, axis=1, keepdims=True)
            normalized_gradients = gradient_outside / gradient_norms
            
            # Calculate distance from surface for inverse square law
            distances = np.maximum(np.sqrt(ellipsoid_val[outside_mask]) - 1, self.min_distance)
            
            # Calculate force magnitude using inverse square law
            # Force = magnetic_strength / distance^2
            force_magnitudes = self.magnetic_strength / (distances * distances)
            
            # Calculate forces (pointing inward)
            forces[outside_mask] = -normalized_gradients * force_magnitudes[:, np.newaxis]
        
        return forces

    def get_cached_forces(self, index: pd.Index) -> np.ndarray:
        """Get cached forces or calculate them if needed."""
        current_time = self.clock()
        cache_key = (current_time, tuple(index))
        
        if cache_key not in self.force_cache:
            pop = self.population_view.get(index)
            active_particles = pop[~pop.frozen]
            
            if active_particles.empty:
                self.force_cache[cache_key] = np.zeros((len(index), 3))
            else:
                positions = active_particles[["x", "y", "z"]].to_numpy()
                forces = np.zeros((len(index), 3))
                active_forces = self.calculate_forces_vectorized(positions)
                forces[active_particles.index.get_indexer(active_particles.index)] = active_forces
                self.force_cache[cache_key] = forces
                
            # Clear old cache entries
            self.force_cache = {k: v for k, v in self.force_cache.items() if k[0] == current_time}
            
        return self.force_cache[cache_key]

    def magnetic_force_x(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add x-component of magnetic repulsion force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 0], index=index)
        return forces

    def magnetic_force_y(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add y-component of magnetic repulsion force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 1], index=index)
        return forces

    def magnetic_force_z(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add z-component of magnetic repulsion force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 2], index=index)
        return forces

class CylinderExclusion(Component):
    """Component that repels particles from inside a cylindrical exclusion zone using Hooke's law."""

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
        return ["x", "y", "z", "frozen"]

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

        # Initialize force cache
        self.force_cache = {}
        self.clock = builder.time.clock()

        # Register force modifiers for each component
        builder.value.register_value_modifier(
            "particle.force.x",
            modifier=self.cylinder_force_x,
            requires_columns=["x", "y", "z", "frozen"]
        )
        builder.value.register_value_modifier(
            "particle.force.y",
            modifier=self.cylinder_force_y,
            requires_columns=["x", "y", "z", "frozen"]
        )
        builder.value.register_value_modifier(
            "particle.force.z",
            modifier=self.cylinder_force_z,
            requires_columns=["x", "y", "z", "frozen"]
        )

    def calculate_forces_vectorized(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized calculation of forces for all particles."""
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

    def get_cached_forces(self, index: pd.Index) -> np.ndarray:
        """Get cached forces or calculate them if needed."""
        current_time = self.clock()
        cache_key = (current_time, tuple(index))
        
        if cache_key not in self.force_cache:
            pop = self.population_view.get(index)
            active_particles = pop[~pop.frozen]
            
            if active_particles.empty:
                self.force_cache[cache_key] = np.zeros((len(index), 3))
            else:
                positions = active_particles[["x", "y", "z"]].to_numpy()
                forces = np.zeros((len(index), 3))
                active_forces = self.calculate_forces_vectorized(positions)
                forces[active_particles.index.get_indexer(active_particles.index)] = active_forces
                self.force_cache[cache_key] = forces
                
            # Clear old cache entries
            self.force_cache = {k: v for k, v in self.force_cache.items() if k[0] == current_time}
            
        return self.force_cache[cache_key]

    def cylinder_force_x(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add x-component of cylinder repulsion force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 0], index=index)
        return forces

    def cylinder_force_y(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add y-component of cylinder repulsion force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 1], index=index)
        return forces

    def cylinder_force_z(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add z-component of cylinder repulsion force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 2], index=index)
        return forces


class MagneticRepulsion(Component):
    """Component that creates a point-based magnetic repulsion for active particles"""

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
        return ["x", "y", "z", "frozen", "path_id"]

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

        # Initialize force cache
        self.force_cache = {}
        self.clock = builder.time.clock()

        # Register force modifiers for each component
        builder.value.register_value_modifier(
            "particle.force.x",
            modifier=self.magnetic_force_x,
            requires_columns=["x", "y", "z", "frozen", "path_id"]
        )
        builder.value.register_value_modifier(
            "particle.force.y",
            modifier=self.magnetic_force_y,
            requires_columns=["x", "y", "z", "frozen", "path_id"]
        )
        builder.value.register_value_modifier(
            "particle.force.z",
            modifier=self.magnetic_force_z,
            requires_columns=["x", "y", "z", "frozen", "path_id"]
        )

    def calculate_magnetic_forces(self, positions: np.ndarray) -> np.ndarray:
        """Calculate magnetic forces on particles based on their positions."""
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

    def get_cached_forces(self, index: pd.Index) -> np.ndarray:
        """Get cached forces or calculate them if needed."""
        current_time = self.clock()
        cache_key = (current_time, tuple(index))
        
        if cache_key not in self.force_cache:
            pop = self.population_view.get(index)
            
            # Get active particles (not frozen and with a path)
            active_mask = (~pop.frozen) & (pop.path_id.notna())
            active = pop[active_mask]
            
            # Initialize forces array
            forces = np.zeros((len(index), 3))
            
            if not active.empty:
                # Calculate forces for active particles
                positions = active[["x", "y", "z"]].to_numpy()
                active_forces = self.calculate_magnetic_forces(positions)
                
                # Assign forces to correct indices
                forces[active.index.get_indexer(active.index)] = active_forces
            
            self.force_cache[cache_key] = forces
            
            # Clear old cache entries
            self.force_cache = {k: v for k, v in self.force_cache.items() if k[0] == current_time}
            
        return self.force_cache[cache_key]

    def magnetic_force_x(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add x-component of magnetic force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 0], index=index)
        return forces

    def magnetic_force_y(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add y-component of magnetic force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 1], index=index)
        return forces

    def magnetic_force_z(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        """Add z-component of magnetic force."""
        forces += pd.Series(self.get_cached_forces(index)[:, 2], index=index)
        return forces