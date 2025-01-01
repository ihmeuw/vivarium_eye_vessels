from typing import Dict, List, Protocol
import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder

class ForceCalculator(Protocol):
    """Protocol defining the interface for force calculation strategies"""
    def calculate_force_magnitude(self, distances: np.ndarray) -> np.ndarray:
        """Calculate force magnitudes based on distances"""
        pass

class HookeanForce:
    """Implements Hooke's law force calculation"""
    def __init__(self, spring_constant: float):
        self.spring_constant = spring_constant
        
    def calculate_force_magnitude(self, distances: np.ndarray) -> np.ndarray:
        return self.spring_constant * distances

class MagneticForce:
    """Implements inverse square law force calculation"""
    def __init__(self, magnetic_strength: float, min_distance: float):
        self.magnetic_strength = magnetic_strength
        self.min_distance = min_distance
        
    def calculate_force_magnitude(self, distances: np.ndarray) -> np.ndarray:
        capped_distances = np.maximum(distances, self.min_distance)
        return self.magnetic_strength / (capped_distances * capped_distances)

class BaseForceComponent(Component):
    """Base class for force-based components with shared caching logic"""
    
    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "frozen"]
    
    @property
    def filter_str(self) -> str:
        return "not frozen"
        
    def setup(self, builder: Builder) -> None:
        self.force_cache = {}
        self.clock = builder.time.clock()
        
        # Register force modifiers
        for axis in ['x', 'y', 'z']:
            builder.value.register_value_modifier(
                f"particle.force.{axis}",
                modifier=getattr(self, f"force_{axis}"),
                requires_columns=self.columns_required
            )

    def setup_force_calculator(self, config: Dict) -> None:
        # Set up force calculator
        if config.force_type == "magnetic":
            self.force_calculator = MagneticForce(
                float(config.magnetic_strength),
                float(config.min_distance)
            )
        else:
            self.force_calculator = HookeanForce(float(config.spring_constant))
            
            
    def get_cached_forces(self, index: pd.Index) -> np.ndarray:
        """Get cached forces or calculate them if needed"""
        current_time = self.clock()
        cache_key = (current_time, tuple(index))
        
        if cache_key not in self.force_cache:
            pop = self.population_view.get(index)
            active_particles = pop.query(self.filter_str)
            
            if active_particles.empty:
                self.force_cache[cache_key] = np.zeros((len(index), 3))
            else:
                forces = np.zeros((len(index), 3))
                active_forces = self.calculate_forces_vectorized(active_particles)
                forces[active_particles.index.get_indexer(active_particles.index)] = active_forces
                self.force_cache[cache_key] = forces
                
            # Clear old cache entries
            self.force_cache = {k: v for k, v in self.force_cache.items() if k[0] == current_time}
            
        return self.force_cache[cache_key]
        
    def force_x(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        forces += pd.Series(self.get_cached_forces(index)[:, 0], index=index)
        return forces
        
    def force_y(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        forces += pd.Series(self.get_cached_forces(index)[:, 1], index=index)
        return forces
        
    def force_z(self, index: pd.Index, forces: pd.Series) -> pd.Series:
        forces += pd.Series(self.get_cached_forces(index)[:, 2], index=index)
        return forces

class EllipsoidContainment(BaseForceComponent):
    """Component that keeps particles within an ellipsoid boundary"""
    
    CONFIGURATION_DEFAULTS = {
        "ellipsoid_containment": {
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "force_type": "hookean",  # "magnetic" or "hookean"
            "magnetic_strength": 0.1,
            "min_distance": 0.01,
            "spring_constant": 1.0,
        }
    }

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        config = builder.configuration.ellipsoid_containment
        self.setup_force_calculator(config)

        # Set up geometry parameters
        self.a = float(config.a)
        self.b = float(config.b)
        self.c = float(config.c)
        self.a2 = self.a * self.a
        self.b2 = self.b * self.b
        self.c2 = self.c * self.c
            
    def calculate_forces_vectorized(self, particles: pd.DataFrame) -> np.ndarray:
        positions = particles[["x", "y", "z"]].to_numpy()

        # Calculate normalized coordinates
        x_norm = positions[:, 0] / self.a
        y_norm = positions[:, 1] / self.b
        z_norm = positions[:, 2] / self.c
        
        # Calculate ellipsoid equation value
        ellipsoid_val = x_norm**2 + y_norm**2 + z_norm**2
        
        # Calculate gradient components
        grad = np.column_stack([
            2 * x_norm / self.a,
            2 * y_norm / self.b,
            2 * z_norm / self.c
        ])
        
        # Initialize forces array
        forces = np.zeros_like(positions)
        outside_mask = ellipsoid_val > 1
        
        if np.any(outside_mask):
            grad_outside = grad[outside_mask]
            grad_norms = np.linalg.norm(grad_outside, axis=1, keepdims=True)
            normalized_grads = grad_outside / grad_norms
            
            # Calculate distances from surface
            distances = np.sqrt(ellipsoid_val[outside_mask]) - 1
            
            # Calculate force magnitudes using the selected force calculator
            force_magnitudes = self.force_calculator.calculate_force_magnitude(distances)
            
            # Calculate final forces
            forces[outside_mask] = -normalized_grads * force_magnitudes[:, np.newaxis]
            
        return forces

class CylinderExclusion(BaseForceComponent):
    """Component that repels particles from inside a cylindrical exclusion zone"""
    
    CONFIGURATION_DEFAULTS = {
        "cylinder_exclusion": {
            "radius": 1.0,
            "center": [0.0, 0.0, 0.0],
            "direction": [0.0, 0.0, 1.0],
            "force_type": "hookean",  # or "magnetic"
            "magnetic_strength": 0.1,
            "min_distance": 0.01,
            "spring_constant": 0.1,
        }
    }
    
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        
        config = builder.configuration.cylinder_exclusion
        self.setup_force_calculator(config)
        
        # Set up geometry parameters
        self.radius = float(config.radius)
        self.center = np.array(config.center, dtype=float)
        self.direction = np.array(config.direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        
        # Pre-compute random perpendicular vector
        random_perpendicular = np.array([1, 0, 0]) if abs(self.direction[0]) < 0.9 else np.array([0, 1, 0])
        self.default_outward = np.cross(self.direction, random_perpendicular)
        self.default_outward /= np.linalg.norm(self.default_outward)
        
            
    def calculate_forces_vectorized(self, particles: pd.DataFrame) -> np.ndarray:
        positions = particles[["x", "y", "z"]].to_numpy()

        # Calculate relative positions and components
        rel_positions = positions - self.center
        axial_dots = np.dot(rel_positions, self.direction)
        axial_components = axial_dots[:, np.newaxis] * self.direction
        radial_vectors = rel_positions - axial_components
        radial_distances = np.linalg.norm(radial_vectors, axis=1)
        
        # Calculate penetration depths
        penetrations = self.radius - radial_distances
        
        # Handle points on axis
        mask_on_axis = radial_distances < 1e-10
        outward_directions = np.zeros_like(positions)
        outward_directions[mask_on_axis] = self.default_outward
        
        # Calculate outward directions for off-axis points
        mask_off_axis = ~mask_on_axis
        outward_directions[mask_off_axis] = (
            radial_vectors[mask_off_axis] / 
            radial_distances[mask_off_axis, np.newaxis]
        )
        
        # Apply forces only inside cylinder
        mask_inside = penetrations > 0
        force_magnitudes = np.zeros_like(radial_distances)
        force_magnitudes[mask_inside] = self.force_calculator.calculate_force_magnitude(
            penetrations[mask_inside]
        )
        
        return outward_directions * force_magnitudes[:, np.newaxis]

class PointRepulsion(BaseForceComponent):
    """Component that creates a point-based repulsion force"""
    
    CONFIGURATION_DEFAULTS = {
        "point_repulsion": {
            "position": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            },
            "force_type": "magnetic",  # or "hookean" 
            "magnetic_strength": 0.05,
            "min_distance": 0.1,
            "spring_constant": 0.1,
            "radius": 0.05,  # Interaction radius
        }
    }
    
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        
        config = builder.configuration.point_repulsion
        self.setup_force_calculator(config)

        self.position = np.array([
            float(config.position.x),
            float(config.position.y),
            float(config.position.z)
        ])
        self.radius = float(config.radius)
        
    def calculate_forces_vectorized(self, particles: pd.DataFrame) -> np.ndarray:
        positions = particles[["x", "y", "z"]].to_numpy()

        # Calculate displacements and distances
        displacements = self.position - positions
        distances = np.sqrt(np.sum(displacements**2, axis=1))
        distances = np.where(distances > self.radius, 0, distances)
        
        # Calculate normalized directions
        with np.errstate(invalid='ignore', divide='ignore'):
            direction_vectors = displacements / distances[:, np.newaxis]
        direction_vectors = np.nan_to_num(direction_vectors)
        
        # Calculate force magnitudes using selected force calculator
        force_magnitudes = self.force_calculator.calculate_force_magnitude(distances)
        
        # Return repulsive forces
        return -direction_vectors * force_magnitudes[:, np.newaxis]

class FrozenRepulsion(BaseForceComponent):
    """Component that repels active particles from frozen particles using spatial indexing"""
    
    CONFIGURATION_DEFAULTS = {
        "frozen_repulsion": {
            "radius": 0.05,  # Interaction radius
            "force_type": "magnetic",  # or "hookean"
            "magnetic_strength": 0.1,
            "min_distance": 0.01,
            "spring_constant": 0.1,
            "delay": 1.0,  # days frozen before exerting force on particles in same path
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return super().columns_required + ["freeze_time", "path_id", "parent_id"]

    @property
    def filter_str(self) -> str:
        return "not frozen and path_id >= 0"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        config = builder.configuration.frozen_repulsion
        self.setup_force_calculator(config)
        self.clock = builder.time.clock()

        self.radius = float(config.radius)
        self.delay = float(config.delay)
        self.freezer = builder.components.get_component("path_freezer")

    def calculate_forces_vectorized(self, particles: pd.DataFrame) -> np.ndarray:
        """Calculate repulsion forces from frozen particles"""
        positions = particles[["x", "y", "z"]].to_numpy()

        forces = np.zeros_like(positions)
        neighbor_lists = self.freezer.query_radius(positions, self.radius)
        
        if neighbor_lists is None:
            return forces
            
        for i, frozen_neighbors in enumerate(neighbor_lists):
            # Calculate displacement vectors from frozen particles
            frozen = self.freezer.get_population(frozen_neighbors)
            frozen = frozen[((frozen.path_id == particles.iloc[i].path_id) 
                & ((self.clock() - frozen.freeze_time)/pd.Timedelta(days=1) > self.delay))
                | (frozen.path_id != particles.iloc[i].path_id)]
            frozen_neighbor_positions = frozen[["x", "y", "z"]].to_numpy()
            displacements = positions[i] - frozen_neighbor_positions
            
            # Calculate distances
            distances = np.sqrt(np.sum(displacements**2, axis=1))
            
            # Calculate normalized direction vectors
            with np.errstate(invalid='ignore', divide='ignore'):
                directions = displacements / distances[:, np.newaxis]
            directions = np.nan_to_num(directions)
            
            # Calculate and sum forces from all frozen neighbors
            force_magnitudes = self.force_calculator.calculate_force_magnitude(
                self.radius - distances
            )
            forces[i] = np.sum(
                directions * force_magnitudes[:, np.newaxis], axis=0
            )
        
        return forces
