from typing import List, Dict, Any
import numpy as np
import pandas as pd
from vivarium import Component 
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

class Particle3D(Component):
    """Base component for managing 3D particle positions and velocities."""
    
    @property
    def columns_created(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "frozen", "parent_id", "path_id", "creation_time"]
        
    CONFIGURATION_DEFAULTS = {
        "particles": {
            "step_size": 0.01,
            "overall_max_velocity_change": 0.1,
            "initial_velocity_range": (-0.05, 0.05),
            "initial_circle": {
                "center": [1.5, 0.0, 0.5],
                "radius": 0.1,
                "n_vessels": 5
            }
        }
    }

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.particles
        self.step_size = self.config.step_size
        self.overall_max_velocity_change = self.config.overall_max_velocity_change
        self.initial_velocity_range = self.config.initial_velocity_range
        
        self.clock = builder.time.clock()

        # Register velocity change pipeline
        self.max_velocity_change = builder.value.register_value_producer(
            "particle.max_velocity_change",
            source=lambda index: pd.Series(self.overall_max_velocity_change, index=index),
        )

        self.randomness = builder.randomness.get_stream("particle.particles_3d")

    def on_initialize_simulants(self, simulant_data: SimulantData) -> None:
        """Initialize particles with positions, velocities, and path tracking information."""
        pop = pd.DataFrame(index=simulant_data.index)

        # Generate random positions in [0,1) x [0,1) x [0,1)
        pop["x"] = self.randomness.get_draw(pop.index, additional_key="x")
        pop["y"] = self.randomness.get_draw(pop.index, additional_key="y")
        pop["z"] = self.randomness.get_draw(pop.index, additional_key="z")

        # Generate random initial velocities
        v_range = self.initial_velocity_range
        for v in ['vx', 'vy', 'vz']:
            pop[v] = self.randomness.get_draw(pop.index, additional_key=v) * \
                    (v_range[1] - v_range[0]) + v_range[0]

        # Initialize path tracking columns
        pop["frozen"] = False
        pop["parent_id"] = pd.NA
        pop["path_id"] = pd.NA
        pop["creation_time"] = self.clock()

        # Initialize active vessel in circle position
        config = self.config.initial_circle

        center = config.center
        radius = config.radius
        n_vessels = config.n_vessels

        for i in range(n_vessels):
            if i in pop.index:
                angle = 2 * np.pi * i / n_vessels
                pop.loc[i, ["x", "y", "z"]] = [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[2]
                ]
                pop.loc[i, "path_id"] = i

        self.population_view.update(pop)

    def on_time_step(self, event: Event) -> None:
        """Update positions and velocities of non-frozen particles."""
        pop = self.population_view.get(event.index)
        active_particles = pop[~pop.frozen]
        
        if not active_particles.empty:
            self.update_positions(active_particles)

    def update_positions(self, particles: pd.DataFrame) -> None:
        """Update positions and velocities of active particles."""
        # Update positions based on current velocities
        for pos, vel in [('x','vx'), ('y','vy'), ('z','vz')]:
            particles.loc[:,pos] = particles[pos] + self.step_size * particles[vel]

        # Get max velocity change from pipeline
        max_velocity = self.max_velocity_change(particles.index)

        # Update velocities with random changes
        for v in ['vx', 'vy', 'vz']:
            dv = (self.randomness.get_draw(particles.index, additional_key=f'd{v}') - 0.5) * \
                 2 * max_velocity
            particles.loc[:,v] += dv
            particles.loc[:,v] = np.clip(particles.loc[:,v], -1, 1)

        self.population_view.update(particles)

class PathFreezer(Component):
    """Component for freezing particle paths and creating continuations."""
    
    CONFIGURATION_DEFAULTS = {
        "path_freezer": {
            "freeze_interval": 10,  # Steps between freezing particle positions
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "frozen", "parent_id", "path_id", "creation_time"]
        
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_freezer
        self.step_count = 0

    def on_time_step(self, event: Event) -> None:
        """Freeze particles on configured interval."""
        self.step_count += 1
        
        if self.step_count % self.config.freeze_interval == 0:
            pop = self.population_view.get(event.index)
            self.freeze_particles(pop)

    def freeze_particles(self, pop: pd.DataFrame) -> None:
        """Create frozen path points and continue paths with new particles."""
        # Find active particles (not frozen, with path)
        active = pop[~pop.frozen & pop.path_id.notna()]
        
        if active.empty:
            return

        # Find available particles to use for new branches
        available = pop[~pop.frozen & pop.path_id.isna()]
        
        if len(available) >= len(active):
            to_freeze = available.sample(len(active))

            # Copy positions and velocities from active paths to new particles
            to_freeze = to_freeze.assign(
                x=active.x.values,
                y=active.y.values,
                z=active.z.values,
                vx=active.vx.values,
                vy=active.vy.values,
                vz=active.vz.values,
                path_id=active.path_id.values,
                parent_id=active.index.values,
                frozen=False
            )

            to_freeze['parent_id'] = to_freeze['parent_id'].astype(object)
            self.population_view.update(to_freeze)

        # Mark original path points as frozen
        active.loc[:, "frozen"] = True
        self.population_view.update(active)

class PathSplitter(Component):
    """Component for splitting particle paths into two branches."""
    
    CONFIGURATION_DEFAULTS = {
        "path_splitter": {
            "split_interval": 20,  # Steps between path splits
            "split_angle": 30,  # Angle in degrees between split paths
            "split_probability": 0.3,  # Probability of a path splitting when eligible
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "frozen", "parent_id", "path_id", "creation_time"]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_splitter
        self.step_count = 0
        self.next_path_id = 1000  # Start high to avoid conflicts
        self.randomness = builder.randomness.get_stream("path_splitter")

    def on_time_step(self, event: Event) -> None:
        """Split paths on configured interval."""
        self.step_count += 1
        
        if self.step_count % self.config.split_interval == 0:
            pop = self.population_view.get(event.index)
            self.split_paths(pop)

    def split_paths(self, pop: pd.DataFrame) -> None:
        """Split eligible paths into two branches."""
        # Find active particles (not frozen, with path)
        active = pop[~pop.frozen & pop.path_id.notna()]
        
        if active.empty:
            return
            
        # Randomly select paths to split based on probability
        split_mask = self.randomness.get_draw(active.index) < self.config.split_probability
        to_split = active[split_mask]
        
        if to_split.empty:
            return
            
        # Find available particles to use for new branches
        available = pop[~pop.frozen & pop.path_id.isna()]
        
        if len(available) < len(to_split):
            return
            
        # Select particles to use for new branches
        new_branches = available.sample(len(to_split))
        
        # Calculate split velocities
        angle_rad = np.radians(self.config.split_angle / 2)
        for idx, (_, original) in enumerate(to_split.iterrows()):
            # Create rotation matrix for positive angle
            vel = np.array([original.vx, original.vy, original.vz])
            speed = np.linalg.norm(vel)
            if speed == 0:
                continue
                
            # Normalize velocity and create perpendicular vector
            vel_norm = vel / speed
            perp = np.array([-vel_norm[1], vel_norm[0], 0])
            if np.all(perp == 0):
                perp = np.array([0, -vel_norm[2], vel_norm[1]])
            perp = perp / np.linalg.norm(perp)
            
            # Create rotated velocities
            rot_matrix = self._rotation_matrix(perp, angle_rad)
            new_vel_1 = rot_matrix @ vel
            rot_matrix = self._rotation_matrix(perp, -angle_rad)
            new_vel_2 = rot_matrix @ vel
            
            # Update original particle
            to_split.loc[original.name, ['vx', 'vy', 'vz']] = new_vel_1
            
            # Update new branch
            new_branch = new_branches.iloc[idx]
            new_branches.loc[new_branch.name, ['x', 'y', 'z']] = [
                original.x, original.y, original.z
            ]
            new_branches.loc[new_branch.name, ['vx', 'vy', 'vz']] = new_vel_2
            new_branches.loc[new_branch.name, 'path_id'] = self.next_path_id
            new_branches.loc[new_branch.name, 'parent_id'] = original.name
            self.next_path_id += 1
        
        # Update population with split paths
        self.population_view.update(pd.concat([to_split, new_branches]))

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
        """Return the rotation matrix for rotation around axis by theta radians."""
        axis = axis/np.sqrt(np.dot(axis, axis))
        a = np.cos(theta/2.0)
        b, c, d = -axis*np.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

class FrozenParticleRepulsion(Component):
    """Component that creates repulsive forces between active particles and frozen particles
    from different paths."""
    
    CONFIGURATION_DEFAULTS = {
        'frozen_repulsion': {
            'repulsion_strength': 0.01,  # Base strength of repulsion force
            'min_distance': 0.01,  # Minimum distance to prevent infinite forces
            'max_distance': 0.5,  # Maximum distance for force calculation
            'force_falloff': 2.0,  # Power law for force falloff with distance
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ['x', 'y', 'z', 'vx', 'vy', 'vz', 'frozen', 'path_id']
    
    def setup(self, builder: Builder) -> None:
        """Setup the component."""
        self.config = builder.configuration.frozen_repulsion
        
        # Get parameters from config
        self.repulsion_strength = float(self.config.repulsion_strength)
        self.min_distance = float(self.config.min_distance)
        self.max_distance = float(self.config.max_distance)
        self.force_falloff = float(self.config.force_falloff)
        
        # Register with time stepping system
        builder.event.register_listener('time_step', self.on_time_step)

    def calculate_pairwise_forces(
        self, 
        active_positions: np.ndarray,
        active_path_ids: np.ndarray,
        frozen_positions: np.ndarray,
        frozen_path_ids: np.ndarray
    ) -> np.ndarray:
        """Calculate repulsive forces between active particles and frozen particles from different paths.
        
        Parameters
        ----------
        active_positions : np.ndarray
            Array of shape (n_active, 3) containing positions of active particles
        active_path_ids : np.ndarray
            Array of shape (n_active,) containing path IDs of active particles
        frozen_positions : np.ndarray
            Array of shape (n_frozen, 3) containing positions of frozen particles
        frozen_path_ids : np.ndarray
            Array of shape (n_frozen,) containing path IDs of frozen particles
            
        Returns
        -------
        np.ndarray
            Array of shape (n_active, 3) containing net force vectors for each active particle
        """
        n_active = len(active_positions)
        n_frozen = len(frozen_positions)
        
        # Initialize forces array
        net_forces = np.zeros((n_active, 3))
        
        # If either group is empty, return zero forces
        if n_active == 0 or n_frozen == 0:
            return net_forces
            
        # Calculate all pairwise displacement vectors
        # This creates arrays of shape (n_active, n_frozen, 3)
        displacements = active_positions[:, np.newaxis, :] - frozen_positions[np.newaxis, :, :]
        
        # Calculate distances between all pairs
        # This creates array of shape (n_active, n_frozen)
        distances = np.sqrt(np.sum(displacements**2, axis=2))
        
        # Create mask for different paths
        # This creates array of shape (n_active, n_frozen)
        different_paths = active_path_ids[:, np.newaxis] != frozen_path_ids[np.newaxis, :]
        
        # Apply minimum distance to prevent infinite forces
        distances = np.maximum(distances, self.min_distance)
        
        # Calculate force magnitudes using power law falloff
        # Zero out forces beyond max_distance and for same path
        force_magnitudes = np.where(
            (distances <= self.max_distance) & different_paths,
            self.repulsion_strength / (distances ** self.force_falloff),
            0.0
        )
        
        # Normalize displacement vectors
        with np.errstate(invalid='ignore', divide='ignore'):
            unit_vectors = displacements / distances[..., np.newaxis]
        unit_vectors = np.nan_to_num(unit_vectors)
        
        # Calculate force vectors and sum contributions from all frozen particles
        forces = unit_vectors * force_magnitudes[..., np.newaxis]
        net_forces = np.sum(forces, axis=1)
        
        return net_forces

    def on_time_step(self, event: Event) -> None:
        """Apply repulsion forces on each time step."""
        # Get current state of all particles
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
            
        # Split into active and frozen particles
        active_mask = (~pop.frozen) & (pop.path_id.notna())
        active = pop[active_mask]
        frozen = pop[pop.frozen]
        
        if len(active) == 0 or len(frozen) == 0:
            return
            
        # Get positions and path IDs as numpy arrays
        active_positions = active[['x', 'y', 'z']].values
        active_path_ids = active['path_id'].values
        frozen_positions = frozen[['x', 'y', 'z']].values
        frozen_path_ids = frozen['path_id'].values
        
        # Calculate forces
        forces = self.calculate_pairwise_forces(
            active_positions, 
            active_path_ids, 
            frozen_positions, 
            frozen_path_ids
        )
        
        # Update velocities based on forces
        dt = event.step_size / pd.Timedelta(days=1)
        
        active.loc[:, 'vx'] += forces[:, 0] * dt
        active.loc[:, 'vy'] += forces[:, 1] * dt
        active.loc[:, 'vz'] += forces[:, 2] * dt
        
        # Update population
        self.population_view.update(active)