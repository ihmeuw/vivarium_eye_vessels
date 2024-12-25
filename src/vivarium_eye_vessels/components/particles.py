from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class Particle3D(Component):
    """Base component for managing 3D particle positions, velocities, and forces."""

    @property
    def columns_created(self) -> List[str]:
        return [
            "x", "y", "z",
            "vx", "vy", "vz",
            "blocked_time",     # Time spent in high-force state
            "frozen",
            "parent_id",
            "path_id",
            "creation_time",
            "frozen_duration",
        ]

    CONFIGURATION_DEFAULTS = {
        "particles": {
            "step_size": 0.01,
            "overall_max_velocity_change": 0.1,
            "initial_velocity_range": (-0.05, 0.05),
            "initial_circle": {"center": [1.5, 0.0, 0.5], "radius": 0.1, "n_vessels": 5},
            "force_blocking_threshold": 0.5,  # Threshold for considering a particle blocked
            "blocked_time_threshold": 1.0,    # Time threshold (in simulation days) for terminating blocked paths
        }
    }

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.particles
        self.step_size = self.config.step_size
        self.overall_max_velocity_change = self.config.overall_max_velocity_change
        self.initial_velocity_range = self.config.initial_velocity_range
        self.force_blocking_threshold = self.config.force_blocking_threshold
        self.blocked_time_threshold = self.config.blocked_time_threshold

        self.clock = builder.time.clock()

        # Register force pipelines
        self.register_force_pipelines(builder)

        self.max_velocity_change = builder.value.register_value_producer(
            "particle.max_velocity_change",
            source=lambda index: pd.Series(self.overall_max_velocity_change, index=index),
        )

        self.randomness = builder.randomness.get_stream("particle.particles_3d")
        self.builder = builder

    def register_force_pipelines(self, builder: Builder) -> None:
        """Register pipelines for force components and total magnitude."""
        # Register individual force component pipelines
        self.force_x = builder.value.register_value_producer(
            "particle.force.x",
            source=lambda index: pd.Series(0.0, index=index)
        )
        self.force_y = builder.value.register_value_producer(
            "particle.force.y",
            source=lambda index: pd.Series(0.0, index=index)
        )
        self.force_z = builder.value.register_value_producer(
            "particle.force.z",
            source=lambda index: pd.Series(0.0, index=index)
        )

        # Register total force magnitude pipeline
        self.force_magnitude = builder.value.register_value_producer(
            "particle.force.magnitude",
            source=self.get_force_magnitude,
            requires_values=['particle.force.x', 'particle.force.y', 'particle.force.z']
        )

    def get_force_magnitude(self, index: pd.Index) -> pd.Series:
        """Calculate total force magnitude from components."""
        fx = self.force_x(index)
        fy = self.force_y(index)
        fz = self.force_z(index)
        return np.sqrt(fx**2 + fy**2 + fz**2)

    def on_initialize_simulants(self, simulant_data: SimulantData) -> None:
        """Initialize particles with positions, velocities, and path tracking information."""
        pop = pd.DataFrame(index=simulant_data.index)

        has_ellipsoid = "ellipsoid_containment" in self.builder.components.list_components()

        if has_ellipsoid:
            # Get ellipsoid parameters
            config = self.builder.configuration.ellipsoid_containment
            a = float(config.a)
            b = float(config.b)
            c = float(config.c)
            self.scale = np.array([a,b,c])
        else:
            self.scale = np.ones(3)

        # Generate 3D normal points using ppf (inverse CDF)
        points = np.column_stack([
            norm.ppf(self.randomness.get_draw(pop.index, additional_key=f'xyz_{i}'))
            for i in range(3)
        ])
    
        # Normalize and scale by random radius
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
        radii = np.array(self.randomness.get_draw(pop.index, additional_key='radius'))**(1/3)
        points *= radii[:, np.newaxis]
        pop[["x", "y", "z"]] = points 
        
        pop[["x", "y", "z"]] *= self.scale

        # Generate random initial velocities
        v_range = self.initial_velocity_range
        for v in ["vx", "vy", "vz"]:
            pop[v] = (
                self.randomness.get_draw(pop.index, additional_key=v)
                * (v_range[1] - v_range[0])
                + v_range[0]
            )
        pop[["vx", "vy", "vz"]] *= self.scale

        # Initialize blocked time and tracking columns
        pop["blocked_time"] = 0.0
        pop["frozen"] = False
        pop["frozen_duration"] = np.nan
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
                    center[2],
                ]
                pop.loc[i, "path_id"] = i
                pop.loc[i, ["vz"]] = [0,]

        self.population_view.update(pop)

    def on_time_step(self, event: Event) -> None:
        """Update positions and velocities of non-frozen particles and track blocking forces."""
        pop = self.population_view.get(event.index)
        active_particles = pop[~pop.frozen]

        if not active_particles.empty:
            self.update_positions(active_particles)
            self.check_blocked_paths(active_particles, event.step_size)

    def update_positions(self, particles: pd.DataFrame) -> None:
        """Update positions and velocities based on forces and random changes."""
        # Update positions based on current velocities
        for pos, vel in [("x", "vx"), ("y", "vy"), ("z", "vz")]:
            particles.loc[:, pos] = particles[pos] + self.step_size * particles[vel]

        # Get max velocity change from pipeline
        max_velocity_change = self.max_velocity_change(particles.index)

        # Get current forces from pipelines
        fx = self.force_x(particles.index)
        fy = self.force_y(particles.index)
        fz = self.force_z(particles.index)

        # Update velocities with random changes and forces
        for i, (v, f) in enumerate(zip(["vx", "vy", "vz"], [fx, fy, fz])):
            # Random velocity change
            dv = (self.randomness.get_draw(particles.index, additional_key=f"d{v}") - 0.5) * 2 * max_velocity_change * self.scale[i]
            
            # Add force contribution to velocity
            particles.loc[:, v] += (dv + f) * self.step_size

        self.population_view.update(particles)

    def check_blocked_paths(self, particles: pd.DataFrame, step_size: pd.Timedelta) -> None:
        """Check for and handle blocked paths based on force magnitude."""
        # Get force magnitude from pipeline
        force_magnitude = self.force_magnitude(particles.index)

        # Update blocked time for particles experiencing high forces
        blocked_mask = force_magnitude > self.force_blocking_threshold
        particles.loc[blocked_mask, "blocked_time"] += step_size / pd.Timedelta(days=1)
        particles.loc[~blocked_mask, "blocked_time"] = 0

        # Freeze particles that have been blocked too long
        to_freeze = particles[
            (particles["blocked_time"] > self.blocked_time_threshold) & 
            (particles["path_id"].notna())
        ]
        
        # if not to_freeze.empty:
        #     to_freeze.loc[:, "frozen"] = True
        #     to_freeze.loc[:, "path_id"] = -1

        #     self.population_view.update(to_freeze)


class PathFreezer(Component):
    """Component for freezing particle paths and creating continuations."""

    CONFIGURATION_DEFAULTS = {
        "path_freezer": {
            "freeze_interval": 10,
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return [
            "x", "y", "z",
            "vx", "vy", "vz",
            "fx", "fy", "fz",
            "frozen", "frozen_duration",
            "parent_id", "path_id",
            "creation_time"
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_freezer
        self.step_count = 0

    def on_time_step(self, event: Event) -> None:
        self.step_count += 1
        if self.step_count % self.config.freeze_interval == 0:
            pop = self.population_view.get(event.index)
            self.freeze_particles(pop)

    def freeze_particles(self, pop: pd.DataFrame) -> None:
        """Create frozen path points and continue paths with new particles."""
        active = pop[~pop.frozen & pop.path_id.notna()]
        if active.empty:
            return

        available = pop[~pop.frozen & pop.path_id.isna()]
        if len(available) >= len(active):
            to_freeze = available.sample(len(active))

            to_freeze = to_freeze.assign(
                x=active.x.values,
                y=active.y.values,
                z=active.z.values,
                vx=active.vx.values,
                vy=active.vy.values,
                vz=active.vz.values,
                fx=active.fx.values,  # Copy force components
                fy=active.fy.values,
                fz=active.fz.values,
                path_id=active.path_id.values,
                parent_id=active.index.values,
                frozen=False,
                frozen_duration=0.0,
            )

            to_freeze["parent_id"] = to_freeze["parent_id"].astype(object)
            self.population_view.update(to_freeze)

        active.loc[:, "frozen"] = True
        self.population_view.update(active)


class PathSplitter(Component):
    """Component for splitting particle paths into two branches."""

    CONFIGURATION_DEFAULTS = {
        "path_splitter": {
            "split_interval": 20,
            "split_angle": 30,
            "split_probability": 0.3,
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return [
            "x", "y", "z",
            "vx", "vy", "vz",
            "fx", "fy", "fz",
            "frozen", "parent_id",
            "path_id", "creation_time"
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_splitter
        self.step_count = 0
        self.next_path_id = 1000
        self.randomness = builder.randomness.get_stream("path_splitter")

    def on_time_step(self, event: Event) -> None:
        self.step_count += 1
        if self.step_count % self.config.split_interval == 0:
            pop = self.population_view.get(event.index)
            self.split_paths(pop)

    def split_paths(self, pop: pd.DataFrame) -> None:
        """Split eligible paths into two branches, freezing the original particle."""
        # Get active particles that have valid path_ids
        active = pop[~pop.frozen & pop.path_id.notna()]
        if active.empty:
            return

        # Determine which paths will split
        split_mask = self.randomness.get_draw(active.index) < self.config.split_probability
        to_split = active[split_mask]
        if to_split.empty:
            return

        # Find available particles for new branches - need two per split
        available = pop[~pop.frozen & pop.path_id.isna()]
        if len(available) < 2 * len(to_split):
            return

        # Sample particles for new branches - two per split point
        new_branches = available.sample(2 * len(to_split))
        angle_rad = np.radians(self.config.split_angle / 2)

        # Small offset distance for initial positions
        offset_distance = 0.05  # Adjust as needed

        # Track updates for frozen originals and new branches
        updates = []

        for idx, (orig_idx, original) in enumerate(to_split.iterrows()):
            vel = np.array([original.vx, original.vy, original.vz])
            speed = np.linalg.norm(vel)
            if speed == 0:
                continue

            # Calculate normalized velocity and perpendicular vector
            vel_norm = vel / speed
            perp = np.array([-vel_norm[1], vel_norm[0], 0])
            if np.all(perp == 0):
                perp = np.array([0, -vel_norm[2], vel_norm[1]])
            perp = perp / np.linalg.norm(perp)

            # Calculate new velocities for both branches
            rot_matrix_1 = self._rotation_matrix(perp, angle_rad)
            rot_matrix_2 = self._rotation_matrix(perp, -angle_rad)
            new_vel_1 = rot_matrix_1 @ vel
            new_vel_2 = rot_matrix_2 @ vel

            # Normalize new velocities for position offsets
            new_vel_1_norm = new_vel_1 / np.linalg.norm(new_vel_1)
            new_vel_2_norm = new_vel_2 / np.linalg.norm(new_vel_2)

            # Calculate offset positions
            original_pos = np.array([original.x, original.y, original.z])
            pos_1 = original_pos + offset_distance * new_vel_1_norm
            pos_2 = original_pos + offset_distance * new_vel_2_norm

            # Create DataFrame rows with correct dtypes from the start
            # Freeze original particle at split point
            original_update = pd.DataFrame(
                {
                    'x': [original.x], 'y': [original.y], 'z': [original.z],
                    'vx': [original.vx], 'vy': [original.vy], 'vz': [original.vz],
                    'fx': [0.0], 'fy': [0.0], 'fz': [0.0],
                    'frozen': [True],
                    'path_id': [-1],
                    'parent_id': [original.parent_id],
                }, index=[orig_idx]
            )
            updates.append(original_update)
            
            # Create first new branch
            new_branch_1 = pd.DataFrame(
                {
                    'x': [pos_1[0]], 'y': [pos_1[1]], 'z': [pos_1[2]],
                    'vx': [new_vel_1[0]], 'vy': [new_vel_1[1]], 'vz': [new_vel_1[2]],
                    'fx': [0.0], 'fy': [0.0], 'fz': [0.0],
                    'frozen': [False],
                    'path_id': [self.next_path_id],
                    'parent_id': [orig_idx],
                }, index=[new_branches.iloc[2*idx].name]
            )
            updates.append(new_branch_1)
            
            # Create second new branch
            new_branch_2 = pd.DataFrame(
                {
                    'x': [pos_2[0]], 'y': [pos_2[1]], 'z': [pos_2[2]],
                    'vx': [new_vel_2[0]], 'vy': [new_vel_2[1]], 'vz': [new_vel_2[2]],
                    'fx': [0.0], 'fy': [0.0], 'fz': [0.0],
                    'frozen': [False],
                    'path_id': [self.next_path_id + 1],
                    'parent_id': [orig_idx],
                }, index=[new_branches.iloc[2*idx + 1].name]
            )
            updates.append(new_branch_2)
            
            self.next_path_id += 2

        if updates:
            # Combine all updates with consistent dtypes
            all_updates = pd.concat(updates, axis=0)
            
            # Ensure object dtypes for id columns
            all_updates['path_id'] = all_updates['path_id'].astype(object)
            all_updates['parent_id'] = all_updates['parent_id'].astype(object)
            
            self.population_view.update(all_updates)

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
        """Return the rotation matrix for rotation around axis by theta radians."""
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
        ])


class PathExtinction(Component):
    """Component for controlling extinction of active paths over time."""

    CONFIGURATION_DEFAULTS = {
        "path_extinction": {
            "extinction_start_time": "2020-01-01",
            "extinction_end_time": "2020-12-31",
            "initial_freeze_probability": 0.0,
            "final_freeze_probability": 0.3,
            "check_interval": 5,
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["frozen", "path_id"]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_extinction
        self.start_time = pd.Timestamp(self.config.extinction_start_time)
        self.end_time = pd.Timestamp(self.config.extinction_end_time)
        self.p_start = self.config.initial_freeze_probability
        self.p_end = self.config.final_freeze_probability
        self.clock = builder.time.clock()
        self.step_count = 0
        self.randomness = builder.randomness.get_stream("path_extinction")

    def get_current_freeze_probability(self) -> float:
        """Calculate current freeze probability based on time."""
        current_time = self.clock()

        if current_time < self.start_time:
            return self.p_start
        elif current_time > self.end_time:
            return self.p_end
        else:
            progress = (current_time - self.start_time) / (self.end_time - self.start_time)
            return self.p_start + (self.p_end - self.p_start) * progress

    def on_time_step(self, event: Event) -> None:
        """Check for path freezing on configured interval."""
        self.step_count += 1

        if self.step_count % self.config.check_interval != 0:
            return

        pop = self.population_view.get(event.index)
        active = pop[~pop.frozen & pop.path_id.notna()]

        if active.empty:
            return

        p_freeze = self.get_current_freeze_probability()
        to_freeze = active[self.randomness.get_draw(active.index) < p_freeze]

        if not to_freeze.empty:
            to_freeze.loc[:, "frozen"] = True
            to_freeze.loc[:, "path_id"] = -1
            self.population_view.update(to_freeze)


class FrozenParticleRepulsion(Component):
    """Component that creates spring-like repulsive forces between active particles and frozen particles."""

    CONFIGURATION_DEFAULTS = {
        "frozen_repulsion": {
            "spring_constant": 0.01,
            "max_distance": 0.5,
            "min_frozen_duration": 1.0,
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return [
            "x", "y", "z", 
            "vx", "vy", "vz",
            "fx", "fy", "fz",
            "frozen", "path_id",
            "frozen_duration"
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.frozen_repulsion
        self.spring_constant = float(self.config.spring_constant)
        self.max_distance = float(self.config.max_distance)
        self.min_frozen_duration = float(self.config.min_frozen_duration)

    def calculate_pairwise_forces(
        self,
        active_positions: np.ndarray,
        active_path_ids: np.ndarray,
        frozen_positions: np.ndarray,
        frozen_path_ids: np.ndarray,
    ) -> np.ndarray:
        """Calculate spring-like repulsive forces between active particles and frozen particles."""
        n_active = len(active_positions)
        n_frozen = len(frozen_positions)

        # Initialize forces array
        net_forces = np.zeros((n_active, 3))

        if n_active == 0 or n_frozen == 0:
            return net_forces

        # Calculate all pairwise displacement vectors
        displacements = active_positions[:, np.newaxis, :] - frozen_positions[np.newaxis, :, :]
        
        # Calculate distances between all pairs
        distances = np.sqrt(np.sum(displacements**2, axis=2))
        
        # Create mask for different paths
        different_paths = active_path_ids[:, np.newaxis] != frozen_path_ids[np.newaxis, :]

        # Calculate spring compression
        compression = distances - self.max_distance

        # Calculate force magnitudes using Hooke's law
        force_magnitudes = np.where(
            (compression < 0) & different_paths,
            -self.spring_constant * compression,
            0.0,
        )

        # Normalize displacement vectors
        with np.errstate(invalid="ignore", divide="ignore"):
            unit_vectors = displacements / distances[..., np.newaxis]
        unit_vectors = np.nan_to_num(unit_vectors)

        # Calculate force vectors and sum contributions
        forces = unit_vectors * force_magnitudes[..., np.newaxis]
        net_forces = np.sum(forces, axis=1)

        return net_forces

    def on_time_step(self, event: Event) -> None:
        """Apply and track repulsion forces on each time step."""
        pop = self.population_view.get(event.index)
        if pop.empty:
            return

        # Update frozen durations
        frozen_mask = pop.frozen
        pop.loc[frozen_mask, "frozen_duration"] += event.step_size / pd.Timedelta(days=1)

        # Split into active and eligible frozen particles
        active_mask = (~pop.frozen) & (pop.path_id.notna())
        active = pop[active_mask]
        
        eligible_frozen = pop[
            (pop.frozen) & (pop.frozen_duration >= self.min_frozen_duration)
        ]

        if len(active) == 0 or len(eligible_frozen) == 0:
            return

        # Calculate forces
        active_positions = active[["x", "y", "z"]].values
        active_path_ids = active["path_id"].values
        frozen_positions = eligible_frozen[["x", "y", "z"]].values
        frozen_path_ids = eligible_frozen["path_id"].values

        forces = self.calculate_pairwise_forces(
            active_positions, active_path_ids, frozen_positions, frozen_path_ids
        )

        # Update force components and velocities
        dt = event.step_size / pd.Timedelta(days=1)
        
        # Store force components
        active.loc[:, "fx"] = forces[:, 0]
        active.loc[:, "fy"] = forces[:, 1]
        active.loc[:, "fz"] = forces[:, 2]
        
        # Update velocities based on forces
        active.loc[:, "vx"] += forces[:, 0] * dt
        active.loc[:, "vy"] += forces[:, 1] * dt
        active.loc[:, "vz"] += forces[:, 2] * dt

        self.population_view.update(active)


class PathDLA(Component):
    """Component for freezing particles at the end of a path using DLA.
    
    The near radius scales exponentially from initial_near_radius to final_near_radius
    between dla_start_time and dla_end_time.
    """

    CONFIGURATION_DEFAULTS = {
        "path_dla": {
            "stickiness": 0.9,
            "initial_near_radius": 0.1,
            "final_near_radius": 0.01,
            "dla_start_time": "2000-01-01",  # Start time for DLA freezing
            "dla_end_time": "2001-01-01",    # End time for radius scaling
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return [
            "x", "y", "z",
            "fx", "fy", "fz",
            "frozen", "path_id",
            "parent_id",
        ]

    def setup(self, builder: Builder) -> None:
        """Setup the component with configuration and validate parameters."""
        self.config = builder.configuration.path_dla
        self.randomness = builder.randomness.get_stream("path_dla")
        self.clock = builder.time.clock()
        
        # Convert times to pandas Timestamps
        self.dla_start_time = pd.Timestamp(self.config.dla_start_time)
        self.dla_end_time = pd.Timestamp(self.config.dla_end_time)
        
        # Validate configuration
        if self.dla_end_time <= self.dla_start_time:
            raise ValueError("dla_end_time must be after dla_start_time")
        
        if self.config.initial_near_radius <= 0 or self.config.final_near_radius <= 0:
            raise ValueError("near radius values must be positive")
            
        if self.config.final_near_radius > self.config.initial_near_radius:
            raise ValueError("final_near_radius must be smaller than initial_near_radius")
            
        # Calculate decay rate for exponential scaling
        total_time = (self.dla_end_time - self.dla_start_time).total_seconds()
        self.decay_rate = -np.log(self.config.final_near_radius / self.config.initial_near_radius) / total_time

    def get_current_near_radius(self) -> float:
        """Calculate the current near radius based on exponential decay."""
        current_time = self.clock()
        
        if current_time < self.dla_start_time:
            return self.config.initial_near_radius
        elif current_time > self.dla_end_time:
            return self.config.final_near_radius
        
        # Calculate time since start
        time_elapsed = (current_time - self.dla_start_time).total_seconds()
        
        # Calculate exponentially decayed radius
        current_radius = self.config.initial_near_radius * np.exp(-self.decay_rate * time_elapsed)
        return current_radius

    def on_time_step(self, event: Event) -> None:
        """Perform DLA freezing with current near radius if after start time."""
        if self.clock() >= self.dla_start_time:
            self.near_radius = self.get_current_near_radius()
            pop = self.population_view.get(event.index)
            self.dla_freeze(pop)

    def dla_freeze(self, pop: pd.DataFrame) -> None:
        """Freeze particles near frozen particles using DLA.
        Only freeze to particles with path_id < 0
        """
        frozen = pop[pop.frozen & (pop.path_id < 0)]
        if frozen.empty:
            return

        not_frozen = pop[~pop.frozen & pop.path_id.isna()]
        if not_frozen.empty:
            return

        tree = sklearn.neighbors.KDTree(frozen[["x", "y", "z"]].values, leaf_size=2)
        near_frozen_indices = tree.query_radius(
            not_frozen[["x", "y", "z"]].values, r=self.near_radius
        )
        
        near_particles = np.array([len(indices) > 0 for indices in near_frozen_indices])
        stickiness_probabilities = self.randomness.get_draw(
            not_frozen.index, additional_key="stickiness"
        )
        
        freeze_condition = stickiness_probabilities < self.config.stickiness
        freeze_mask = near_particles & freeze_condition

        to_freeze = not_frozen[freeze_mask].copy()
        if not to_freeze.empty:
            nearest_frozen_indices = tree.query(
                to_freeze[["x", "y", "z"]].values, k=1, return_distance=False
            )
            
            to_freeze["parent_id"] = frozen.index[nearest_frozen_indices.flatten()].astype(object)
            to_freeze["path_id"] = -1
            to_freeze["path_id"] = to_freeze["path_id"].astype(object)
            to_freeze["frozen"] = True
            
            # Reset forces for newly frozen particles
            to_freeze[["fx", "fy", "fz"]] = 0.0

            self.population_view.update(to_freeze)