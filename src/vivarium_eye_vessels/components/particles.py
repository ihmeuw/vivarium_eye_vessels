from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial import cKDTree

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
            "frozen",
            "depth",
            "parent_id",
            "path_id",
        ]

    CONFIGURATION_DEFAULTS = {
        "particles": {
            "overall_max_velocity_change": 0.1,
            "initial_velocity_range": (-0.05, 0.05),
            "terminal_velocity": 0.2,         # Maximum allowed velocity magnitude
            "initial_circle": {"center": [1.5, 0.0, 0.5], "radius": 0.1, "n_vessels": 5},
        }
    }

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.particles
        self.step_size = builder.configuration.time.step_size
        self.overall_max_velocity_change = self.config.overall_max_velocity_change
        self.initial_velocity_range = self.config.initial_velocity_range
        self.terminal_velocity = self.config.terminal_velocity

        self.clock = builder.time.clock()

        # Register force pipelines
        self.register_force_pipelines(builder)

        self.max_velocity_change = builder.value.register_value_producer(
            "particle.max_velocity_change",
            source=lambda index: pd.Series(self.overall_max_velocity_change, index=index),
        )

        self.randomness = builder.randomness.get_stream("particle.particles_3d")
        self.setup_scale(builder)

    def setup_scale(self, builder):
        has_ellipsoid = "ellipsoid_containment" in builder.components.list_components()

        if has_ellipsoid:
            # Get ellipsoid parameters
            config = builder.configuration.ellipsoid_containment
            a = float(config.a)
            b = float(config.b)
            c = float(config.c)
            self.scale = np.array([a,b,c])
        else:
            self.scale = np.ones(3)

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

        # Generate 3D normal points using ppf (inverse CDF)
        points = np.column_stack([
            norm.ppf(self.randomness.get_draw(pop.index, additional_key=f'xyz_{i}'))
            for i in range(3)
        ])
    
        # Normalize and scale by random radius
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
        radii = np.array(self.randomness.get_draw(pop.index, additional_key='radius'))**(1/3)
        points *= radii[:, np.newaxis]
        pop[["x", "y", "z"]] = points * self.scale

        # Generate random initial velocities
        v_range = self.initial_velocity_range
        for v in ["vx", "vy", "vz"]:
            pop[v] = (
                self.randomness.get_draw(pop.index, additional_key=v)
                * (v_range[1] - v_range[0])
                + v_range[0]
            )
        pop[["vx", "vy", "vz"]] *= self.scale

        # Initialize tree-structure-related columns
        pop["frozen"] = False
        pop["depth"] = -1
        pop["parent_id"] = -1
        pop["path_id"] = -1

        self.initialize_circle_positions(pop)

        self.population_view.update(pop)

    def initialize_circle_positions(self, pop: pd.DataFrame) -> None:
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

    def on_time_step(self, event: Event) -> None:
        """Update positions and velocities of non-frozen particles and track blocking forces."""
        pop = self.population_view.get(event.index)
        active_particles = pop[~pop.frozen]

        if not active_particles.empty:
            self.update_positions(active_particles)

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

        # Apply terminal velocity constraint
        velocity_vectors = particles[["vx", "vy", "vz"]].to_numpy() / self.scale
        velocities_magnitude = np.linalg.norm(velocity_vectors, axis=1)
        over_limit = velocities_magnitude > self.terminal_velocity
        
        if np.any(over_limit):
            # Scale down velocity components to satisfy terminal velocity
            scale_factors = self.terminal_velocity / velocities_magnitude[over_limit]
            particles.loc[over_limit, ["vx", "vy", "vz"]] *= scale_factors[:, np.newaxis]

        self.population_view.update(particles)


class PathFreezer(Component):
    """Component for freezing particle paths and creating continuations.
    Also mantains KDTree of frozen particles for efficient querying."""

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
            "frozen", "depth",
            "parent_id", "path_id",
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_freezer
        self.step_count = 0

        self._current_tree = None
        self._current_frozen = None

    def on_time_step(self, event: Event) -> None:
        self.step_count += 1
        if self.step_count % self.config.freeze_interval == 0:
            pop = self.population_view.get(event.index)
            self.freeze_particles(pop)

            self._current_frozen = pop[pop.frozen]
            if len(self._current_frozen) < 2:
                self._current_tree = None
            else:
                self._current_tree = cKDTree(self._current_frozen[["x", "y", "z"]].values)

    def get_neighbor_pairs(self, radius: float):
        """Get all pairs of frozen particles within radius using efficient pair query."""
        if self._current_tree is None:
            return None

        return self._current_tree.query_pairs(radius)

    def query_radius(self, pop, radius: float):
        """Get neighbor indices for each particle within radius."""
        if self._current_tree is None:
            return None

        return self._current_tree.query_ball_point(pop[["x", "y", "z"]].values, radius)


    def freeze_particles(self, pop: pd.DataFrame) -> None:
        """Create frozen path points and continue paths with new particles."""
        active = pop[~pop.frozen & (pop.path_id >= 0)]
        if active.empty:
            return

        available = pop[~pop.frozen & (pop.path_id < 0)]
        if len(available) >= len(active):
            to_freeze = available.iloc[:len(active)]

            to_freeze = to_freeze.assign(
                x=active.x.values,
                y=active.y.values,
                z=active.z.values,
                vx=active.vx.values,
                vy=active.vy.values,
                vz=active.vz.values,
                path_id=active.path_id.values,
                parent_id=active.index.values,
                frozen=False,
                depth=active.depth.values,
            )

            self.population_view.update(to_freeze)

        active.loc[:, "frozen"] = True
        self.population_view.update(active)


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
            "frozen", "parent_id",
            "path_id",
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_splitter
        self.step_count = 0
        self.next_path_id = builder.configuration.particles.initial_circle.n_vessels+1
        self.step_size = builder.configuration.time.step_size
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
        to_split = self.randomness.filter_for_probability(active.index, self.config.split_probability)
        if to_split.empty:
            return

        # Find available particles for new branches - need two per split
        available = pop[~pop.frozen & pop.path_id.isna()]
        if len(available) < 2 * len(to_split):
            return

        # Sample particles for new branches - two per split point
        new_branches = available.iloc[:(2 * len(to_split))]
        angle_rad = np.radians(self.config.split_angle / 2)

        # Track updates for frozen originals and new branches
        updates = []

        for idx, orig_idx in enumerate(to_split):
            original = active.loc[orig_idx]
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
            pos_1 = original_pos
            pos_2 = original_pos

            # Create DataFrame rows with correct dtypes from the start
            # Freeze original particle at split point
            original_update = pd.DataFrame(
                {
                    'x': [original.x], 'y': [original.y], 'z': [original.z],
                    'vx': [original.vx], 'vy': [original.vy], 'vz': [original.vz],
                    'frozen': [True],
                    'path_id': [original.path_id],
                    'parent_id': [original.parent_id],
                }, index=[orig_idx]
            )
            updates.append(original_update)
            
            # Create first new branch
            new_branch_1 = pd.DataFrame(
                {
                    'x': [pos_1[0]], 'y': [pos_1[1]], 'z': [pos_1[2]],
                    'vx': [new_vel_1[0]], 'vy': [new_vel_1[1]], 'vz': [new_vel_1[2]],
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


class Flock(Component):
    """Component for implementing flocking behavior using shared spatial index"""

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z",
                "vx", "vy", "vz",
                "path_id",
                "frozen"]

    CONFIGURATION_DEFAULTS = {
        "flock": {"radius": 0.05, "alignment_strength": 0.91}
    }

    def setup(self, builder):
        config = builder.configuration.flock
        self.flock_radius = config.radius
        self.alignment_strength = config.alignment_strength
        self.spatial_index = builder.components.get_component("particle_spatial_index")

    def on_time_step(self, event: Event) -> None:
        """Update particle directions based on neighboring particles"""
        pop = self.population_view.get(event.index)
        if len(pop) < 2:
            return
        active = pop[(~pop.frozen) & pop.path_id.notna()]

        # Get neighbor lists for each particle
        neighbor_lists = self.spatial_index.query_radius(active, self.flock_radius)
        if neighbor_lists is None:
            return

        # Track particles that will be updated
        particles_to_update = []
        new_vs = []

        # Calculate new directions based on neighbors
        for i, neighbors in enumerate(neighbor_lists):
            if len(neighbors) > 1:  # Only update if particle has neighbors
                # Calculate average direction of neighbors (excluding self)
                neighbors = [n for n in neighbors if n != i]
                neighbor_v = pop.iloc[neighbors][["vx", "vy", "vz"]].values

                avg_v = np.mean(neighbor_v, axis=0)
                # Interpolate between current direction and neighbor average
                current_v = pop.iloc[i][["vx", "vy", "vz"]].values
                new_v = (
                    1 - self.alignment_strength
                ) * current_v + self.alignment_strength * avg_v

                particles_to_update.append(i)
                new_vs.append(new_v)
        new_vs = np.array(new_vs, dtype=float)
        if particles_to_update:
            updates = pd.DataFrame(
                {"vx": new_vs[:, 0], "vy": new_vs[:, 1], "vz": new_vs[:, 2]},
                index=active.index[particles_to_update],
            )
            self.population_view.update(updates)


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
            
            self.population_view.update(to_freeze)