from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
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
            # location
            "x",
            "y",
            "z",
            # velocity
            "vx",
            "vy",
            "vz",
            # "freeze" information
            # used to form eye vessels
            "frozen",
            "freeze_time",
            "unfreeze_time",
            "depth",
            # addl information relevant to
            # eye vessel structure
            "parent_id",  # tree structure
            "path_id",  # used to hack PathExtinction dynamics so that splits don't go extinct immediately
        ]

    CONFIGURATION_DEFAULTS = {
        "particles": {
            "overall_max_velocity_change": 0.1,
            "initial_velocity_range": (-0.05, 0.05),
            "terminal_velocity": 0.2,  # Maximum allowed velocity magnitude
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
            self.scale = np.array([a, b, c])
        else:
            self.scale = np.ones(3)

    def register_force_pipelines(self, builder: Builder) -> None:
        """Register pipelines for force components and total magnitude."""
        # Register individual force component pipelines
        self.force_x = builder.value.register_value_producer(
            "particle.force.x", source=lambda index: pd.Series(0.0, index=index)
        )
        self.force_y = builder.value.register_value_producer(
            "particle.force.y", source=lambda index: pd.Series(0.0, index=index)
        )
        self.force_z = builder.value.register_value_producer(
            "particle.force.z", source=lambda index: pd.Series(0.0, index=index)
        )

        # Register total force magnitude pipeline
        self.force_magnitude = builder.value.register_value_producer(
            "particle.force.magnitude",
            source=self.get_force_magnitude,
            requires_values=["particle.force.x", "particle.force.y", "particle.force.z"],
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
        points = np.column_stack(
            [
                norm.ppf(self.randomness.get_draw(pop.index, additional_key=f"xyz_{i}"))
                for i in range(3)
            ]
        )

        # Normalize and scale by random radius
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
        radii = np.array(self.randomness.get_draw(pop.index, additional_key="radius")) ** (
            1 / 3
        )
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
        pop["freeze_time"] = pd.NaT
        pop["unfreeze_time"] = pd.NaT
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
                pop.loc[i, ["vx", "vy", "vz"]] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.0,
                ]
                pop.loc[i, "path_id"] = i
                pop.loc[i, ["depth"]] = 0

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
            dv = (
                (self.randomness.get_draw(particles.index, additional_key=f"d{v}") - 0.5)
                * 2
                * max_velocity_change
                * self.scale[i]
            )

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
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "frozen",
            "freeze_time",
            "depth",
            "parent_id",
            "path_id",
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_freezer
        self.particles_to_add = builder.configuration.population.population_size
        self.step_count = 0
        self.clock = builder.time.clock()

        self._current_tree = None
        self._current_frozen = None
        self.simulant_creator = builder.population.get_simulant_creator()

    def add_particles(self):
        self.simulant_creator(self.particles_to_add)

    def on_time_step(self, event: Event) -> None:
        self.step_count += 1
        if self.step_count % self.config.freeze_interval == 0:
            pop = self.population_view.get(event.index)
            self.freeze_particles(pop)
            self.update_tree(pop)

    def update_tree(self, pop):
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

        if isinstance(pop, pd.DataFrame):
            positions = pop[["x", "y", "z"]].values
        else:
            positions = pop

        return self._current_tree.query_ball_point(positions, radius)

    def get_population(self, indices: List[int]) -> pd.DataFrame:
        pos = self._current_frozen.reindex(indices)
        pos = pos.dropna(how="all")
        return pos

    def freeze_particles(self, pop: pd.DataFrame) -> None:
        """Create frozen path points and continue paths with new particles."""
        active = pop[~pop.frozen & (pop.path_id >= 0)]
        if active.empty:
            return

        available = pop[~pop.frozen & (pop.path_id < 0)]
        if len(available) >= len(active):
            to_freeze = available.iloc[: len(active)]

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
        active.loc[:, "freeze_time"] = self.clock()
        self.population_view.update(active)

        if len(available) < len(active) * 3:
            self.add_particles()


class PathExtinction(Component):
    """Component for controlling extinction of active paths based force."""

    CONFIGURATION_DEFAULTS = {
        "path_extinction": {
            "force_threshold": 10.0,  # Force magnitude threshold for extinction
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["frozen", "freeze_time", "path_id", "vx", "vy", "vz"]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_extinction
        self.force_threshold = self.config.force_threshold
        self.clock = builder.time.clock()

        # Get force pipelines
        self.force_magnitude = builder.value.get_value("particle.force.magnitude")
        self.force_x = builder.value.get_value("particle.force.x")
        self.force_y = builder.value.get_value("particle.force.y")
        self.force_z = builder.value.get_value("particle.force.z")

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        active = pop[~pop.frozen & (pop.path_id >= 0)]

        if active.empty:
            return

        force_values = self.force_magnitude(active.index)
        to_freeze = active[force_values > self.force_threshold]

        if not to_freeze.empty:
            to_freeze.loc[:, "frozen"] = True
            to_freeze.loc[:, "freeze_time"] = self.clock()
            to_freeze.loc[:, "path_id"] = -1  # Mark as end of path
            self.population_view.update(to_freeze)


class PathSplitter(Component):
    """Component for splitting particle paths into two branches."""

    CONFIGURATION_DEFAULTS = {
        "path_splitter": {
            "split_interval": 200,
            "split_angle": 30,
            "split_probability": 0.5,
            "max_depth": 4,
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "frozen",
            "freeze_time",
            "unfreeze_time",
            "depth",
            "parent_id",
            "path_id",
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.path_splitter
        self.particles_to_add = builder.configuration.population.population_size
        self.step_count = 0
        self.next_path_id = builder.configuration.particles.initial_circle.n_vessels + 1
        self.step_size = builder.configuration.time.step_size
        self.randomness = builder.randomness.get_stream("path_splitter")
        self.clock = builder.time.clock()
        self.simulant_creator = builder.population.get_simulant_creator()

    def add_particles(self):
        self.simulant_creator(self.particles_to_add)

    def on_time_step(self, event: Event) -> None:
        self.step_count += 1
        if self.step_count % self.config.split_interval == 0:
            pop = self.population_view.get(event.index)
            self.split_paths(pop)

    def split_paths(self, pop: pd.DataFrame) -> None:
        """Split eligible paths into two branches, freezing the original particle."""
        # Get active particles that have valid path_ids
        mode = "split_unfrozen"
        active = pop[~pop.frozen & (pop.path_id >= 0)]
        if active.empty:
            active_index = self.randomness.filter_for_probability(
                pop[pop.frozen].index, 0.01, "active_empty"
            )
            mode = "split_frozen"
        else:
            active_index = active.index

        # Determine which paths will split
        to_consider = self.randomness.filter_for_probability(
            active_index, self.config.split_probability
        )

        not_too_deep = pop.loc[to_consider, "depth"] < self.config.max_depth
        to_split = to_consider[not_too_deep]

        if to_split.empty:
            return

        # Find available particles for new branches - need two per split
        if mode == "split_unfrozen":
            updates = self.split_unfrozen(pop, to_split)
        elif mode == "split_frozen":
            updates = self.split_frozen(pop, to_split)
        else:
            assert 0, f"mode {mode} not implemented"

        if updates:
            # Combine all updates with consistent dtypes
            all_updates = pd.concat(updates, axis=0)

            self.population_view.update(all_updates)

    def split_frozen(self, pop, to_split):
        available = pop[~pop.frozen & (pop.path_id < 0)]
        if len(available) < len(to_split):
            self.add_particles()
            return

        new_branches = available.iloc[: len(to_split)]

        angle_rad = np.radians(90) * self.randomness.choice(
            to_split, [-1, 1], [0.5, 0.5], "split_direction"
        )
        angle_rad = angle_rad * (
            0.75 + 0.5 * (self.randomness.get_draw(to_split, "split_angle"))
        )

        # Track updates for frozen originals and new branches
        updates = []

        for idx, orig_idx in enumerate(to_split):
            original = pop.loc[orig_idx]
            vel = np.array([original.vx, original.vy, original.vz])
            speed = np.linalg.norm(vel)
            if speed == 0:
                continue

            # Calculate normalized velocity and perpendicular vector
            vel_norm = vel / speed
            perp = np.array([0, -vel_norm[2], vel_norm[1]])
            if np.allclose(perp, 0):
                perp = np.array([-vel_norm[1], vel_norm[0], 0])
            perp = perp / np.linalg.norm(perp)

            # Calculate new velocities for both branches
            rot_matrix_1 = self._rotation_matrix(perp, angle_rad[orig_idx])
            new_vel_1 = rot_matrix_1 @ vel

            # Normalize new velocities for position offsets
            new_vel_1_norm = new_vel_1 / np.linalg.norm(new_vel_1)

            # Calculate offset positions
            original_pos = np.array([original.x, original.y, original.z])
            pos_1 = original_pos + new_vel_1_norm * speed * self.step_size

            new_branch_1 = pd.DataFrame(
                {
                    "x": [pos_1[0]],
                    "y": [pos_1[1]],
                    "z": [pos_1[2]],
                    "vx": [new_vel_1[0]],
                    "vy": [new_vel_1[1]],
                    "vz": [new_vel_1[2]],
                    "frozen": [False],
                    "freeze_time": [pd.NaT],
                    "depth": [original.depth + 1],
                    "path_id": [self.next_path_id],
                    "parent_id": [orig_idx],
                },
                index=[new_branches.iloc[idx].name],
            )
            updates.append(new_branch_1)

        return updates

    def split_unfrozen(self, pop, to_split):
        available = pop[~pop.frozen & (pop.path_id < 0)]
        if len(available) < 2 * len(to_split):
            self.add_particles()
            return

        # Sample particles for new branches - two per split point
        new_branches = available.iloc[: (2 * len(to_split))]
        angle_rad = np.radians(self.config.split_angle / 2)
        angle_rad_1 = angle_rad * (0.5 + (self.randomness.get_draw(to_split, "split_angle")))
        angle_rad_2 = angle_rad * (0.5 + (self.randomness.get_draw(to_split, "split_angle")))

        # Track updates for frozen originals and new branches
        updates = []

        for idx, orig_idx in enumerate(to_split):
            original = pop.loc[orig_idx]
            vel = np.array([original.vx, original.vy, original.vz])
            speed = np.linalg.norm(vel)
            if speed == 0:
                continue

            # Calculate normalized velocity and perpendicular vector
            vel_norm = vel / speed
            perp = np.array([0, -vel_norm[2], vel_norm[1]])
            if np.allclose(perp, 0):
                perp = np.array([-vel_norm[1], vel_norm[0], 0])
            perp = perp / np.linalg.norm(perp)

            # Calculate new velocities for both branches
            rot_matrix_1 = self._rotation_matrix(perp, angle_rad_1[orig_idx])
            rot_matrix_2 = self._rotation_matrix(perp, -angle_rad_2[orig_idx])
            new_vel_1 = rot_matrix_1 @ vel
            new_vel_2 = rot_matrix_2 @ vel

            # Normalize new velocities for position offsets
            new_vel_1_norm = new_vel_1 / np.linalg.norm(new_vel_1)
            new_vel_2_norm = new_vel_2 / np.linalg.norm(new_vel_2)

            # Calculate offset positions
            original_pos = np.array([original.x, original.y, original.z])
            pos_1 = original_pos + new_vel_1_norm * speed * self.step_size
            pos_2 = original_pos + new_vel_2_norm * speed * self.step_size

            # Create DataFrame rows with correct dtypes from the start
            # Freeze original particle at split point
            original_update = pd.DataFrame(
                {
                    "x": [original.x],
                    "y": [original.y],
                    "z": [original.z],
                    "vx": [original.vx],
                    "vy": [original.vy],
                    "vz": [original.vz],
                    "frozen": [True],
                    "freeze_time": [self.clock()],
                    "depth": [original.depth],
                    "path_id": [original.path_id],
                    "parent_id": [original.parent_id],
                },
                index=[orig_idx],
            )
            updates.append(original_update)

            new_branch_1 = pd.DataFrame(
                {
                    "x": [pos_1[0]],
                    "y": [pos_1[1]],
                    "z": [pos_1[2]],
                    "vx": [new_vel_1[0]],
                    "vy": [new_vel_1[1]],
                    "vz": [new_vel_1[2]],
                    "frozen": [False],
                    "freeze_time": [pd.NaT],
                    "depth": [original.depth],
                    "path_id": [self.next_path_id],
                    "parent_id": [orig_idx],
                },
                index=[new_branches.iloc[2 * idx].name],
            )
            updates.append(new_branch_1)

            new_branch_2 = pd.DataFrame(
                {
                    "x": [pos_2[0]],
                    "y": [pos_2[1]],
                    "z": [pos_2[2]],
                    "vx": [new_vel_2[0]],
                    "vy": [new_vel_2[1]],
                    "vz": [new_vel_2[2]],
                    "frozen": [False],
                    "freeze_time": [pd.NaT],
                    "depth": [original.depth + 1],
                    "path_id": [self.next_path_id],
                    "parent_id": [orig_idx],
                },
                index=[new_branches.iloc[2 * idx + 1].name],
            )
            updates.append(new_branch_2)

            self.next_path_id += 2
        return updates

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
        """Return the rotation matrix for rotation around axis by theta radians."""
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )


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
            "dla_end_time": "2001-01-01",  # End time for radius scaling
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return [
            "x",
            "y",
            "z",
            "frozen",
            "freeze_time",
            "depth",
            "path_id",
            "parent_id",
        ]

    def setup(self, builder: Builder) -> None:
        """Setup the component with configuration and validate parameters."""
        self.config = builder.configuration.path_dla
        self.randomness = builder.randomness.get_stream("path_dla")
        self.clock = builder.time.clock()
        self.freezer = builder.components.get_component("path_freezer")

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
        self.decay_rate = (
            -np.log(self.config.final_near_radius / self.config.initial_near_radius)
            / total_time
        )

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
        current_radius = self.config.initial_near_radius * np.exp(
            -self.decay_rate * time_elapsed
        )
        return current_radius

    def on_time_step(self, event: Event) -> None:
        """Perform DLA freezing with current near radius if after start time."""
        if self.clock() >= self.dla_start_time:
            self.near_radius = self.get_current_near_radius()
            pop = self.population_view.get(event.index)
            self.dla_freeze(pop)

    def update_tree(self, pop):
        self._current_frozen = pop[pop.frozen]
        if len(self._current_frozen) < 2:
            self._current_tree = None
        else:
            self._current_tree = cKDTree(self._current_frozen[["x", "y", "z"]].values)

    def dla_freeze(self, pop: pd.DataFrame) -> None:
        """Freeze particles near frozen particles using DLA.
        Only freeze to particles with path_id < 0
        """
        #  only use particles with path_id < 0 (i.e. in frozen DataFrame, not all in freezer object )
        frozen = pop[pop.frozen]  # & (pop.path_id < 0)]
        if frozen.empty:
            return
        self.update_tree(frozen)

        not_frozen = pop[~pop.frozen & (pop.path_id < 0)]
        if not_frozen.empty:
            return

        near_frozen_indices = self._current_tree.query_ball_point(
            not_frozen[["x", "y", "z"]].values, self.near_radius
        )
        near_particles = np.array([len(indices) > 0 for indices in near_frozen_indices])
        stickiness_probabilities = self.randomness.get_draw(
            not_frozen.index, additional_key="stickiness"
        )

        freeze_condition = stickiness_probabilities < self.config.stickiness
        freeze_mask = near_particles & freeze_condition

        to_freeze = not_frozen[freeze_mask].copy()
        if not to_freeze.empty:
            to_freeze["parent_id"] = frozen.index[
                [indices[0] for indices in near_frozen_indices[freeze_mask]]
            ]
            to_freeze["path_id"] = 1
            to_freeze["depth"] = 1000
            to_freeze["frozen"] = False
            to_freeze["freeze_time"] = pd.NaT  # self.clock()

            self.population_view.update(to_freeze)
