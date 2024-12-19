from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class Basic3D(Component):
    name = "basis_3d_particles"

    @property
    def columns_created(self):
        return ["x", "y", "z", "vx", "vy", "vz"]

    CONFIGURATION_DEFAULTS = {
        "particles": {
            "step_size": 0.01,
            "overall_max_velocity_change": 0.1,
            "initial_velocity_range": (-0.05, 0.05),
        }
    }

    def setup(self, builder):
        config = builder.configuration.particles
        self.step_size = config.step_size
        self.overall_max_velocity_change = config.overall_max_velocity_change
        self.initial_velocity_range = config.initial_velocity_range

        # Register the max velocity change pipeline
        self.max_velocity_change = builder.value.register_value_producer(
            "particle.max_velocity_change",
            source=lambda index: pd.Series(
                self.overall_max_velocity_change, index=index
            ),
        )

        self.randomness = builder.randomness.get_stream("particle.particles_3d")

    def on_initialize_simulants(self, simulant_data):
        """Start new simulants at random location in unit cube,
        with random velocities within specified range."""
        pop = pd.DataFrame(index=simulant_data.index)

        # Generate random x, y, z coordinates in [0,1) x [0,1) x [0,1)
        pop["x"] = self.randomness.get_draw(pop.index, additional_key="x")
        pop["y"] = self.randomness.get_draw(pop.index, additional_key="y")
        pop["z"] = self.randomness.get_draw(pop.index, additional_key="z")

        # Generate random initial velocities within the specified range
        velocity_range = self.initial_velocity_range
        pop["vx"] = self.randomness.get_draw(pop.index, additional_key="vx") * (
            velocity_range[1] - velocity_range[0]
        ) + velocity_range[0]
        pop["vy"] = self.randomness.get_draw(pop.index, additional_key="vy") * (
            velocity_range[1] - velocity_range[0]
        ) + velocity_range[0]
        pop["vz"] = self.randomness.get_draw(pop.index, additional_key="vz") * (
            velocity_range[1] - velocity_range[0]
        ) + velocity_range[0]

        self.population_view.update(pop)

    def on_time_step(self, event):
        """Update particle positions and velocities."""
        pop = self.population_view.get(event.index)

        # Update positions based on current velocities
        pop["x"] = (pop["x"] + self.step_size * pop["vx"]) % 1.0
        pop["y"] = (pop["y"] + self.step_size * pop["vy"]) % 1.0
        pop["z"] = (pop["z"] + self.step_size * pop["vz"]) % 1.0

        # Get max velocity change from pipeline
        max_velocity = self.max_velocity_change(pop.index)

        # Update velocities with random changes
        dvx = (
            self.randomness.get_draw(pop.index, additional_key="dvx") - 0.5
        ) * 2 * max_velocity
        dvy = (
            self.randomness.get_draw(pop.index, additional_key="dvy") - 0.5
        ) * 2 * max_velocity
        dvz = (
            self.randomness.get_draw(pop.index, additional_key="dvz") - 0.5
        ) * 2 * max_velocity

        pop["vx"] = pop["vx"] + dvx
        pop["vy"] = pop["vy"] + dvy
        pop["vz"] = pop["vz"] + dvz

        self.population_view.update(pop)
