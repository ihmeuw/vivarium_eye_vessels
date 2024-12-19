from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class ParticlePaths3D(Component):
    """A component that simulates particles moving in 3D space and freezing their paths."""
    
    name = "particle_paths_3d"

    @property
    def columns_created(self) -> List[str]:
        return ["x", "y", "z", "vx", "vy", "vz", "frozen", "parent_id", "path_id", "creation_time"]

    CONFIGURATION_DEFAULTS = {
        "particles": {
            "step_size": 0.01,
            "overall_max_velocity_change": 0.1,
            "initial_velocity_range": (-0.05, 0.05),
            "particles_per_year": 1000,  # Number of new particles to add per year
            "max_active_particles": 1000,  # Maximum number of moving particles
            "freeze_interval": 10,  # Steps between freezing particle positions
        }
    }

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.particles
        self.step_size = self.config.step_size
        self.overall_max_velocity_change = self.config.overall_max_velocity_change
        self.initial_velocity_range = self.config.initial_velocity_range
        
        self.clock = builder.time.clock()
        self.step_count = 0
        self.next_path_id = 0

        # Setup for adding new simulants
        self.fractional_new_particles = 0
        self.simulant_creator = builder.population.get_simulant_creator()

        # Register the max velocity change pipeline
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
        pop["parent_id"] = pd.NA  # Will store the index of the previous point in path
        pop["path_id"] = list(range(self.next_path_id, self.next_path_id + len(pop)))
        self.next_path_id += len(pop)
        pop["creation_time"] = self.clock()

        self.population_view.update(pop)

    def on_time_step(self, event: Event) -> None:
        """Update particle positions, freeze paths, and create new particles."""
        self.step_count += 1
        
        # Get current population state
        pop = self.population_view.get(event.index)
        active_particles = pop[~pop.frozen]
        
        # Update positions of active particles
        if not active_particles.empty:
            self.update_positions(active_particles)

        # Freeze particles on interval
        if self.step_count % self.config.freeze_interval == 0:
            self.freeze_particles(active_particles)

        # Add new particles if below max
        self.add_new_particles(event)

    def update_positions(self, particles: pd.DataFrame) -> None:
        """Update positions and velocities of active particles."""
        # Update positions based on current velocities
        for pos, vel in [('x','vx'), ('y','vy'), ('z','vz')]:
            particles.loc[:,pos] = (particles[pos] + self.step_size * particles[vel]) % 1.0

        # Get max velocity change from pipeline
        max_velocity = self.max_velocity_change(particles.index)

        # Update velocities with random changes
        for v in ['vx', 'vy', 'vz']:
            dv = (self.randomness.get_draw(particles.index, additional_key=f'd{v}') - 0.5) * \
                 2 * max_velocity
            particles.loc[:,v] += dv

        self.population_view.update(particles)

    def freeze_particles(self, active_particles: pd.DataFrame) -> None:
        """Create frozen path points and add them to population."""
        if active_particles.empty:
            return
            
        # Get previous frozen points for each path
        all_pop = self.population_view.get(self.population_view._manager.population.index)  # Get full population
        frozen_points = []
        
        for idx, particle in active_particles.iterrows():
            # Look for the most recent frozen point in this path
            path_points = all_pop[all_pop.path_id == particle.path_id]
            frozen_path_points = path_points[path_points.frozen]
            
            new_point = particle.copy()
            new_point['frozen'] = True
            
            if not frozen_path_points.empty:
                # Link to most recently frozen point in this path
                last_frozen = frozen_path_points.index[-1]
                new_point['parent_id'] = last_frozen
            else:
                # First frozen point in path
                new_point['parent_id'] = pd.NA
                
            frozen_points.append(new_point)
            
        if frozen_points:
            frozen_df = pd.DataFrame(frozen_points)
            self.population_view.update(frozen_df)

    def add_new_particles(self, event: Event) -> None:
        """Add new particles based on yearly rate."""
        pop = self.population_view.get(event.index)
        active_count = len(pop[~pop.frozen])
        
        if active_count >= self.config.max_active_particles:
            return
            
        # Calculate number of particles to add this step
        step_size = event.step_size / pd.Timedelta(days=365)  # Convert to years
        particles_to_add = self.config.particles_per_year * step_size + self.fractional_new_particles
        self.fractional_new_particles = particles_to_add % 1
        particles_to_add = int(particles_to_add)

        # Cap addition to maintain max active particles
        particles_to_add = min(particles_to_add, 
                             self.config.max_active_particles - active_count)

        if particles_to_add > 0:
            self.simulant_creator(
                particles_to_add,
                {
                    "sim_state": "time_step",
                },
            )