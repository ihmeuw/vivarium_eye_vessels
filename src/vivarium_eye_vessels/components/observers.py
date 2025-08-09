import numpy as np
import pandas as pd
from vivarium import Component


class SaveParticles(Component):
    name = "SaveImage"

    @property
    def columns_required(self):
        return ["x", "y", "z", "parent_id", "frozen", "depth"]

    def setup(self, builder):
        self.seed = builder.configuration.randomness.random_seed

    def on_simulation_end(self, event):
        pop = self.population_view.get(event.index)
        fname = f"{self.seed}.csv.bz2"
        pop.to_csv(fname)
