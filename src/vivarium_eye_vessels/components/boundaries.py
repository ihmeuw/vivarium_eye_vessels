from typing import List, Dict, Any
import numpy as np, pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

class EllipsoidContainment(Component):
    """Component that keeps particles within an ellipsoid boundary using magnetic repulsion.
    
    This component applies a repulsive force that increases as particles approach the
    ellipsoid boundary, preventing them from escaping the containment field.
    """
    
    CONFIGURATION_DEFAULTS = {
        'ellipsoid_containment': {
            'a': 1.0,  # Semi-major axis in x direction
            'b': 1.0,  # Semi-major axis in y direction
            'c': 1.0,  # Semi-major axis in z direction
            'repulsion_strength': 0.1,  # Strength of repulsion force
            'boundary_thickness': 0.1,  # Thickness of repulsion field
        }
    }

    @property
    def columns_required(self) -> List[str]:
        # We need position and velocity components from the particle simulation
        return ['x', 'y', 'z', 'vx', 'vy', 'vz', 'frozen']
    
    def setup(self, builder: Builder) -> None:
        """Setup the component."""
        self.config = builder.configuration.ellipsoid_containment
        
        # Get semi-major axes from config
        self.a = float(self.config.a)
        self.b = float(self.config.b)
        self.c = float(self.config.c)
        
        # Get other parameters
        self.repulsion_strength = float(self.config.repulsion_strength)
        self.boundary_thickness = float(self.config.boundary_thickness)
        
        # Register with time stepping system
        builder.event.register_listener('time_step', self.on_time_step)

    def calculate_ellipsoid_distance(self, x: float, y: float, z: float) -> float:
        """Calculate normalized distance from center of ellipsoid.
        
        A value > 1 means the point is outside the ellipsoid.
        A value = 1 means the point is on the surface.
        A value < 1 means the point is inside the ellipsoid.
        """
        return np.sqrt((x/self.a)**2 + (y/self.b)**2 + (z/self.c)**2)
    
    def calculate_repulsion_force(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Calculate the repulsive force vector at a given point."""
        # Get normalized distance from ellipsoid center
        d = self.calculate_ellipsoid_distance(x, y, z)
        
        # If particle is well within ellipsoid, no force needed
        if d < (1.0 - self.boundary_thickness):
            return 0.0, 0.0, 0.0
            
        # Calculate normalized direction vector from center
        dx = x/(self.a**2 * d)
        dy = y/(self.b**2 * d)
        dz = z/(self.c**2 * d)
        
        # Force increases as particles approach boundary
        force_magnitude = self.repulsion_strength * (d - (1.0 - self.boundary_thickness))
        
        # Return force vector components
        return (
            -force_magnitude * dx,
            -force_magnitude * dy, 
            -force_magnitude * dz
        )

    def on_time_step(self, event: Event) -> None:
        """Apply repulsion forces on each time step."""
        # Get current state of all unfrozen particles
        pop = self.population_view.get(event.index, query='frozen == False')
        if pop.empty:
            return
            
        # Calculate and apply forces for each particle
        forces = np.array([
            self.calculate_repulsion_force(row.x, row.y, row.z)
            for idx, row in pop.iterrows()
        ])
        
        # Update velocities based on forces
        dt = event.step_size / pd.Timedelta(days=1)
        pop['vx'] += forces[:, 0] * dt
        pop['vy'] += forces[:, 1] * dt  
        pop['vz'] += forces[:, 2] * dt
        
        # Update population
        self.population_view.update(pop)