import pygame
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class ParticleVisualizer3D(Component):
    """A simple 3D visualizer for particles."""
    
    CONFIGURATION_DEFAULTS = {
        'visualization': {
            'rotation_speed': 0.02,
            'projection_scale': 400,
            'background_color': (0, 0, 0),
            'particle_color': (255, 255, 255),
            'progress_color': (0, 255, 0),
            'fps': 60,
            'screen_width': 800,
            'screen_height': 800
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z"]

    def setup(self, builder: Builder):
        """Initialize the visualization component."""
        pygame.init()

        # Read configuration
        self.config = builder.configuration.visualization
        
        # Setup display
        self.width = self.config.screen_width
        self.height = self.config.screen_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3D Particle Visualization")
        
        # Initialize visualization parameters
        self.rotation_angle = 0
        self.rotation_speed = self.config.rotation_speed
        self.projection_scale = self.config.projection_scale
        
        # Get time information directly from builder
        self.clock = builder.time.clock()
        self.start_time = builder.configuration.time.start
        self.end_time = builder.configuration.time.end
        
        # Initialize the game clock for FPS control
        self.fps = self.config.fps
        self.pygame_clock = pygame.time.Clock()

    def on_time_step(self, event: Event) -> None:
        """Update visualization on each time step."""
        population = self.population_view.get(event.index)
        if population.empty:
            return

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]
            ):
                pygame.quit()
                return

        # Clear screen
        self.screen.fill(self.config.background_color)

        # Update rotation
        self.rotation_angle += self.rotation_speed
        self.rotation_angle %= 2 * np.pi

        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(self.rotation_angle), 0, np.sin(self.rotation_angle)],
            [0, 1, 0],
            [-np.sin(self.rotation_angle), 0, np.cos(self.rotation_angle)],
        ])

        # Draw particles
        self._draw_particles(population, rotation_matrix)
        
        # Draw progress bar and UI elements
        self._draw_progress_bar()
        self._draw_fps()

        pygame.display.flip()
        self.pygame_clock.tick(self.fps)

    def _project_point(self, point: np.ndarray, rotation_matrix: np.ndarray) -> Tuple[int, int]:
        """Project a 3D point onto 2D screen space."""
        # Scale the point to make the unit cube larger
        scaled_point = point * 2 - 1  # Convert from [0,1] to [-1,1] range
        
        # Apply rotation and move camera back to see full cube
        rotated = np.dot(rotation_matrix, scaled_point)
        rotated[2] += 4  # Move further back from camera
        
        # Project to screen space with adjusted scale
        if rotated[2] != 0:  # Prevent division by zero
            screen_x = int(self.width/2 + (self.projection_scale * rotated[0]) / rotated[2])
            screen_y = int(self.height/2 - (self.projection_scale * rotated[1]) / rotated[2])
        else:
            screen_x = int(self.width/2 + self.projection_scale * rotated[0])
            screen_y = int(self.height/2 - self.projection_scale * rotated[1])
        return screen_x, screen_y

    def _draw_particles(self, population: pd.DataFrame, rotation_matrix: np.ndarray) -> None:
        """Draw all particles."""
        for _, particle in population.iterrows():
            position = np.array([particle.x, particle.y, particle.z])
            screen_pos = self._project_point(position, rotation_matrix)
            
            # Draw particle
            pygame.draw.circle(
                self.screen,
                self.config.particle_color,
                screen_pos,
                5
            )

    def _draw_progress_bar(self) -> None:
        """Draw the simulation progress bar."""
        current_time = self.clock()
        try:
            progress = (current_time - self.start_time) / (self.end_time - self.start_time)
            progress = float(progress)  # Ensure it's a float
        except TypeError:
            progress = 0.0  # Default to 0 if there's an error
            
        bar_height = 5
        bar_width = int(self.width * progress)
        progress_rect = pygame.Rect(0, 0, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.config.progress_color, progress_rect)

    def _draw_fps(self) -> None:
        """Draw the current FPS counter."""
        font = pygame.font.Font(None, 36)
        fps_text = font.render(f"FPS: {int(self.pygame_clock.get_fps())}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        try:
            pygame.quit()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")