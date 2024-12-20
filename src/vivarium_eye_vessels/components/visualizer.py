import pygame
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class ParticleVisualizer3D(Component):
    """A 3D visualizer for particles and their path connections."""
    
    CONFIGURATION_DEFAULTS = {
        'visualization': {
            'rotation_speed': 0.02,
            'projection_scale': 400,
            'background_color': (0, 0, 0),
            'particle_color': (255, 255, 255),
            'path_color': (100, 100, 255),  # Color for connection lines
            'frozen_color': (255, 100, 100),  # Color for frozen particles
            'path_width': 1,  # Width of connection lines
            'progress_color': (0, 255, 0),
            'fps': 60,
            'screen_width': 800,
            'screen_height': 800,
            'particle_size': 3,  # Size of particle dots
        }
    }

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "z", "frozen", "parent_id", "path_id"]

    def setup(self, builder: Builder):
        """Initialize the visualization component."""
        pygame.init()

        # Read configuration
        self.config = builder.configuration.visualization
        
        # Setup display
        self.width = self.config.screen_width
        self.height = self.config.screen_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3D Particle Path Visualization")
        
        # Initialize visualization parameters
        self.rotation_angle = 0
        self.rotation_speed = self.config.rotation_speed
        self.projection_scale = self.config.projection_scale
        
        # Get time information
        self.clock = builder.time.clock()
        self.start_time = builder.configuration.time.start
        self.end_time = builder.configuration.time.end
        
        # Initialize the game clock
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

        # Draw connections first so they appear behind particles
        self._draw_connections(population, rotation_matrix)
        
        # Then draw particles
        self._draw_particles(population, rotation_matrix)
        
        # Draw UI elements
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
        for idx, particle in population.iterrows():
            position = np.array([particle.x, particle.y, particle.z])
            screen_pos = self._project_point(position, rotation_matrix)
            
            # Choose color based on frozen state
            color = self.config.frozen_color if particle.frozen else self.config.particle_color
            
            # Draw particle
            pygame.draw.circle(
                self.screen,
                color,
                screen_pos,
                self.config.particle_size
            )

    def _draw_connections(self, population: pd.DataFrame, rotation_matrix: np.ndarray) -> None:
        """Draw connections between particles based on parent_id links."""
        # Debug output
        has_parent = population[pd.notna(population.parent_id)]
        if False:
            print(f"\nTimestep connection debug:")
            print(f"Total particles: {len(population)}")
            print(f"Particles with parents: {len(has_parent)}")
            if not has_parent.empty:
                print("Sample of parent relationships:")
                sample = has_parent.head(3)
                for idx, particle in sample.iterrows():
                    print(f"Particle {idx} has parent {particle.parent_id}")
                    print(f"Parent exists in population: {particle.parent_id in population.index}")
                    if particle.parent_id in population.index:
                        parent = population.loc[particle.parent_id]
                        print(f"Parent position: ({parent.x:.2f}, {parent.y:.2f}, {parent.z:.2f})")
                        print(f"Child position: ({particle.x:.2f}, {particle.y:.2f}, {particle.z:.2f})")

        # Create a dictionary for quick lookup of particle positions
        particle_positions = {
            idx: self._project_point(
                np.array([row.x, row.y, row.z]), 
                rotation_matrix
            )
            for idx, row in population.iterrows()
        }
        
        # Draw connections for particles with valid parent_ids
        connections_drawn = 0
        for idx, particle in population.iterrows():
            if pd.notna(particle.parent_id) and particle.parent_id in particle_positions:
                start_pos = particle_positions[particle.parent_id]
                end_pos = particle_positions[idx]
                
                pygame.draw.line(
                    self.screen,
                    self.config.path_color,
                    start_pos,
                    end_pos,
                    self.config.path_width
                )
                connections_drawn += 1
        
        # if connections_drawn > 0:
        #     print(f"Drew {connections_drawn} connections")

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