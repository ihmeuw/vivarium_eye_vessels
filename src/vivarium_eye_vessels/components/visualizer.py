import pygame
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from collections import defaultdict


class ParticleVisualizer3D(Component):
    """An enhanced 3D visualizer for particles and their path connections with interactive controls."""

    CONFIGURATION_DEFAULTS = {
        'visualization': {
            'rotation_speed': 0.02,
            'projection_scale': 400,
            'background_color': (0, 0, 0),
            'particle_color': (255, 255, 255),
            'path_color': (100, 100, 255),
            'frozen_color': (255, 100, 100),
            'ellipsoid_color': (50, 150, 50),
            'base_path_width': 3,  # Base width for paths
            'progress_color': (0, 255, 0),
            'fps': 60,
            'screen_width': 800,
            'screen_height': 800,
            'particle_size': 3,
            'zoom_speed': 1.1,
            'ellipsoid_points': 20,
            'movement_speed': 0.05,  # Speed for WASD movement
            'manual_rotation_step': 0.05,  # Rotation step for arrow keys
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
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = pygame.display.get_surface().get_size()
        pygame.display.set_caption("3D Particle Path Visualization")

        # Initialize visualization parameters
        self.y_rotation = 0  # Rotation around y-axis (left/right)
        self.x_rotation = 0  # Rotation around x-axis (up/down)
        self.rotation_speed = self.config.rotation_speed
        self.projection_scale = self.config.projection_scale
        self.auto_rotate = True  # Flag for rotation control
        self.zoom_level = 1.0
        
        # Camera position and movement
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        
        # Initialize path branching cache
        self.path_branch_counts = None
        self.path_widths = None

        # Setup ellipsoid if present
        self._setup_ellipsoid(builder)

        # Get time information
        self.start_time = pd.Timestamp(**builder.configuration.time.start)
        self.end_time = pd.Timestamp(**builder.configuration.time.end)
        self.clock = builder.time.clock()

        # Initialize the game clock
        self.fps = self.config.fps
        self.pygame_clock = pygame.time.Clock()

    def _setup_ellipsoid(self, builder: Builder):
        """Setup ellipsoid parameters if component is present."""
        import pdb; pdb.set_trace()
        if 'ellipsoid_containment' in builder.components.list_components():
            try:
                self.ellipsoid_params = {
                    'a': float(builder.configuration.ellipsoid_containment.a),
                    'b': float(builder.configuration.ellipsoid_containment.b),
                    'c': float(builder.configuration.ellipsoid_containment.c),
                }
                self.has_ellipsoid = True
                self.ellipsoid_lines = self._generate_ellipsoid_wireframe()
            except AttributeError:
                self.has_ellipsoid = False
        else:
            self.has_ellipsoid = False

    def _calculate_path_branching(self, population: pd.DataFrame) -> None:
        """Calculate branching counts and path widths for all paths."""
        # Count children for each particle
        child_counts = defaultdict(int)
        for _, particle in population.iterrows():
            if pd.notna(particle.parent_id):
                child_counts[particle.parent_id] += 1

        # Calculate path widths based on branching
        self.path_widths = {}
        for idx in population.index:
            # Start with base width and reduce based on number of children
            width = self.config.base_path_width
            current_id = idx
            while pd.notna(current_id) and current_id in child_counts:
                if child_counts[current_id] > 1:
                    width = width / np.sqrt(child_counts[current_id])
                current_id = population.loc[current_id, 'parent_id']
            self.path_widths[idx] = max(1, width)  # Ensure minimum width of 1

    def _handle_input(self) -> bool:
        """Handle user input for interaction. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]
            ):
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.auto_rotate = not self.auto_rotate
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom_level *= self.config.zoom_speed
                elif event.key == pygame.K_MINUS:
                    self.zoom_level /= self.config.zoom_speed

        # Handle held keys
        keys = pygame.key.get_pressed()
        
        # Manual rotation when paused
        if not self.auto_rotate:
            if keys[pygame.K_LEFT]:
                self.y_rotation -= self.config.manual_rotation_step
            if keys[pygame.K_RIGHT]:
                self.y_rotation += self.config.manual_rotation_step
            if keys[pygame.K_UP]:
                self.x_rotation -= self.config.manual_rotation_step
            if keys[pygame.K_DOWN]:
                self.x_rotation += self.config.manual_rotation_step

        # WASD movement
        if keys[pygame.K_w]:
            self.camera_pos[1] -= self.config.movement_speed
        if keys[pygame.K_s]:
            self.camera_pos[1] += self.config.movement_speed
        if keys[pygame.K_a]:
            self.camera_pos[0] -= self.config.movement_speed
        if keys[pygame.K_d]:
            self.camera_pos[0] += self.config.movement_speed

        return True

    def _project_point(self, point: np.ndarray, rotation_matrix: np.ndarray) -> Tuple[int, int]:
        """Project a 3D point onto 2D screen space."""
        # Apply camera offset
        point = point - self.camera_pos

        # Scale and rotate
        scaled_point = point * 2 - 1
        rotated = np.dot(rotation_matrix, scaled_point)
        rotated[2] += 4

        # Apply zoom
        rotated[:2] *= self.zoom_level

        # Project to screen space
        if rotated[2] != 0:
            screen_x = int(self.width/2 + (self.projection_scale * rotated[0]) / rotated[2])
            screen_y = int(self.height/2 - (self.projection_scale * rotated[1]) / rotated[2])
        else:
            screen_x = int(self.width/2 + self.projection_scale * rotated[0])
            screen_y = int(self.height/2 - self.projection_scale * rotated[1])
        
        return screen_x, screen_y

    def _draw_connections(self, population: pd.DataFrame, rotation_matrix: np.ndarray) -> None:
        """Draw connections between particles with varying line thickness based on branching."""
        # Calculate path branching if not already done
        if self.path_widths is None:
            self._calculate_path_branching(population)

        # Create position lookup dictionary
        particle_positions = {
            idx: self._project_point(
                np.array([row.x, row.y, row.z]),
                rotation_matrix
            )
            for idx, row in population.iterrows()
        }

        # Draw connections
        for idx, particle in population.iterrows():
            if pd.notna(particle.parent_id) and particle.parent_id in particle_positions:
                start_pos = particle_positions[particle.parent_id]
                end_pos = particle_positions[idx]
                width = int(self.path_widths[idx])

                pygame.draw.line(
                    self.screen,
                    self.config.path_color,
                    start_pos,
                    end_pos,
                    width
                )

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

    def on_time_step(self, event: Event) -> None:
        """Update visualization on each time step."""
        population = self.population_view.get(event.index)
        if population.empty:
            return

        # Handle input
        if not self._handle_input():
            pygame.quit()
            return

        # Clear screen
        self.screen.fill(self.config.background_color)

        # Update rotation if auto-rotate is enabled
        if self.auto_rotate:
            self.y_rotation += self.rotation_speed
        
        # Keep angles in range [0, 2π]
        self.y_rotation %= 2 * np.pi
        self.x_rotation %= 2 * np.pi

        # Create rotation matrices for both axes
        # First rotate around y-axis (left/right)
        y_rotation = np.array([
            [np.cos(self.y_rotation), 0, np.sin(self.y_rotation)],
            [0, 1, 0],
            [-np.sin(self.y_rotation), 0, np.cos(self.y_rotation)],
        ])
        
        # Then rotate around x-axis (up/down)
        x_rotation = np.array([
            [1, 0, 0],
            [0, np.cos(self.x_rotation), -np.sin(self.x_rotation)],
            [0, np.sin(self.x_rotation), np.cos(self.x_rotation)],
        ])
        
        # Combine rotations (order matters - we apply y rotation first, then x)
        rotation_matrix = np.dot(x_rotation, y_rotation)

        # Draw visualization elements
        if self.has_ellipsoid:
            self._draw_ellipsoid(rotation_matrix)
        self._draw_connections(population, rotation_matrix)
        self._draw_particles(population, rotation_matrix)
        self._draw_progress_bar()
        self._draw_fps()
        self._draw_controls_help()

        pygame.display.flip()
        self.pygame_clock.tick(self.fps)

    def _draw_controls_help(self) -> None:
        """Draw help text for controls."""
        font = pygame.font.Font(None, 24)
        controls = [
            "Controls:",
            "Space: Pause/Resume rotation",
            "Left/Right: Rotate horizontally when paused",
            "Up/Down: Rotate vertically when paused",
            "WASD: Move viewpoint",
            "+/-: Zoom in/out",
            "ESC/Q: Quit"
        ]
        
        for i, text in enumerate(controls):
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (10, 40 + i * 25))

    def _draw_progress_bar(self) -> None:
        """Draw a progress bar at the top of the screen."""
        current_time = self.clock()
        progress = (current_time - self.start_time) / (self.end_time - self.start_time)

        bar_height = 3
        bar_width = int(self.width * progress)
        progress_rect = pygame.Rect(0, 0, bar_width, bar_height)
        pygame.draw.rect(self.screen, (0, 255, 0), progress_rect)

    def _draw_fps(self) -> None:
        """Draw the current FPS in the corner of the screen."""
        font = pygame.font.Font(None, 24)
        fps_text = font.render(
            f"FPS: {int(self.pygame_clock.get_fps())}", True, (200, 200, 200)
        )
        self.screen.blit(fps_text, (10, 10))
        
    def _generate_ellipsoid_wireframe(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate points for ellipsoid wireframe visualization."""
        n_points = self.config.ellipsoid_points
        
        # Generate parametric angles
        theta = np.linspace(0, 2*np.pi, n_points)
        phi = np.linspace(-np.pi/2, np.pi/2, n_points//2)
        
        lines = []
        
        # Generate longitude lines
        for t in theta:
            curve = []
            for p in phi:
                x = self.ellipsoid_params['a'] * np.cos(p) * np.cos(t)
                y = self.ellipsoid_params['b'] * np.cos(p) * np.sin(t)
                z = self.ellipsoid_params['c'] * np.sin(p)
                curve.append(np.array([x, y, z]))
            for i in range(len(curve)-1):
                lines.append((curve[i], curve[i+1]))
                
        # Generate latitude lines
        for p in phi:
            curve = []
            for t in theta:
                x = self.ellipsoid_params['a'] * np.cos(p) * np.cos(t)
                y = self.ellipsoid_params['b'] * np.cos(p) * np.sin(t)
                z = self.ellipsoid_params['c'] * np.sin(p)
                curve.append(np.array([x, y, z]))
            curve.append(curve[0])  # Close the loop
            for i in range(len(curve)-1):
                lines.append((curve[i], curve[i+1]))
                
        return lines

    def _draw_ellipsoid(self, rotation_matrix: np.ndarray) -> None:
        """Draw the ellipsoid wireframe."""
        if not self.has_ellipsoid:
            return
            
        for start, end in self.ellipsoid_lines:
            start_screen = self._project_point(start, rotation_matrix)
            end_screen = self._project_point(end, rotation_matrix)
            
            pygame.draw.line(
                self.screen,
                self.config.ellipsoid_color,
                start_screen,
                end_screen,
                1  # Width of ellipsoid lines
            )
