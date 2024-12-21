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
            'screen_width': 0,      # Set to 0 for fullscreen
            'screen_height': 0,     # Set to 0 for fullscreen
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

        # Determine screen mode
        screen_width = self.config.get('screen_width', 0)
        screen_height = self.config.get('screen_height', 0)
        if screen_width == 0 and screen_height == 0:
            # Fullscreen mode
            self.screen = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN
            )
            self.width, self.height = self.screen.get_size()
        else:
            # Windowed mode with specified dimensions
            self.screen = pygame.display.set_mode(
                (screen_width, screen_height)
            )
            self.width, self.height = self.screen.get_size()

        pygame.display.set_caption("3D Particle Path Visualization")

        # Initialize visualization parameters
        self.y_rotation = 0.0  # Rotation around y-axis (left/right)
        self.x_rotation = 0.0  # Rotation around x-axis (up/down)
        self.z_rotation = 0.0  # Rotation around z-axis (roll)
        self.rotation_speed = self.config['rotation_speed']
        self.projection_scale = self.config['projection_scale']
        self.auto_rotate = False  # Flag for rotation control
        self.zoom_level = 1.0

        # Camera position and movement
        self.camera_pos = np.array([0.0, 0.0, 0.0])

        # Initialize path branching cache
        self.path_widths = None

        # Setup ellipsoid if present
        self._setup_ellipsoid(builder)

        # Get time information
        self.start_time = pd.Timestamp(**builder.configuration.time.start)
        self.end_time = pd.Timestamp(**builder.configuration.time.end)
        self.clock = builder.time.clock()

        # Initialize the game clock
        self.fps = self.config['fps']
        self.pygame_clock = pygame.time.Clock()

        # Pre-render control help
        self._pre_render_controls()

        # Create reusable surfaces for particles and connections
        self.particle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.connection_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

    def _setup_ellipsoid(self, builder: Builder):
        """Setup ellipsoid parameters if component is present."""
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

    def _pre_render_controls(self):
        """Pre-render control help text."""
        font = pygame.font.Font(None, 24)
        controls = [
            "Controls:",
            "Space: Pause/Resume rotation",
            "Left/Right: Rotate horizontally when paused",
            "Up/Down: Rotate vertically when paused",
            "Z/X: Rotate around z-axis when paused",
            "WASD: Move viewpoint",
            "+/-: Zoom in/out",
            "ESC/Q: Quit"
        ]
        self.control_surfaces = []
        for i, text in enumerate(controls):
            surface = font.render(text, True, (200, 200, 200))
            self.control_surfaces.append((surface, (10, 40 + i * 25)))

    def _generate_ellipsoid_wireframe(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate lines for the ellipsoid wireframe."""
        lines = []
        num_points = self.config['ellipsoid_points']

        # Generate points for the longitudinal (vertical) circles
        for i in range(num_points):
            phi = 2 * np.pi * i / num_points
            for j in range(num_points - 1):
                theta1 = np.pi * j / num_points
                theta2 = np.pi * (j + 1) / num_points

                x1 = self.ellipsoid_params['a'] * np.sin(theta1) * np.cos(phi)
                y1 = self.ellipsoid_params['b'] * np.sin(theta1) * np.sin(phi)
                z1 = self.ellipsoid_params['c'] * np.cos(theta1)

                x2 = self.ellipsoid_params['a'] * np.sin(theta2) * np.cos(phi)
                y2 = self.ellipsoid_params['b'] * np.sin(theta2) * np.sin(phi)
                z2 = self.ellipsoid_params['c'] * np.cos(theta2)

                lines.append((np.array([x1, y1, z1]), np.array([x2, y2, z2])))

        # Generate points for the latitudinal (horizontal) circles
        for j in range(num_points):
            theta = np.pi * j / num_points
            for i in range(num_points - 1):
                phi1 = 2 * np.pi * i / num_points
                phi2 = 2 * np.pi * (i + 1) / num_points

                x1 = self.ellipsoid_params['a'] * np.sin(theta) * np.cos(phi1)
                y1 = self.ellipsoid_params['b'] * np.sin(theta) * np.sin(phi1)
                z1 = self.ellipsoid_params['c'] * np.cos(theta)

                x2 = self.ellipsoid_params['a'] * np.sin(theta) * np.cos(phi2)
                y2 = self.ellipsoid_params['b'] * np.sin(theta) * np.sin(phi2)
                z2 = self.ellipsoid_params['c'] * np.cos(theta)

                lines.append((np.array([x1, y1, z1]), np.array([x2, y2, z2])))

        return lines

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
                elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                    self.zoom_level *= self.config['zoom_speed']
                elif event.key == pygame.K_MINUS:
                    self.zoom_level /= self.config['zoom_speed']

        # Handle held keys
        keys = pygame.key.get_pressed()

        # Manual rotation when paused
        if not self.auto_rotate:
            if keys[pygame.K_LEFT]:
                self.y_rotation -= self.config['manual_rotation_step']
            if keys[pygame.K_RIGHT]:
                self.y_rotation += self.config['manual_rotation_step']
            if keys[pygame.K_UP]:
                self.x_rotation -= self.config['manual_rotation_step']
            if keys[pygame.K_DOWN]:
                self.x_rotation += self.config['manual_rotation_step']
            if keys[pygame.K_z]:  # Added z rotation
                self.z_rotation -= self.config['manual_rotation_step']
            if keys[pygame.K_x]:  # Added x rotation
                self.z_rotation += self.config['manual_rotation_step']

        # WASD movement for x and y axes
        if keys[pygame.K_w]:
            self.camera_pos[1] -= self.config['movement_speed']
        if keys[pygame.K_s]:
            self.camera_pos[1] += self.config['movement_speed']
        if keys[pygame.K_a]:
            self.camera_pos[0] -= self.config['movement_speed']
        if keys[pygame.K_d]:
            self.camera_pos[0] += self.config['movement_speed']

        # R and F keys for z-axis movement
        if keys[pygame.K_r]:
            self.camera_pos[2] += self.config['movement_speed']  # Move up
        if keys[pygame.K_f]:
            self.camera_pos[2] -= self.config['movement_speed']  # Move down

        return True

    def _project_point(self, point: np.ndarray, rotation_matrix: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project a single 3D point onto 2D screen space.

        Returns:
            Tuple of (x, y) screen coordinates if the point is in front of the camera,
            otherwise None.
        """
        # Apply camera offset
        translated_point = point - self.camera_pos

        # Rotate points
        rotated = translated_point @ rotation_matrix.T
        rotated[2] += 4  # Adjust z-axis to ensure positive depth

        # Apply zoom
        rotated[:2] *= self.zoom_level

        # Project to screen space
        if rotated[2] > 0:
            screen_x = int(self.width / 2 + (self.projection_scale * rotated[0]) / rotated[2])
            screen_y = int(self.height / 2 - (self.projection_scale * rotated[1]) / rotated[2])
            return screen_x, screen_y
        else:
            # Point is behind the camera; do not render
            return None

    def _project_points(self, points: np.ndarray, rotation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project multiple 3D points onto 2D screen space using vectorized operations.

        Returns:
            Tuple containing:
                - screen_points: Nx2 array of screen coordinates
                - mask: N boolean array indicating if the point is in front of the camera
        """
        # Apply camera offset
        translated_points = points - self.camera_pos

        # Rotate points
        rotated = translated_points @ rotation_matrix.T
        rotated[:, 2] += 4  # Adjust z-axis to ensure positive depth

        # Apply zoom
        rotated[:, :2] *= self.zoom_level

        # Create mask for points in front of the camera
        mask = rotated[:, 2] > 0

        # Avoid division by zero
        z = rotated[:, 2].reshape(-1, 1)
        z[z == 0] = 1e-6

        # Project to screen space
        screen_x = (self.width / 2 + (self.projection_scale * rotated[:, 0]) / z[:, 0]).astype(int)
        screen_y = (self.height / 2 - (self.projection_scale * rotated[:, 1]) / z[:, 0]).astype(int)

        # Stack coordinates
        screen_points = np.stack((screen_x, screen_y), axis=-1)

        return screen_points, mask

    def _draw_controls_help(self) -> None:
        """Blit pre-rendered help text onto the screen."""
        for surface, pos in self.control_surfaces:
            self.screen.blit(surface, pos)

    def _calculate_path_branching(self, population: pd.DataFrame) -> None:
        """Calculate branching counts and path widths for all paths using optimized methods."""
        # Drop NaN parent_ids and convert to integers
        parent_ids = population['parent_id'].dropna().astype(int)

        # Count the number of children for each parent_id
        child_counts = parent_ids.value_counts()

        # Map child counts to each particle's parent_id
        counts = population['parent_id'].map(child_counts).fillna(1)

        # Calculate path widths: base_width divided by sqrt(child_counts)
        path_widths = self.config['base_path_width'] / np.sqrt(counts)

        # Ensure minimum width of 1 and convert to integers
        path_widths = path_widths.clip(lower=1).astype(int).values

        self.path_widths = path_widths

    def _draw_connections(self, population: pd.DataFrame, screen_points: np.ndarray, mask: np.ndarray, surface: pygame.Surface) -> None:
        """Draw connections between particles with varying line thickness based on branching."""
        if self.path_widths is None:
            self._calculate_path_branching(population)

        parent_ids = population['parent_id'].values

        for idx, parent_id in enumerate(parent_ids):
            if pd.notna(parent_id):
                try:
                    parent_index = population.index.get_loc(parent_id)
                except KeyError:
                    # Parent not found in the current population
                    continue

                # Check if both parent and child are in front of the camera
                if mask[parent_index] and mask[idx]:
                    start_pos = screen_points[parent_index]
                    end_pos = screen_points[idx]
                    width = self.path_widths[idx]
                    pygame.draw.line(surface, self.config['path_color'], start_pos, end_pos, width)

    def _draw_particles(self, screen_points: np.ndarray, colors: np.ndarray, mask: np.ndarray, surface: pygame.Surface) -> None:
        """Draw all particles onto a given surface."""
        # Filter points and colors based on mask
        visible_points = screen_points[mask]
        visible_colors = colors[mask]

        for pos, color in zip(visible_points, visible_colors):
            pygame.draw.circle(surface, color, pos, self.config['particle_size'])

    def _draw_ellipsoid(self, rotation_matrix: np.ndarray) -> None:
        """Draw the ellipsoid wireframe with current rotation."""
        for line in self.ellipsoid_lines:
            # Project both points with the rotation_matrix
            start_pos = self._project_point(line[0], rotation_matrix)
            end_pos = self._project_point(line[1], rotation_matrix)

            # Only draw if both points are in front of the camera
            if start_pos and end_pos:
                pygame.draw.line(self.screen, self.config['ellipsoid_color'], start_pos, end_pos, 1)

    def _draw_progress_bar(self) -> None:
        """Draw a progress bar at the top of the screen."""
        current_time = self.clock()
        total_duration = (self.end_time - self.start_time).total_seconds()
        if total_duration == 0:
            progress = 0.0
        else:
            progress = (current_time - self.start_time).total_seconds() / total_duration
            progress = np.clip(progress, 0.0, 1.0)

        bar_height = 3
        bar_width = int(self.width * progress)
        progress_rect = pygame.Rect(0, 0, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.config['progress_color'], progress_rect)

    def _draw_fps(self) -> None:
        """Draw the current FPS in the corner of the screen."""
        font = pygame.font.Font(None, 24)
        fps_text = font.render(
            f"FPS: {int(self.pygame_clock.get_fps())}", True, (200, 200, 200)
        )
        self.screen.blit(fps_text, (10, 10))

    def on_time_step(self, event: Event) -> None:
        """Update visualization on each time step."""
        population = self.population_view.get(event.index)
        if population.empty:
            return

        # Handle input
        if not self._handle_input():
            pygame.quit()
            return

        # Clear main screen
        self.screen.fill(self.config['background_color'])

        # Update rotation if auto-rotate is enabled
        if self.auto_rotate:
            self.y_rotation += self.rotation_speed

        # Keep angles in range [0, 2π]
        self.y_rotation %= 2 * np.pi
        self.x_rotation %= 2 * np.pi
        self.z_rotation %= 2 * np.pi

        # Create rotation matrices for all axes
        cy, sy = np.cos(self.y_rotation), np.sin(self.y_rotation)
        cx, sx = np.cos(self.x_rotation), np.sin(self.x_rotation)
        cz, sz = np.cos(self.z_rotation), np.sin(self.z_rotation)

        # Rotation matrices
        y_rotation_matrix = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])

        x_rotation_matrix = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])

        z_rotation_matrix = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])

        # Combine rotations (order: y, then x, then z)
        rotation_matrix = z_rotation_matrix @ x_rotation_matrix @ y_rotation_matrix

        # Prepare points array
        points = population[['x', 'y', 'z']].values

        # Project all points at once
        screen_points, mask = self._project_points(points, rotation_matrix)

        # Determine colors based on frozen state
        frozen_mask = population['frozen'].values
        colors = np.where(frozen_mask[:, np.newaxis],
                          self.config['frozen_color'],
                          self.config['particle_color'])

        # Clear reusable surfaces
        self.connection_surface.fill((0, 0, 0, 0))  # Transparent fill
        self.particle_surface.fill((0, 0, 0, 0))    # Transparent fill

        # Draw connections and particles on their respective surfaces
        self._draw_connections(population, screen_points, mask, self.connection_surface)
        self._draw_particles(screen_points, colors, mask, self.particle_surface)

        # Blit surfaces onto the main screen
        self.screen.blit(self.connection_surface, (0, 0))
        self.screen.blit(self.particle_surface, (0, 0))

        # Draw ellipsoid if present
        if self.has_ellipsoid:
            self._draw_ellipsoid(rotation_matrix)

        # Draw progress bar, FPS, and controls help
        self._draw_progress_bar()
        self._draw_fps()
        self._draw_controls_help()

        pygame.display.flip()
        self.pygame_clock.tick(self.fps)

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        try:
            pygame.quit()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
