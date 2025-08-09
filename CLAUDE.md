# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Vivarium simulation model that creates synthetic data resembling the vascular system of the human eye. The simulation uses agent-based modeling with custom components for vessel growth, splitting, and collision avoidance. Components work together to simulate blood vessel development in 3D space using particle systems with physics-based interactions.

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/             # Run tests in tests directory  
pytest tests/test_sample.py  # Run specific test file
```

### Code Formatting and Linting
```bash
black .                   # Format code (line length: 94)
isort .                   # Sort imports (black profile)
black --check .           # Check formatting without changes
isort --check .           # Check import sorting without changes
```

### Running Simulations
```bash
simulate run src/vivarium_eye_vessels/model_specifications/model_spec.yaml
simulate run -v src/vivarium_eye_vessels/model_specifications/model_spec.yaml  # Verbose logging
```

### Artifact Generation
This sim does not use artifacts, actually

## Architecture

### Core Components (src/vivarium_eye_vessels/components/)

- **particles.py**: Core particle system with Particle3D base component, PathFreezer, PathSplitter, PathExtinction, and PathDLA for vessel growth dynamics
- **boundaries.py**: Force-based boundary conditions including EllipsoidContainment, CylinderExclusion, PointRepulsion, and FrozenRepulsion using Hookean or magnetic force calculations
- **visualizer.py**: 3D visualization using pygame for real-time particle rendering
- **observers.py**: Data collection and output management

### Key Vivarium Patterns

1. **Component Architecture**: All components inherit from `vivarium.Component` with standard lifecycle methods (`setup()`, `columns_created`, `columns_required`)

2. **Configuration Structure**: Model specifications in YAML format define component parameters, with `CONFIGURATION_DEFAULTS` in each component class

3. **Population Management**: Particles are treated as simulants with tabular data (position, velocity, frozen state, parent relationships)

4. **Event-Driven Updates**: Components respond to time step events to update particle states and apply forces

5. **Builder Pattern**: Use `Builder` object in `setup()` methods to register value sources, event listeners, and population views

### Data Structure

- **Particle Columns**: x, y, z (position), vx, vy, vz (velocity), frozen, freeze_time, unfreeze_time, depth, parent_id, path_id
- **Force Types**: Hookean (spring-based) and magnetic (inverse square) force calculations
- **Tree Structure**: Vessels maintain parent-child relationships for branching patterns

### Configuration Management

Model specifications use nested YAML with component-specific parameter sections. Critical parameters include:
- Particle dynamics (velocity limits, force thresholds)  
- Boundary constraints (ellipsoid/cylinder geometry, spring constants)
- Vessel growth (split intervals, angles, probabilities)
- Visualization settings (colors, projection scale, frame rate)

### Code Style

- Python 3.10/3.11 support
- Type hints required for all functions and methods
- Black formatting (94 character line length)  
- Sparse comments only for complex operations
- Descriptive variable and function names to minimize comment needs