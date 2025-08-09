===============================
vivarium_eye_vessels
===============================

Vivarium simulation model to create synthetic data that looks like the vascular system of the human eye.


.. contents::
   :depth: 1

Background
----------

Vivarium uses a modular approach to agent-based modeling, and this sim combines components
for general spatial simulation with custom components to grow blood vessels, splitting them
and avoiding existing vessels as they go.

The simulation works by treating vessel segments as particles in 3D space that move according 
to physics-based forces. Each particle has position, velocity, and state (active or frozen), 
with particles freezing to form permanent vessel segments.

Core Components
~~~~~~~~~~~~~~~

**Particle System (particles.py)**

- **Particle3D()**: The foundational component that manages 3D particle positions, velocities, and states. Handles basic physics updates including velocity changes, terminal velocity limits, and initial particle placement in configurable patterns.

- **PathFreezer()**: The core vessel-building component that periodically freezes moving particles to create permanent vessel segments. The ``freeze_interval`` parameter controls vessel appearance: values of 2-4 create detailed, continuous vessel networks ideal for retinal capillaries, while values of 8-15 create more scattered growth points suitable for sparse vessel distributions. The component automatically manages particle population by converting available particles into vessel continuations and maintains spatial indexing via KDTree for efficient queries by other components like FrozenRepulsion.

- **PathSplitter()**: Creates vessel branching by splitting existing paths when certain conditions are met. Generates new particles at split points with modified trajectories, simulating how blood vessels branch into smaller vessels.

- **PathExtinction()**: Removes particles that are no longer viable for vessel growth, typically when forces acting on them exceed a threshold, preventing unrealistic vessel extensions.

- **PathDLA()**: Implements Diffusion Limited Aggregation dynamics where particles stick to frozen structures when they come within a specified radius, creating organic branching patterns similar to real vessel growth.

**Boundary Forces (boundaries.py)**

- **EllipsoidContainment()**: Applies forces to keep particles within an ellipsoidal boundary, simulating the overall shape constraint of the eye or specific eye regions.

- **CylinderExclusion()**: Creates exclusion zones (like the optic nerve) where particles experience repulsive forces, preventing vessel growth in anatomically inappropriate areas.

- **PointRepulsion()**: Applies localized repulsive forces from specific points, useful for creating clearance around critical structures.

- **FrozenRepulsion()**: Prevents particles from colliding with existing frozen vessel segments, ensuring vessels maintain proper spacing and don't overlap.

**Visualization and Analysis**

- **ParticleVisualizer3D()**: Real-time 3D visualization using pygame, displaying particle positions, frozen vessel segments, and boundary constraints with configurable colors and viewing angles.

- **SaveParticles()**: Records particle positions and states throughout the simulation for post-processing analysis and vessel network extraction.

System Integration
~~~~~~~~~~~~~~~~~~

The components work together through Vivarium's event-driven architecture. Each timestep, 
particles experience forces from boundary components, move according to physics updates, 
and may undergo state changes (freezing, splitting, extinction) based on component-specific 
logic. The result is an emergent vessel network that respects anatomical constraints while 
exhibiting realistic branching patterns.


Installation
------------

You will need ``conda`` to install all of this repository's requirements.
We recommend installing `Miniforge <https://github.com/conda-forge/miniforge>`_.

Once you have conda installed, you should open up your normal shell
(if you're on linux or OSX) or the ``git bash`` shell if you're on windows.
You'll then make an environment, clone this repository, then install
all necessary requirements as follows::

  :~$ conda create --name=vivarium_eye_vessels python=3.11 git git-lfs
  ...conda will download python and base dependencies...
  :~$ conda activate vivarium_eye_vessels
  (vivarium_eye_vessels) :~$ git clone https://github.com/ihmeuw/vivarium_eye_vessels.git
  ...git will copy the repository from github and place it in your current directory...
  (vivarium_eye_vessels) :~$ cd vivarium_eye_vessels
  (vivarium_eye_vessels) :~$ pip install -e .
  ...pip will install vivarium and other requirements...

Supported Python versions: 3.10, 3.11

Note the ``-e`` flag that follows pip install. This will install the python
package in-place, which is important for making the model specifications later.

To install requirements from a provided requirements.txt (e.g. installing an
archived repository with the exact same requirements it was run with), replace
`pip install -e .` with the following::

  (vivarium_eye_vessels) :~$ pip install -r requirements.txt

Cloning the repository should take a fair bit of time as git must fetch
the data artifact associated with the demo (several GB of data) from the
large file system storage (``git-lfs``). **If your clone works quickly,
you are likely only retrieving the checksum file that github holds onto,
and your simulations will fail.** If you are only retrieving checksum
files you can explicitly pull the data by executing ``git-lfs pull``.

Vivarium uses the Hierarchical Data Format (HDF) as the backing storage
for the data artifacts that supply data to the simulation. You may not have
the needed libraries on your system to interact with these files, and this is
not something that can be specified and installed with the rest of the package's
dependencies via ``pip``. If you encounter HDF5-related errors, you should
install hdf tooling from within your environment like so::

  (vivarium_eye_vessels) :~$ conda install hdf5

The ``(vivarium_eye_vessels)`` that precedes your shell prompt will probably show
up by default, though it may not.  It's just a visual reminder that you
are installing and running things in an isolated programming environment
so it doesn't conflict with other source code and libraries on your
system.


Usage
-----

You'll find six directories inside the main
``src/vivarium_eye_vessels`` package directory:

- ``artifacts``

  This directory contains all input data used to run the simulations.
  You can open these files and examine the input data using the vivarium
  artifact tools.  A tutorial can be found at https://vivarium.readthedocs.io/en/latest/tutorials/artifact.html#reading-data

- ``components``

  This directory is for Python modules containing custom components for
  the vivarium_eye_vessels project. You should work with the
  engineering staff to help scope out what you need and get them built.

- ``data``

  If you have **small scale** external data for use in your sim or in your
  results processing, it can live here. This is almost certainly not the right
  place for data, so make sure there's not a better place to put it first.

- ``model_specifications``

  This directory should hold all model specifications and branch files
  associated with the project.

- ``results_processing``

  Any post-processing and analysis code or notebooks you write should be
  stored in this directory.

- ``tools``

  This directory hold Python files used to run scripts used to prepare input
  data or process outputs.


Running Simulations
-------------------

Before running a simulation, you should have a model specification file.
A model specification is a complete description of a vivarium model in
a yaml format.  An example model specification is provided with this repository
in the ``model_specifications`` directory.

With this model specification file and your conda environment active, you can then run simulations by, e.g.::

   (vivarium_eye_vessels) :~$ simulate run src/vivarium_eye_vessels/model_specifications/model_spec.yaml

The ``-v`` flag will log verbosely, so you will get log messages every time
step. For more ways to run simulations, see the tutorials at
https://vivarium.readthedocs.io/en/latest/tutorials/running_a_simulation/index.html
and https://vivarium.readthedocs.io/en/latest/tutorials/exploration.html

Demonstrations
~~~~~~~~~~~~~~

The repository includes several demonstration configurations and tests:

**Basic particle system**::

   python -m pytest tests/test_basic_visualization.py -v -s

**PathFreezer vessel formation**::

   python -m pytest tests/test_path_freezer_demo.py -v -s

**Available model specifications**:

- ``basic_particles.yaml``: Simple particle movement demo (100 particles, 5 seconds)
- ``path_freezer_demo.yaml``: Vessel formation with PathFreezer (200+ particles, eye-like parameters)  
- ``model_spec.yaml``: Full eye vessel simulation with all components
