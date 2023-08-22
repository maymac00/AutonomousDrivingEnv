# Autonomous Driving Simulator

This simulator is an implementation of an environment for reinforcement learning using the OpenAI Gym interface. The environment represents an autonomous driving simulation where the agent must navigate through a city environment filled with various entities such as pedestrians, bumps, crosswalks, sidewalks, and goals. The agent receives three reward signals: speed, internal safety of the car, and external safety (e.g., safety of pedestrians).

## Environment

The main class in this simulator is `AutonomousDriving`, which is a subclass of `gym.Env`. This class implements the following methods:

- `__init__(self, map='small', n_agents=1, fixed_spawn=True, pedestrian_behavior='hardcoded', speed_penalizing=False, keepHistory=False)`: Initializes a new instance of the environment.

- `step(self, action, verbose=0)`: Executes one time step within the environment.

- `reset(self)`: Resets the environment to its initial state.

- `get_observation(self)`: Returns the current observation that the agent sees.

- `char_map_status(self)`: Writes the dynamic entities in the char map.

- `render(self, mode='human', pause=1)`: Renders the environment to the screen.

- `export_history(self, path="")`: Exports the history of the environment to a CSV file.

Main attributes of the environment:
- `n_agents`: Number of cars in the environment.

- `n_pedestrians`: Number of pedestrians in the environment.

- `n_bumps`: Number of bumps in the environment.

The parameters of the environment are the following:

- `map` (default: 'small'): 
    - Specifies the name of the map to be used for the simulation. It should correspond to a folder name in the `map_sketches` folder.

- `n_agents` (default: 1): 
    - Specifies the number of agents in the environment. In this context, it refers to the number of cars.

- `fixed_spawn` (default: True): 
    - If set to True, cars will always spawn at the spawn points specified on the map. If set to False, they will spawn randomly in the environment.

- `pedestrian_behavior` (default: 'hardcoded'): 
    - Defines the behavior of the pedestrians in the environment. Currently, only 'hardcoded' behavior is implemented.

- `speed_penalizing` (default: False): 
    - If set to True, the reward function will penalize higher speeds. This is done to emphasize safety; higher speeds can lead to more severe consequences in case of accidents.

- `keepHistory` (default: False): 
    - If set to True, the simulator keeps a history of the environment. This includes the state and action at each timestep, as well as the subsequent state and reward. This history persists across environment resets. It has to be set to True to be able to use the export functionality.

- `state_representation` (default: 'full_matrix'): 
    - Defines the representation of the state. Options are 'full_matrix' and 'positional'. The 'full_matrix' representation includes the full state of the environment, as it represents the entity sitting in each cell. Positional representation only includes the position of the car, the pedestrians, and the bumps. In that order.
- `value_system` (default: 'iev'):
    - Defines the value system to be used. The order of the objectives prioritization is specified in a string using the initial of each value. Therefore, the options are the 6 permutations of the characters 'v' (velocity), 'i' (internal safety), and 'e' (external safety). For example, 'iev' means that the internal safety is prioritized over the external safety, which is prioritized over the velocity. The default value is 'iev'.
## Installation

This project requires Python and it is built mainly upon the following Python libraries installed:

- NumPy
- Gym
- Matplotlib

A requirements file is provided to install the required libraries. It is recommended to create a new conda environment with the specific dependencies to ensure that the environment works as desired. To install them, run the following command:

```bash 
pip install -r requirements.txt
```
## Usage

To use the `AutonomousDriving` environment in your Python class, you need to import it and create an instance as follows:

```python
from env import AutonomousDriving
env = AutonomousDriving()
```
