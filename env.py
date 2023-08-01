import csv
import json
import time

import gym
import numpy as np

from entities import Car, Pedestrian
from maps import MapManager, Map
import matplotlib.pyplot as plt


class AutonomousDriving(gym.Env):
    def __init__(self, map='small', n_agents=1, fixed_spawn=True, pedestrian_behavior='hardcoded',
                 speed_penalizing=False, keepHistory=False, state_representation='full_matrix'):
        """
        :param map: Folder name of the map to be used. Must be in map_sketches folder
        :param n_agents: Number of agents in the environment. Number of cars
        :param fixed_spawn: If True, cars will always spawn in the spawn points specified on the map. If False, they will spawn randomly
        :param pedestrian_behavior: Defines the behavior of the pedestrians. Only hardcoded implemented
        :param speed_penalizing: Penalize speed in the reward function. Safety issues are magnified at higher speeds
        :param keepHistory: Keeps track of the history of the environment. It works across resets.
        """
        super(AutonomousDriving, self).__init__()

        self.speed_penalizing = speed_penalizing

        self.action_space = gym.spaces.Discrete(9)
        self.map = Map(map)
        self.last_frame = None

        self.state_representation = state_representation

        # Get position of fixed entities. Maybe not necessary to store them as attributes
        self.bump_positions = np.argwhere(self.map["number"] == MapManager.entities['bump']['number']).tolist()
        self.crosswalk_positions = np.argwhere(self.map["number"] == MapManager.entities['crosswalk']['number'])
        self.sidewalk_positions = np.argwhere(
            self.map["number"] == MapManager.entities['sidewalk']['number'])  # May not be necessary
        self.goal_positions = np.argwhere(self.map["number"] == MapManager.entities['goal']['number'])
        self.road_positions = np.argwhere(self.map["number"] == MapManager.entities['road']['number'])

        self.fixed_spawn = fixed_spawn
        self.n_agents = n_agents
        self.keepHistory = keepHistory
        self.history = []

        self.pedestrians = None
        self.pedestrian_behavior = pedestrian_behavior
        self.n_pedestrians = json.load(open(MapManager.map_dir + map + '/description.json'))['n_pedestrians']
        Pedestrian.dead_cell = json.load(open(MapManager.map_dir + map + '/description.json'))['dead_cell']
        self.n_bumps = json.load(open(MapManager.map_dir + map + '/description.json'))['n_bumps']

        self.cars = None

        if state_representation == 'full_matrix':
            self.observation_space = gym.spaces.Box(low=0, high=MapManager.n_entities, shape=(self.map.h, self.map.w),
                                                    dtype=np.uint8)
        elif state_representation == 'tabular':
            # TODO: This is definitely not the way to do this. It's just a placeholder
            self.observation_space = gym.spaces.Discrete(MapManager.n_entities ** (self.map.h * self.map.w))
        elif state_representation == 'positional':
            point_space = gym.spaces.Tuple((gym.spaces.Discrete(self.map.h), gym.spaces.Discrete(self.map.w)))
            self.observation_space = gym.spaces.Tuple([point_space for _ in range(self.n_pedestrians + self.n_agents)])

        self.reset()

    def step(self, action, verbose=0):
        # Execute one time step within the environment
        # First move pedestrians, then cars. So the car is who overtakes the pedestrian not the other way around

        # Print initial positions of cars and pedestrians
        self.last_frame = self.char_map_status()
        init_state = [self.cars[0].pos, *[p.pos for p in self.pedestrians]]
        if verbose > 0:
            print("Initial positions: CAR", self.cars[0].pos, "PEDESTRIANS", [p.pos for p in self.pedestrians])
            print("Action taken:", action)

        # Move pedestrians

        to_remove = []
        pedestrian_info = {k: {} for k in range(len(self.pedestrians))}
        for k, pedestrian in enumerate(self.pedestrians):
            # Pedestrian moves. If return False, path ended. Remove pedestrian
            keep_going, info = pedestrian.step(self.map["char"])
            pedestrian_info[k] = info
            if not keep_going:
                # We keep the pedestrian on the list, it will print in the dead cell.
                pass # to_remove.append(pedestrian)

        for pedestrian in to_remove:
            self.pedestrians.remove(pedestrian)

        map_status = self.char_map_status()
        # Move cars TODO: Extend to MA. Add penalty for event crash
        # Car moves. Returns events.
        events, car_info = self.cars[0].step(map_status, self.map["char"], action, self.pedestrians)
        done = False
        # Reward function based on events
        reward_vec = np.zeros(3)

        # Individual reward
        if 'success' in events:
            reward_vec[0] += 14
            done = True
        else:
            reward_vec[0] += -1

        # Internal safety reward
        if 'bump' in events:
            speed_penalty = self.cars[0].speed if self.speed_penalizing else 1
            reward_vec[1] += -10 * speed_penalty * car_info['bumps']

        # External safety reward
        if 'overrun' in events:
            speed_penalty = self.cars[0].speed if self.speed_penalizing else 1
            reward_vec[2] += -10 * speed_penalty * car_info['fatalities']

        if 'danger' in events:
            speed_penalty = self.cars[0].speed if self.speed_penalizing else 1
            reward_vec[2] += -10 * speed_penalty * 0.3 * car_info['injuries']  # 0.3 as non lethal multiplier

        observation = self.get_observation()
        info = {'car': car_info, 'pedestrians': pedestrian_info}

        final_state = [self.cars[0].pos, *[p.pos for p in self.pedestrians]]
        if self.keepHistory:
            # TODO: Deliberate what we want to keep in history apart from s and s'
            # TODO: Deliberate what we do in case of deletion of entities
            self.history.append(np.hstack([np.array(init_state).reshape(-1), action, np.array(final_state).reshape(-1), reward_vec]))

        if verbose > 1:
            print("Car Events:", events)
        # Print final positions of cars and pedestrians
        if verbose > 0:
            print("Final positions: CAR", self.cars[0].pos, "PEDESTRIANS", [p.pos for p in self.pedestrians])

        return observation, reward_vec, done, info

    def reset(self):
        # Spawn dynamic entities
        # Car spawn points
        if len(self.map.car_spawn_points) < self.n_agents:
            raise ValueError('Not enough spawn points for the number of agents')
        if self.fixed_spawn:
            np.random.shuffle(self.map.car_spawn_points)
            self.cars = {k: Car(v) for k, v in enumerate(self.map.car_spawn_points[:self.n_agents])}
        else:
            # TODO: Do random spawn in the road
            raise NotImplementedError

        # Pedestrian spawn.
        Pedestrian.idx = 0
        self.pedestrians = [Pedestrian(Pedestrian.dead_cell, self.pedestrian_behavior) for i in
                            range(self.n_pedestrians)]
        return self.get_observation()

    def get_observation(self):

        if self.state_representation == 'full_matrix':
            # Get a copy of number map and overwrite the positions of the dynamic entities
            observation = self.map["number"].copy()
            for pedestrian in self.pedestrians:
                observation[pedestrian.pos] = MapManager.entities['pedestrian']['number']
            for car in self.cars.values():
                observation[car.pos] = MapManager.entities['car']['number']
            return observation
        elif self.state_representation == 'tabular':
            raise NotImplementedError
        elif self.state_representation == 'positional':
            return np.array([car.pos for car in self.cars.values()] + [pedestrian.pos for pedestrian in self.pedestrians] + list(self.bump_positions)).reshape(-1)

    def char_map_status(self):
        """
        Writes the dynamic entities in the char map.
        :return:
        """
        char_map = self.map["char"].copy()
        for car in self.cars.values():
            char_map[*car.pos] = MapManager.entities['car']['char']
        for pedestrian in self.pedestrians:
            char_map[*pedestrian.pos] = MapManager.entities['pedestrian']['char']
        return char_map

    def render(self, mode='human', pause=1):
        """
        :param pause: Pause between frames
        :param mode: determines how to render the environment. We could have a pretty image, or a pixel representation
        :return:
        """
        if mode == 'text':
            print(self.char_map_status())
            time.sleep(pause)
        elif mode == 'human':
            plt.figure(pause)
            plt.ion()
            plt.imshow(MapManager.char_to_render(self.char_map_status()))
            plt.axis('off')
            plt.title('Autonomous Driving')  # TODO: Add relevant info
            plt.pause(pause)
            plt.clf()

        # Render the environment to the screen
        # First get fixed entities, then dynamic entities. So they are drawn on top

        pass

    def export_history(self, path=""):
        """
        Export the history of the environment to a csv file
        History is a list of arrays of the form [s, a, s']
        :param path: path to the file
        :return:
        """
        # Build header
        header = []
        for i in range(self.n_agents):
            header += [f'car{i}_x', f'car{i}_y']
        for i in range(len(self.pedestrians)):
            header += [f'ped{i}_x', f'ped{i}_y']
        header += ['action']
        for i in range(self.n_agents):
            header += [f'car{i}_x_prima', f'car{i}_y_prima']
        for i in range(len(self.pedestrians)):
            header += [f'ped{i}_x_prima', f'ped{i}_y_prima']
        header += ['Speed', 'Internal', 'External']
        header = ",".join(header)

        # Append the numpy array
        if path[-4:] != ".csv":
            path += ".csv"
        np.savetxt(path, np.array(self.history), delimiter=",", fmt='%s', newline="\n", footer="", header=header, comments="")