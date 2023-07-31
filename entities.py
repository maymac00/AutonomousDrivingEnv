import sys

from maps import MapManager
import numpy as np
import logging
import importlib


class Pedestrian:
    """
    Pedestrian entity
    Here we define the pedestrian behavior as private methods: deterministic, stochastic, totally random, etc.
    """

    idx = 0
    dead_cell = (-1, -1)
    @staticmethod
    def build_path_instance(pathing):
        # Start at the spawn point
        current_position = next(iter(pathing))

        path_instance = [current_position]

        # While the current position does not lead to itself
        while True:
            moves, probabilities = zip(*pathing[current_position].items())

            next_move = moves[np.random.choice(range(len(moves)), p=probabilities)]

            path_instance.append(next_move)

            current_position = next_move

            try:
                p = pathing[current_position][current_position]
                if p == 1.0:
                    break
            except KeyError:
                pass

        return path_instance

    def __init__(self, spawn_point, behavior='hardcoded', id=None, respawn=True):
        """
        :param spawn_point:
        :param map_status:
        We calculate the pedestrian's path here based on the map status and the spawn point. It should be a markov chain
        encoded as nested dictionaries with coordinates as keys and probabilities as values.

        TODO: Implement deterministic movement: A* pathing for random spawn points to closest crosswalk
        TODO: Implement stochastic movement: Maybe not needed. Erratic movement, added difficulty for cars upon predicting pedestrian movement
        """

        self.id = id if id is not None else Pedestrian.idx
        Pedestrian.idx += 1
        self.logger = logging.getLogger('pedestrian_{}'.format(self.id))
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))

        self.spawn = spawn_point
        self.pos = spawn_point
        self.last_pos = spawn_point
        self.respawn = respawn
        self.delay = 0
        self.dead = False

        self.in_crosswalk = None
        self.behavior = behavior
        self.keyframes = None  # Key frames represent orthogonal short term goals. It is a list of coordinates.
        self.markov_chain = None  # Pathing encoded as a markov chain

    def _deterministic(self, map_status, info):
        """
        :param map_status: The map status as a numpy array
        :param info: The info dictionary
        :return: The pedestrian's next position as deterministic
        """
        # Idea, build an a-star path and store it as a list of coordinates. Then, pop the first element of the list
        pass

    def _stochastic(self, map_status, info):
        """
        :param map_status: The map status as a numpy array
        :param info: The info dictionary
        :return: The pedestrian's next position as stochastic
        """
        # Idea, build an a-star path and store it as a markov chain. Then, sample from the markov chain.
        pass

    def _hardcoded(self, map_status, info):
        """
        :param map_status: The map status as a numpy array
        :param info: The info dictionary
        :return: The pedestrian's next position as hardcoded. We can do that using MapManager.current_map
        """
        if self.markov_chain is None:
            # import pathings
            try:
                m = importlib.import_module(
                    "map_sketches." + MapManager.current_map + ".pathings.pedestrian_" + str(self.id))
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"Pedestrian pathing not found for pedestrian {self.id}. Please create a "
                                          f"pathing for this pedestrian.")
            self.markov_chain = m.pathing
            self.spawn = list(self.markov_chain.keys())[0]
            self.delay = m.delay
            self.respawn = m.respawn

        if self.delay > 0:
            info["events"] = "delaying"
            return True

        if self.keyframes is None:
            self.keyframes = Pedestrian.build_path_instance(self.markov_chain)
            self.pos = self.spawn
            if np.all(self.keyframes[0] == self.pos):
                self.keyframes.pop(0)

        # Calculate the next position to move to next keyframe
        if np.all(self.pos == self.keyframes[0]):
            self.keyframes.pop(0)

        if len(self.keyframes) == 0:
            if self.respawn:
                info["events"] = "respawning"
                self.keyframes = None
                self.pos = self.spawn
                return True
            else:
                info["events"] = "deleting"
                return False
        next_keyframe = self.keyframes[0]
        direction = np.array(next_keyframe) - np.array(self.pos)
        if direction.sum() == 0:
            info["events"] = "staying"
            return True
        direction = direction / np.linalg.norm(direction)
        next_position = tuple(np.array(self.pos).astype(int) + direction.astype(int))
        self.pos = next_position
        info["events"] = "moving"
        return True

    def step(self, map_status):
        """
        :return: The pedestrian's next position. If there isn't a next move, return None. Pedestrian should be removed
        """
        if self.dead:
            self.pos = Pedestrian.dead_cell
            # print("Pedestrian", self.id, {'events': 'dead'})
            return False, {'events': 'dead'}
        self.last_pos = self.pos
        info = {}
        ret = getattr(self, "_" + self.behavior)(map_status, info)
        self.delay -= 1
        self.in_crosswalk = map_status[self.pos] == MapManager.entities["crosswalk"]["char"]
        # print("Pedestrian", self.id, info)
        return ret, info

    def __eq__(self, other):
        if isinstance(other, Pedestrian):
            return self.id == other.id
        elif isinstance(other, tuple):
            return self.pos == other
        elif isinstance(other, list):
            return self.pos in other
        elif isinstance(other, np.ndarray):
            return self.pos == tuple(other)


class Car:
    # Action Constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    UP2 = 4
    DOWN2 = 5
    LEFT2 = 6
    RIGHT2 = 7

    STAY = 8
    idx = 0

    def __init__(self, spawn_point, id=None, logger_level=logging.INFO):
        self.pos = spawn_point
        self.id = id if id is not None else Car.idx
        Car.idx += 1

        self.logger = logging.getLogger("Car-" + str(self.id))
        self.logger.setLevel(logger_level)
        self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))

        self.step_count = 0

        # Speed 1
        self.move_vectors = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        # Speed 2
        self.move_vectors += [(2, 0), (-2, 0), (0, -2), (0, 2)]
        # Stay
        self.move_vectors += [(0, 0)]
        self.move_vectors = np.array(self.move_vectors)

        self.speed = 0
        self.last_pos = self.pos
        pass

    def step(self, map_status, map_sketch, action, pedestrians):
        """
        Here we manage the car's movement and all the possible events that can happen upon movement.
        :param map_status: The current map status
        :param action: The action taken
        :return: Move the car, if possible. Return a list of events:
        - "overrun" if there is a collision with pedestrian
        - "bump" if there is a collision with a bump
        - "crash" if there is a collision with another car
        - "success" if the car has reached its goal
        - "move" if the car has moved successfully
        - "sidewalk" if the car has tried to move to the sidewalk
        - "danger" if the car is too close to a pedestrian
        - "stay" if the car has stayed in place
        """
        info = {
            "fatalities": 0,
            "injuries": 0,
            "bumps": 0,
        }

        move = self.move_vectors[action]
        self.speed = abs(move.sum())
        next_position = np.array(self.pos) + move

        # Check if the next position is valid
        if not MapManager.is_valid(next_position, map_status, map_sketch):
            events = set()
            events.add("sidewalk")
            self.logger.debug(f"Car Step: {self.step_count} Events: {events}")
            self.step_count += 1
            return events, info
        if action == Car.STAY:
            events = set()
            events.add("stay")
            self.logger.debug(f"Car Step: {self.step_count} Events: {events}")
            self.step_count += 1
            return events, info

        next_position_entity = map_status[*next_position]
        trajectory = self.trajectory(move)
        static_trajectory_hits = [map_sketch[*pos] for pos in trajectory]
        dynamic_trajectory_hits = [map_status[*pos] for pos in trajectory]
        events = set()
        # Dynamic events

        if MapManager.entities["pedestrian"]["char"] in dynamic_trajectory_hits:
            for i in range(len(pedestrians)):
                if np.any([pedestrians[i] == t for t in trajectory]) and not pedestrians[i].dead:
                    pedestrians[i].dead = True
            info["fatalities"] = dynamic_trajectory_hits.count(MapManager.entities["pedestrian"]["char"])
            events.add("overrun")

        if next_position_entity == MapManager.entities["car"]["char"]:
            events.add("crash")

        for i in range(len(pedestrians)):
            if np.any([np.array_equal(pedestrians[i],t) for t in trajectory]):
                info["injuries"] += 1
                events.add("danger")

        # Static events
        if MapManager.entities["bump"]["char"] in static_trajectory_hits:
            events.add("bump")
            events.add("moved")
            info["bumps"] = static_trajectory_hits.count(MapManager.entities["bump"]["char"])

        if next_position_entity == MapManager.entities["goal"]["char"]:
            events.add("success")
            events.add("moved")

        if next_position_entity == MapManager.entities["road"]["char"] \
                or next_position_entity == MapManager.entities["crosswalk"]["char"]:
            events.add("moved")

        if "moved" in events:
            self.last_pos = self.pos
            self.pos = next_position

        self.logger.debug(f"Car Step: {self.step_count} Events: {events}")
        self.step_count += 1
        return events, info

    def trajectory(self, move):
        """
        :param move: The move vector
        :return: The trajectory of the car as a list of coordinates
        """
        if self.speed == 0:
            return np.array([self.pos])
        steps = np.arange(1, self.speed + 1)
        steps = steps.reshape(-1, 1)
        progression = (move / self.speed * steps).astype(int)
        return [self.pos + p for p in progression]
