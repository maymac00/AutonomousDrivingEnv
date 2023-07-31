import numpy as np
from PIL import Image


class MapManager:
    """
    MapManager is a static class that manages the maps used in the environment.
    The images loaded contain the following entities:
    - Car:  1   (Yellow) '@'
    - Road: 0   (Black)  ' '
    - Bump: 4   (Brown)  '?'
    - Pedestrian: 2 (Purple) '!'
    - Crosswalk: 3  (Blue) '='
    - SideWalk: 5   (Gray)  '#'
    - Goal: 6   (Green) 'G'

    The images contain the fixed position of the entities. Dynamic entities (pedestrian, car) in the sketch are interpreted as spawn points.
    Cars can only spawn on the road, pedestrians can spawn on the sidewalk.
    """

    entities = {
        "car": {'number': 1, 'char': '@', 'RGB': (255, 255, 0)},
        "road": {'number': 0, 'char': ' ', 'RGB': (0, 0, 0)},
        "bump": {'number': 4, 'char': '?', 'RGB': (117, 29, 29)},
        "pedestrian": {'number': 2, 'char': '!', 'RGB': (255, 0, 255)},
        "crosswalk": {'number': 3, 'char': '=', 'RGB': (0, 0, 255)},
        "sidewalk": {'number': 5, 'char': '#', 'RGB': (107, 107, 107)},
        "goal": {'number': 6, 'char': 'G', 'RGB': (0, 255, 0)},
    }
    entities_by_char = {v['char']: v for v in entities.values()}
    n_entities = len(entities)
    map_dir = 'map_sketches/'
    map_ext = '.png'
    current_map = None

    @staticmethod
    def image_to_arrays(image_name):
        # Open image and convert to RGB array
        MapManager.current_map = image_name
        image_path = MapManager.map_dir + image_name + "/" + image_name + MapManager.map_ext
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)

        # Initialize empty arrays for entity numbers and chars
        number_array = np.empty(image_array.shape[:2], dtype=int)
        char_array = np.empty(image_array.shape[:2], dtype=str)

        # Iterate over each pixel
        key_order = list(MapManager.entities.keys())
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                pixel = tuple(image_array[i, j, :])
                closest_color = np.argmin(
                    [np.sum((np.array(v['RGB']) - pixel) ** 2) for v in MapManager.entities.values()])
                number_array[i, j] = MapManager.entities[key_order[closest_color]]['number']
                char_array[i, j] = MapManager.entities[key_order[closest_color]]['char']

        # Find spawn points of dynamic entities
        car_spawn_points = np.argwhere(number_array == MapManager.entities['car']['number'])
        pedestrian_spawn_points = np.argwhere(number_array == MapManager.entities['pedestrian']['number'])

        # Remove spawn points from map
        number_array[car_spawn_points[:, 0], car_spawn_points[:, 1]] = MapManager.entities['road']['number']
        char_array[car_spawn_points[:, 0], car_spawn_points[:, 1]] = MapManager.entities['road']['char']
        number_array[pedestrian_spawn_points[:, 0], pedestrian_spawn_points[:, 1]] = MapManager.entities['sidewalk'][
            'number']
        char_array[pedestrian_spawn_points[:, 0], pedestrian_spawn_points[:, 1]] = MapManager.entities['sidewalk'][
            'char']
        return number_array, char_array, car_spawn_points, pedestrian_spawn_points

    @staticmethod
    def char_to_render(char_array):
        """
        Converts the char array into a pixel image using entity colors.
        :param char_array: Array of chars representing the current status of the map
        :return:
        """

        # Initialize empty array for RGB values
        image_array = np.empty((*char_array.shape, 3), dtype=int)
        # Iterate over each pixel
        for i in range(char_array.shape[0]):
            for j in range(char_array.shape[1]):
                char = char_array[i, j]
                image_array[i, j, :] = MapManager.entities_by_char[char]['RGB']

        return image_array

    @staticmethod
    def is_valid(position, map, map_sketch):
        """
        Checks if a position is valid on the given array map
        :param position:
        :param map:
        :return:
        """
        if 0 <= position[0] < map.shape[0] and 0 <= position[1] < map.shape[1]:
            if map_sketch[*position] == MapManager.entities['sidewalk']['char']:
                return False
            else:
                return True
        else:
            return False


class Map:
    def __init__(self, map_name):
        self.name = map_name
        self.number_map, self.char_map, self.car_spawn_points, self.pedestrian_spawn_points = MapManager.image_to_arrays(
            map_name)
        self.w = self.number_map.shape[0]
        self.h = self.number_map.shape[1]

    def __getitem__(self, item):
        if item == 'number':
            return self.number_map
        elif item == 'char':
            return self.char_map
        elif isinstance(item, tuple):
            return self.number_map[item]
        else:
            raise KeyError("Map has no attribute '{}'".format(item))

    def normalized_map(self):
        return self.number_map / (MapManager.n_entities - 1)


if __name__ == '__main__':
    # Test the Map and MapManager class

    # Create a map using the image named "small"
    map_obj = Map("small")

    # Display the map's dimensions
    print(f"Map dimensions: {map_obj.w} x {map_obj.h}")

    # Get and print the numerical and character representation of the map
    print("Numerical map representation:")
    print(map_obj['number'])

    print("Character map representation:")
    print(map_obj['char'])

    # Access a specific location on the map
    # This will give you the number representing the entity at location (0, 0)
    print(f"Entity at location (0,0): {map_obj[(0, 0)]}")

    # Get and print the normalized map
    print("Normalized and denormalized map representation:")
    print(map_obj.normalized_map() * (MapManager.n_entities - 1))
