"""
Pedestrian pathing specification. Example
The pathing is encoded as a dictionary with name pathing. the keys are the tuples of the coordinates, starting from the
spawn point and ending at the last position. The coordinates do not need to be at distance 1 from each other, they just
need to represent an orthogonal movement (no diagonal movement). The values are the probabilities of moving to the next
coordinate.

Respawn is a boolean that indicates if the pedestrian should respawn after reaching the last coordinate.
Delay is the number of steps before the pedestrian spawns for the first time.

Important notes: !!!
- The sum of the probabilities must be 1.0.
- The last coordinate of a path must have a probability of 1.0 to itself.
- First coordinate must be the spawn point.
- If a path is circular, the last coordinate must be the coordinate before the spawn point.
"""
pathing = {
    (4, 3): {(3, 3): 1.0},
    (3, 3): {(3, 0): 0.333, (0, 3): 0.333, (3, 3): 0.334},
    (3, 0): {(7, 0): 1.0},
    (7, 0): {(7, 0): 1.0},
    (0, 3): {(0, 0): 1.0},
    (0, 0): {(7, 0): 1.0},
}
respawn = True
delay = 0
