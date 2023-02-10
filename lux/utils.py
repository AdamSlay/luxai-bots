import random
import sys

import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree


def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


def next_position(unit, direction):
    if direction == 0:  # center
        return unit.pos
    if direction == 1:  # up
        return [unit.pos[0], unit.pos[1] - 1]
    elif direction == 2:  # right
        return [unit.pos[0] + 1, unit.pos[1]]
    elif direction == 3:  # down
        return [unit.pos[0], unit.pos[1] + 1]
    elif direction == 4:  # left
        return [unit.pos[0] - 1, unit.pos[1]]
    else:
        print(f"Error: invalid direction in next_position {direction}", file=sys.stderr)


def find_new_direction(unit, unit_positions, game_state) -> int:
    r = list(range(1, 5))
    random.shuffle(r)
    for d in r:
        new_pos = next_position(unit, d)
        for pos in unit_positions:
            if new_pos[0] == pos[0] and new_pos[1] == pos[1]:
                return 0
        if 0 <= new_pos[0] < 48 and 0 <= new_pos[1] < 48:
            return d
    return 0


def closest_type_tile(tile_type: str, unit, game_state, obs, unit_positions=None) -> np.ndarray:
    type_tiles = obs["board"][tile_type]
    if tile_type != "rubble":
        tile_locations = np.argwhere(type_tiles == 1)
    else:
        tile_locations = np.argwhere((game_state.board.rubble <= 80) & (game_state.board.rubble > 0))
    tile_distances = np.mean((tile_locations - unit.pos) ** 2, 1)
    target_tile = tile_locations[np.argmin(tile_distances)]
    i = 1
    if unit_positions is not None:
        while (target_tile & unit_positions).all():
            print(f"{unit.unit_id} {target_tile} is occupied", file=sys.stderr)
            target_tile = tile_locations[np.argpartition(tile_distances, i + 1)[i]]
            print(f"{unit.unit_id} finding next closest tile: {target_tile}", file=sys.stderr)
            i += 1
    return target_tile


def factory_adjacent(factory_tile, unit) -> bool:
    return np.mean((factory_tile - unit.pos) ** 2) <= 1


def closest_factory(factory_units, factory_tiles, unit, game_state) -> np.ndarray:
    factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
    closest = factory_units[np.argmin(factory_distances)]
    return closest


def factory_distance(factory_tiles, unit) -> int:
    factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
    return np.min(factory_distances)


def manhattan_distance(binary_mask):
    # Get the distance map from every pixel to the nearest positive pixel
    distance_map = distance_transform_cdt(binary_mask, metric='taxicab')
    return distance_map


def manhattan_dist_to_nth_closest(arr, n):
    if n == 1:
        distance_map = distance_transform_cdt(1 - arr, metric='taxicab')
        return distance_map
    else:
        true_coords = np.transpose(np.nonzero(arr))  # get the coordinates of true values
        tree = KDTree(true_coords)  # build a KDTree
        dist, _ = tree.query(np.transpose(np.nonzero(~arr)), k=n,
                             p=1)  # query the nearest to nth closest distances using p=1 for Manhattan distance
        return np.reshape(dist[:, n - 1], arr.shape)  # reshape the result to match the input shap
