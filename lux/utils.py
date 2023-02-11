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


def closest_type_tile(tile_type: str, unit, player, opponent, game_state, obs, light=False) -> np.ndarray:
    all_units = game_state.units[player]
    unit_positions = [u.pos for u in all_units.values() if u.unit_id != unit.unit_id]
    unit_positions.extend([u.pos for u in game_state.units[opponent].values()])
    opp_factories = get_factory_tiles(game_state, opponent)
    unit_positions.extend(opp_factories)
    type_tiles = obs["board"][tile_type]
    if tile_type != "rubble":
        tile_locations = np.argwhere(type_tiles == 1)
        tile_distances = np.mean((tile_locations - unit.pos) ** 2, 1)
        if light is True:
            unit_positions.extend([tile_locations[np.argmin(tile_distances)]])
        target_tile = tile_locations[np.argmin(tile_distances)]
    else:
        tile_locations = np.argwhere((game_state.board.rubble <= 40) & (game_state.board.rubble > 0))
        tile_distances = np.mean((tile_locations - unit.pos) ** 2, 1)
        if 10 < np.min(tile_distances) < 15:
            tile_locations = np.argwhere((game_state.board.rubble <= 60) & (game_state.board.rubble > 0))
            tile_distances = np.mean((tile_locations - unit.pos) ** 2, 1)
            target_tile = tile_locations[np.argmin(tile_distances)]
        elif np.min(tile_distances) > 15:
            tile_locations = np.argwhere(game_state.board.rubble > 0)
            tile_distances = np.mean((tile_locations - unit.pos) ** 2, 1)
            target_tile = tile_locations[np.argmin(tile_distances)]
        else:
            target_tile = tile_locations[np.argmin(tile_distances)]

    if unit_positions is not None:
        occupied = True
        while occupied:
            for i, u in enumerate(unit_positions):
                if i < len(tile_distances) - 1:
                    if u[0] == target_tile[0] and u[1] == target_tile[1]:
                        target_tile = tile_locations[np.argpartition(tile_distances, i + 1)[i]]
                        break
            occupied = False

    return target_tile


def closest_opp_lichen(lichen_tiles, unit, player, opponent, game_state):
    all_units = game_state.units[player]
    opp_factories = get_factory_tiles(game_state, opponent)
    unit_positions = [u.pos for u in all_units.values() if u.unit_id != unit.unit_id]
    unit_positions.extend([u.pos for u in game_state.units[opponent].values()])
    unit_positions.extend(opp_factories)
    tile_distances = np.mean((lichen_tiles - unit.pos) ** 2, 1)
    if len(tile_distances) < 1:
        print("No lichen tiles found", file=sys.stderr)
        return None
    target_tile = lichen_tiles[np.argmin(tile_distances)]

    if unit_positions is not None:
        occupied = True
        while occupied:
            for i, u in enumerate(unit_positions):
                if i < len(tile_distances) - 1:
                    if u[0] == target_tile[0] and u[1] == target_tile[1]:
                        target_tile = lichen_tiles[np.argpartition(tile_distances, i + 1)[i]]
                        break
            occupied = False

    return target_tile

def get_factory_tiles(game_state, player):
    factory_tiles = []
    factories = [u.pos for u in game_state.factories[player].values()]
    for f in factories:
        tiles = []
        tiles.append([f[0], f[1]])
        tiles.append([f[0], f[1] + 1])
        tiles.append([f[0] + 1, f[1]])
        tiles.append([f[0], f[1] - 1])
        tiles.append([f[0] - 1, f[1]])
        tiles.append([f[0] + 1, f[1] + 1])
        tiles.append([f[0] - 1, f[1] + 1])
        tiles.append([f[0] + 1, f[1] - 1])
        tiles.append([f[0] - 1, f[1] - 1])
        factory_tiles.extend(tiles)

    return factory_tiles

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