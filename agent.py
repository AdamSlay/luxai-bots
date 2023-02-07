from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree

import numpy as np
import sys


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    # ------------------- Factory Setup -------------------

    def manhattan_distance(self, binary_mask):
        # Get the distance map from every pixel to the nearest positive pixel
        distance_map = distance_transform_cdt(binary_mask, metric='taxicab')
        return distance_map

    def manhattan_dist_to_nth_closest(self, arr, n):
        if n == 1:
            distance_map = distance_transform_cdt(1 - arr, metric='taxicab')
            return distance_map
        else:
            true_coords = np.transpose(np.nonzero(arr))  # get the coordinates of true values
            tree = KDTree(true_coords)  # build a KDTree
            dist, _ = tree.query(np.transpose(np.nonzero(~arr)), k=n,
                                 p=1)  # query the nearest to nth closest distances using p=1 for Manhattan distance
            return np.reshape(dist[:, n - 1], arr.shape)  # reshape the result to match the input shap

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # TODO: place factories in a smart way
                ice = obs["board"]["ice"]
                ore = obs["board"]["ore"]
                ice_distances = [self.manhattan_dist_to_nth_closest(ice, i) for i in range(1, 5)]
                ore_distances = [self.manhattan_dist_to_nth_closest(ore, i) for i in range(1, 5)]
                ICE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
                weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ORE_WEIGHTS = np.array([0.8, 0.33, 0.25, 0.1])
                weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ICE_PREFERENCE = 3  # if you want to make ore more important, change to 0.3 for example

                low_rubble = (obs["board"]["rubble"] < 25)

                def count_region_cells(array, start, min_dist=2, max_dist=np.inf, exponent=1.0):

                    def dfs(array, loc):
                        distance_from_start = abs(loc[0] - start[0]) + abs(loc[1] - start[1])
                        if not (0 <= loc[0] < array.shape[0] and 0 <= loc[1] < array.shape[
                            1]):  # check to see if we're still inside the map
                            return 0
                        if (not array[loc]) or visited[
                            loc]:  # we're only interested in low rubble, not visited yet cells
                            return 0
                        if not (min_dist <= distance_from_start <= max_dist):
                            return 0

                        visited[loc] = True

                        count = 1.0 * exponent ** distance_from_start
                        count += dfs(array, (loc[0] - 1, loc[1]))
                        count += dfs(array, (loc[0] + 1, loc[1]))
                        count += dfs(array, (loc[0], loc[1] - 1))
                        count += dfs(array, (loc[0], loc[1] + 1))

                        return count

                    visited = np.zeros_like(array, dtype=bool)
                    return dfs(array, start)

                low_rubble_scores = np.zeros_like(low_rubble, dtype=float)
                for i in range(low_rubble.shape[0]):
                    for j in range(low_rubble.shape[1]):
                        low_rubble_scores[i, j] = count_region_cells(low_rubble, (i, j), min_dist=0, max_dist=8, exponent=0.9)

                combined_score = (weigthed_ice_dist * ICE_PREFERENCE + weigthed_ore_dist)
                combined_score = (np.max(combined_score) - combined_score) * obs["board"]["valid_spawns_mask"]
                overall_score = (low_rubble_scores + combined_score) * obs["board"]["valid_spawns_mask"]

                best_loc = np.argmax(overall_score)
                x, y = np.unravel_index(best_loc, (48, 48))
                spawn_loc = (x, y)
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    # ------------------- Agent Behavior -------------------
    def build_factory(self, player):
        if player.metal >= self.env_cfg.FACTORY_CHARGE:
            player.build_factory()

    def build_robot(self, unit_type: str, factory, actions, unit_id, game_state) -> bool:
        if game_state.env_steps % 3 == 0:
            unit_type = unit_type.upper()
            if unit_type == "LIGHT" and factory.can_build_light(game_state):
                actions[unit_id] = factory.build_light()
                return True
            elif unit_type == "HEAVY" and factory.can_build_heavy(game_state):
                actions[unit_id] = factory.build_heavy()
                return True
            return False

    def mine_ice(self, unit, closest_ice_tile, ice_distance, actions, unit_id, game_state):
        if np.all(closest_ice_tile == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                digs = (unit.power - unit.action_queue_cost(game_state)) // (unit.dig_cost(game_state))
                actions[unit_id] = [unit.dig(n=digs)]
        else:
            direction = direction_to(unit.pos, closest_ice_tile)
            move_cost = unit.move_cost(game_state, direction)
            if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                actions[unit_id] = [unit.move(direction, repeat=0)]

    def deliver_payload(self, unit, closest_factory_tile, adjacent_to_factory, actions, unit_id, game_state):
        direction = direction_to(unit.pos, closest_factory_tile)
        if adjacent_to_factory:
            if unit.power >= unit.action_queue_cost(game_state):
                actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, n=1)]
        else:
            move_cost = unit.move_cost(game_state, direction)
            if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                actions[unit_id] = [unit.move(direction, repeat=0)]

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]  # all of your units
        # game_state.teams[self.player].place_first   NOT SURE WHAT THIS IS

        # build heavy robots
        factory_tiles, factory_units = [], []

        for unit_id, factory in factories.items():
            self.build_robot("heavy", factory, actions, unit_id, game_state)
            if self.env_cfg.max_episode_length - game_state.env_steps < 200:
                if factory.can_water(game_state):
                    actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]

        factory_tiles = np.array(factory_tiles)  # the locations of your factories

        ice_map = game_state.board.ice  # the ice tiles
        ice_tile_locations = np.argwhere(ice_map == 1)  # the locations of the ice tiles
        ore_locations = np.argwhere(game_state.board.ore == 1)  # the locations of the ore tiles

        for unit_id, unit in units.items():
            closest_factory = None
            adjacent_to_factory = False

            # factory
            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
            adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) <= 1
            closest_factory = factory_units[np.argmin(factory_distances)]

            # ice
            ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
            ice_distance = np.min(ice_tile_distances)
            closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]

            # ore
            ore_tile_distances = np.mean((ore_locations - unit.pos) ** 2, 1)
            ore_distance = np.min(ore_tile_distances)
            closest_ore_tile = ore_locations[np.argmin(ore_tile_distances)]

            if step < 20:
                if unit_id not in actions:
                    if unit.power < 700:
                        actions[unit_id] = [unit.pickup(4, 500, n=1)]
                    else:
                        actions[unit_id] = [unit.pickup(4, 50, n=9)]
            else:
                if unit.power < 500 and not adjacent_to_factory:
                    self.deliver_payload(unit, closest_factory_tile, adjacent_to_factory, actions, unit_id, game_state)
                elif unit.power < 500 and adjacent_to_factory:
                    pickup_amt = 2000 - unit.power
                    print(f"Step: {step} - {unit_id} is RECHARGING {pickup_amt} power", file=sys.stderr)
                    actions[unit_id] = [unit.pickup(4, pickup_amt, n=1)]

                elif adjacent_to_factory and unit.cargo.ice > 0:
                    print(f"Step: {step} - {unit_id} is DELIVERING {unit.cargo.ice}", file=sys.stderr)
                    self.deliver_payload(unit, closest_factory_tile, adjacent_to_factory, actions, unit_id, game_state)

                elif not adjacent_to_factory and unit.cargo.ice < 400:
                    self.mine_ice(unit, closest_ice_tile, ice_distance, actions, unit_id, game_state)

                elif not adjacent_to_factory and unit.cargo.ice >= 400:
                    self.deliver_payload(unit, closest_factory_tile, adjacent_to_factory, actions, unit_id, game_state)

                elif unit_id not in actions:
                    self.mine_ice(unit, closest_ice_tile, ice_distance, actions, unit_id, game_state)

        return actions
