from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory, closest_type_tile, find_new_direction, next_position, \
    factory_adjacent, closest_factory
from lux.actions import mine_type, deliver_to_factory
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree

import numpy as np
import sys


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.act_step = 0
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

            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)

            if factories_to_place > 0 and my_turn_to_place:
                # TODO: place factories in a smart way
                ice = obs["board"]["ice"]
                ore = obs["board"]["ore"]
                ice_distances = [self.manhattan_dist_to_nth_closest(ice, i) for i in range(1, 5)]
                ore_distances = [self.manhattan_dist_to_nth_closest(ore, i) for i in range(1, 5)]
                ICE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
                weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ORE_WEIGHTS = np.array([0.7, 0, 0, 0])
                weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ICE_PREFERENCE = 5  # if you want to make ore more important, change to 0.3 for example

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
                        low_rubble_scores[i, j] = count_region_cells(low_rubble, (i, j), min_dist=0, max_dist=8,
                                                                     exponent=0.9)

                combined_score = (weigthed_ice_dist * ICE_PREFERENCE + weigthed_ore_dist)
                combined_score = (np.max(combined_score) - combined_score) * obs["board"]["valid_spawns_mask"]
                overall_score = (low_rubble_scores + (combined_score * 1)) * obs["board"]["valid_spawns_mask"]

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
        unit_type = unit_type.upper()
        if unit_type == "LIGHT" and factory.can_build_light(game_state):
            actions[unit_id] = factory.build_light()
            return True
        elif unit_type == "HEAVY" and factory.can_build_heavy(game_state):
            actions[unit_id] = factory.build_heavy()
            print(f"Step {self.act_step}: {unit_id} Building heavy robot at {factory.pos}", file=sys.stderr)
            return True
        return False

    def path_direction(self, target_tile, unit_positions, unit, game_state) -> int:
        # direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        direction = direction_to(unit.pos, target_tile)
        move_cost = unit.move_cost(game_state, direction)

        # all_units = game_state.units[self.player]
        # unit_positions = [u.pos for u in all_units.values() if u.unit_id != unit.unit_id]
        if move_cost is not None and unit.power >= (move_cost + unit.action_queue_cost(game_state)):
            new_pos = next_position(unit, direction)
            for pos in unit_positions:
                if new_pos[0] == pos[0] and new_pos[1] == pos[1]:
                    print(f"S{self.act_step}: {unit.unit_id} new_pos {new_pos} in unit_positions {unit_positions}", file=sys.stderr)
                    new_direction = find_new_direction(unit, unit_positions, game_state)
                    return new_direction
            else:
                return direction
        else:
            return 0

    def dig_rubble(self, unit, actions, game_state, obs):
        rubble_at_pos = game_state.board.rubble[unit.pos[0]][unit.pos[1]]

        target_tile = closest_type_tile("rubble", unit, game_state, self.player, obs)

        if np.all(target_tile == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
                digs_needed = (rubble_at_pos // 2) + 1
                cost = digs_needed * unit.dig_cost(game_state)
                digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                print(f"Step {self.act_step}: {unit.unit_id} Diggin rubble at {unit.pos} {digs} times", file=sys.stderr)
                actions[unit.unit_id] = [unit.dig(repeat=True, n=digs)]
        else:
            all_units = game_state.units[self.player]
            unit_positions = [u.pos for u in all_units.values() if u.unit_id != unit.unit_id]
            direction = self.path_direction(target_tile, unit_positions, unit, game_state)
            actions[unit.unit_id] = [unit.move(direction, repeat=0)]

    def deliver_payload(self, unit, closest_factory_tile, actions, game_state):
        direction = direction_to(unit.pos, closest_factory_tile)
        adjacent_to_factory = factory_adjacent(closest_factory_tile, unit)
        if adjacent_to_factory and unit.unit_type == "HEAVY":
            if unit.power >= unit.action_queue_cost(game_state):
                actions[unit.unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, n=1)]
        else:
            move_cost = unit.move_cost(game_state, direction)
            if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                actions[unit.unit_id] = [unit.move(direction, repeat=0)]

    def heavy_actions(self, step, unit, closest_f, actions, game_state, obs):
        adjacent_to_factory = factory_adjacent(closest_f.pos, unit)

        if self.act_step < 12:
            if unit.unit_id not in actions:
                if unit.power < 700:
                    actions[unit.unit_id] = [unit.pickup(4, 500, n=1)]
                else:
                    actions[unit.unit_id] = [unit.pickup(4, 50, n=9)]

        elif step < 850:
            if unit.power < 500 and not adjacent_to_factory:
                self.deliver_payload(unit, closest_f.pos, actions, game_state)
            elif unit.power < 500 and adjacent_to_factory:
                pickup_amt = (closest_f.power % 3000) // 2
                # print(f"Step: {step} - {unit_id} is RECHARGING {pickup_amt} power", file=sys.stderr)
                actions[unit.unit_id] = [unit.pickup(4, pickup_amt, n=1)]

            elif adjacent_to_factory and unit.cargo.ice > 0:
                # print(f"Step: {step} - {unit_id} is DELIVERING {unit.cargo.ice}", file=sys.stderr)
                self.deliver_payload(unit, closest_f.pos, actions, game_state)

            elif not adjacent_to_factory and unit.cargo.ice < 400:
                mine_type("ice", unit, actions, game_state, self.player, obs)

            elif not adjacent_to_factory and unit.cargo.ice >= 400:
                self.deliver_payload(unit, closest_f.pos, actions, game_state)

            elif unit.unit_id not in actions:
                mine_type("ice", unit, actions, game_state, self.player, obs)

        else:
            if unit.power < 400 and not adjacent_to_factory:
                self.deliver_payload(unit, closest_f.pos, actions, game_state)
            elif unit.power < 400 and adjacent_to_factory:
                pickup_amt = (closest_f.power % 3000) // 2  # apparently heavies cant pickup more than 3000 power
                # print(f"Step: {step} - {unit_id} is RECHARGING {pickup_amt} power", file=sys.stderr)
                actions[unit.unit_id] = [unit.pickup(4, pickup_amt, n=1)]

            elif adjacent_to_factory and unit.cargo.ice > 0:
                # print(f"Step: {step} - {unit_id} is DELIVERING {unit.cargo.ice}", file=sys.stderr)
                self.deliver_payload(unit, closest_f.pos, actions, game_state)

            elif not adjacent_to_factory and unit.cargo.ice < 400:
                # self.mine_ice(unit, closest_ice_tile, ice_distance, actions, unit_id, game_state)
                mine_type("ice", unit, actions, game_state, self.player, obs)

            elif not adjacent_to_factory and unit.cargo.ice >= 400:
                self.deliver_payload(unit, closest_f.pos, actions, game_state)

            elif unit.unit_id not in actions:
                # self.mine_ice(unit, closest_ice_tile, ice_distance, actions, unit_id, game_state)
                mine_type("ice", unit, actions, game_state, self.player, obs)

    def light_actions(self, unit, closest_f, actions, game_state, obs):
        adjacent_to_factory = factory_adjacent(closest_f.pos, unit)

        if unit.power < 80 and not adjacent_to_factory:
            self.deliver_payload(unit, closest_f.pos, actions, game_state)
        elif unit.power < 80 and adjacent_to_factory:
            if closest_f.power > 500:
                pickup_amt = closest_f.power // 8
                print(f"Step {self.act_step}: {unit.unit_id}(light) is RECHARGING {pickup_amt} power",
                      file=sys.stderr)
                actions[unit.unit_id] = [unit.pickup(4, 200, n=1)]
        if unit.unit_id not in actions:
            self.dig_rubble(unit, actions, game_state, obs)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # SETUP
        self.act_step += 1
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]  # all of your units

        # FACTORIES
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if self.act_step == 2:
                self.build_robot("heavy", factory, actions, unit_id, game_state)
            if self.act_step == 14 or self.act_step == 24 or self.act_step == 300 or self.act_step == 600:
                self.build_robot("light", factory, actions, unit_id, game_state)

            if factory.cargo.water > 50 and game_state.env_steps > 750:
                actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]

        factory_tiles = np.array(factory_tiles)  # the locations of your factories

        # UNITS
        for unit_id, unit in units.items():
            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            closest_f = factory_units[np.argmin(factory_distances)]

            if unit.unit_type == "HEAVY":
                self.heavy_actions(step, unit, closest_f, actions, game_state, obs)

            elif unit.unit_type == "LIGHT":
                self.light_actions(unit, closest_f, actions, game_state, obs)

        return actions
