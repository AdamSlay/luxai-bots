from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.actions import deliver_to_factory
from lux.utils import *  # it's ok, these are just helper functions
import numpy as np
import sys


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.act_step = 0
        self.opp_strains = []
        self.strains = []
        self.new_positions = []
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    # ------------------- Factory Setup -------------------
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
                ice_distances = [manhattan_dist_to_nth_closest(ice, i) for i in range(1, 5)]
                ore_distances = [manhattan_dist_to_nth_closest(ore, i) for i in range(1, 5)]
                ICE_WEIGHTS = np.array([1, 0.8, 0, 0])
                weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ORE_WEIGHTS = np.array([0, 0, 0, 0])
                weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ICE_PREFERENCE = 10  # if you want to make ore more important, change to 0.3 for example

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
                overall_score = (low_rubble_scores + (combined_score * 5)) * obs["board"]["valid_spawns_mask"]

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

    def mine_ice_heavy(self, resource, unit, closest_f, actions, game_state, obs) -> None:
        target_tile = closest_type_tile(resource, unit, self.player, self.opp_player, game_state, obs, factory=True)
        if np.all(target_tile == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                digs = (unit.power - unit.action_queue_cost(game_state)) // (unit.dig_cost(game_state))
                actions[unit.unit_id] = [unit.dig(n=digs)]
        else:
            self.move_toward(target_tile, unit, actions, game_state)

    def move_toward(self, target_tile, unit, actions, game_state) -> None:
        direction = direction_to(unit.pos, target_tile)
        opp_factories = get_factory_tiles(game_state, self.opp_player)
        unit_positions = [u.pos for u in game_state.units[self.player].values() if u.unit_id != unit.unit_id]
        unit_positions.extend([u.pos for u in game_state.units[self.opp_player].values()])
        unit_positions.extend(opp_factories)
        unit_positions.extend(self.new_positions)
        unit_positions = unit_positions

        next_pos = next_position(unit, direction)
        for u in unit_positions:
            if next_pos[0] == u[0] and next_pos[1] == u[1]:
                new_direction = find_new_direction(unit, unit_positions, game_state)
                new_new_pos = next_position(unit, new_direction)
                self.new_positions.append(new_new_pos)
                actions[unit.unit_id] = [unit.move(new_direction, repeat=0)]
                return

        self.new_positions.append(next_pos)
        actions[unit.unit_id] = [unit.move(direction, repeat=0)]

    def attack_opp(self, unit, opp_lichen, actions, game_state) -> int or None:
        target_tile = closest_opp_lichen(opp_lichen, unit, self.player, self.opp_player, game_state)
        if target_tile is None:
            return None

        if np.all(target_tile == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
                digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                actions[unit.unit_id] = [unit.dig(repeat=True, n=digs)]
                return 1
        else:
            self.move_toward(target_tile, unit, actions, game_state)
            return 1

    def dig_rubble(self, unit, actions, game_state, obs):
        target_tile = closest_type_tile("rubble", unit, self.player, self.opp_player, game_state, obs)
        if np.all(target_tile == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
                digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                actions[unit.unit_id] = [unit.dig(repeat=True, n=digs)]
        else:
            self.move_toward(target_tile, unit, actions, game_state)

    def deliver_payload(self, unit, resource, closest_f, actions, game_state):
        direction = direction_to(unit.pos, closest_f.pos)
        adjacent_to_factory = factory_adjacent(closest_f.pos, unit)
        if adjacent_to_factory:
            actions[unit.unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, n=1)]
        else:
            self.move_toward(closest_f.pos, unit, actions, game_state)

    def recharge(self, unit, closest_f, actions, game_state):
        adjacent_to_factory = factory_adjacent(closest_f.pos, unit)
        if adjacent_to_factory:
            if unit.unit_type == "LIGHT":
                if closest_f.power < 500:
                    pickup_amt = 100
                elif closest_f.power < 1000:
                    pickup_amt = 200
                elif closest_f.power < 2000:
                    pickup_amt = 300
                else:
                    pickup_amt = 1000 + (closest_f.power % 1000)
            else:
                # if unit.cargo.ice > 0:
                #     self.deliver_payload(unit, "ice", closest_f, actions, game_state)
                #     return
                if closest_f.power < 1000:
                    pickup_amt = (closest_f.power - 50)
                elif closest_f.power < 3000:
                    pickup_amt = 1000 + ((closest_f.power - 1000) // 2)
                else:
                    pickup_amt = 2000 + (closest_f.power % 1000)

            actions[unit.unit_id] = [unit.pickup(4, pickup_amt, n=1)]
        else:
            self.move_toward(closest_f.pos, unit, actions, game_state)

    def heavy_actions(self, step, unit, closest_f, actions, game_state, obs):
        if self.act_step < 3:
            if unit.unit_id not in actions:
                if unit.power < 500:
                    actions[unit.unit_id] = [unit.pickup(4, 500, n=1)]

        if unit.power < 121:
            return
        elif unit.power < 200:
            self.recharge(unit, closest_f, actions, game_state)

        else:
            if unit.cargo.ice > 200 and closest_f.cargo.water < 100:
               self.deliver_payload(unit, unit.cargo.ice, closest_f, actions, game_state)

            elif unit.cargo.ice > 900:
                self.deliver_payload(unit, unit.cargo.ice, closest_f, actions, game_state)

            elif unit.unit_id not in actions:
                self.mine_ice_heavy("ice", unit, closest_f, actions, game_state, obs)

    def light_actions(self, unit, closest_f, actions, game_state, obs):
        if unit.power < 7:
            return

        if unit.power < 20:
            self.recharge(unit, closest_f, actions, game_state)

        elif self.act_step < 850:
            if closest_f.cargo.water < 200:
                # TODO: mine ice
                pass
            if unit.unit_id not in actions:
                self.dig_rubble(unit, actions, game_state, obs)
        else:
            opp_lichen, my_lichen = [], []
            for i in self.opp_strains:
                opp_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            for i in self.strains:
                my_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            if np.sum(opp_lichen) > (np.sum(my_lichen) * 0.4):
                self.attack_opp(unit, opp_lichen, actions, game_state)
            else:
                self.dig_rubble(unit, actions, game_state, obs)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # SETUP
        self.act_step += 1
        self.new_positions = []
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]  # all of your units

        if self.act_step == 5:
            opp_factories = game_state.factories[self.opp_player]
            for u_id, factory in opp_factories.items():
                self.opp_strains.append(factory.strain_id)
            for u_id, factory in factories.items():
                self.strains.append(factory.strain_id)
        # FACTORIES
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if self.act_step == 2:
                self.build_robot("heavy", factory, actions, unit_id, game_state)
            elif self.act_step == 100 or self.act_step == 104 or self.act_step == 108 or self.act_step == 112 or self.act_step == 116:
                self.build_robot("light", factory, actions, unit_id, game_state)

            elif self.act_step > 800 and factory.cargo.water > 50:
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
