from math import floor

from lux.dijkstra import dijkstras_path
from lux.inventory import Inventory
from lux.kit import obs_to_game_state, EnvConfig
from lux.spawn import SpawnSpot
from lux.utils import *  # it's ok, these are just helper functions


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.act_step = 0
        self.old_units = []
        self.inventory = Inventory([], dict(), dict(), dict())
        self.homers = []
        self.actions = dict()
        self.prev_actions = dict()

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
            return dict(faction="TheBuilders", bid=5)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)

            if factories_to_place > 0 and my_turn_to_place:
                # TODO: place factories according to the order of the spawn spots
                # TODO: the first factory placement should value ore somewhat
                # TODO: the last factory placement should only value ice
                ice = obs["board"]["ice"]
                ore = obs["board"]["ore"]
                ice_distances = [manhattan_dist_to_nth_closest(ice, i) for i in range(1, 5)]
                ore_distances = [manhattan_dist_to_nth_closest(ore, i) for i in range(1, 5)]
                ICE_WEIGHTS = np.array([1, 0, 0, 0])
                weighted_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ORE_WEIGHTS = np.array([0, 0, 0, 0])
                weighted_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

                ICE_PREFERENCE = 7  # if you want to make ore more important, change to 0.3 for example

                low_rubble = (obs["board"]["rubble"] < 25)

                def count_region_cells(array, start, min_dist=2, max_dist=np.inf, exponent=1.0):

                    def dfs(array, loc):
                        distance_from_start = abs(loc[0] - start[0]) + abs(loc[1] - start[1])
                        if not (0 <= loc[0] < array.shape[0] and 0 <= loc[1] < array.shape[1]):
                            return 0
                        if (not array[loc]) or visited[loc]:
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

                combined_score = (weighted_ice_dist * ICE_PREFERENCE + weighted_ore_dist)
                combined_score = (np.max(combined_score) - combined_score) * obs["board"]["valid_spawns_mask"]
                overall_score = (low_rubble_scores + (combined_score * 7)) * obs["board"]["valid_spawns_mask"]

                best_loc = np.argmax(overall_score)
                x, y = np.unravel_index(best_loc, (48, 48))
                spawn_loc = (x, y)
                spot = SpawnSpot(spawn_loc)
                m, w = 150, 150  # metal, water
                if water_left % 10 != 0:
                    closest_ore = closest_type_tile("ore", spot, self.player, self.opp_player, game_state, obs)
                    dist_to_ore = self.distance_to(spot.pos, closest_ore)
                    if dist_to_ore <= 10:
                        m, w = 145, 145
                if factories_to_place == 1:
                    m, w = metal_left, water_left
                return dict(spawn=spawn_loc, metal=m, water=w)
            return dict()

    # ------------------- Agent Behavior -------------------
    def build_factory(self, player):
        print(f"Step {self.act_step}: {player.id} Building factory")
        if player.metal >= self.env_cfg.FACTORY_CHARGE:
            player.build_factory()

    def build_robot(self, unit_type: str, factory, unit_id, game_state) -> bool:
        unit_type = unit_type.upper()
        if unit_type == "LIGHT" and factory.can_build_light(game_state):
            queue = factory.build_light()
            self.update_actions(unit_id, queue)
            return True
        elif unit_type == "HEAVY" and factory.can_build_heavy(game_state):
            queue = factory.build_heavy()
            self.update_actions(unit_id, queue)
            return True
        return False

    def move_toward(self, target_tile, unit, game_state, evading=False) -> None:
        # TODO: Move_toward can be replaced by a path_to function that returns dijkstra path to the target tile
        # TODO: This unit_positions code is what can be replaced by self.occupied_next_step
        o_facto = [u.pos for u in game_state.factories[self.opp_player].values()]
        opp_factories = get_factory_tiles(o_facto)
        unit_positions = [u.pos for u in game_state.units[self.player].values() if u.unit_id != unit.unit_id]
        unit_positions.extend(opp_factories)
        unit_positions.extend(self.new_positions)

        direction = direction_to(unit.pos, target_tile)
        if not evading:
            unit_positions.extend([u.pos for u in game_state.units[self.opp_player].values()])
        next_pos = next_position(unit, direction)
        for u in unit_positions:
            # TODO: this method is really bad, keep everything in np arrays and use 'in'
            if next_pos[0] == u[0] and next_pos[1] == u[1]:
                new_direction = find_new_direction(unit, unit_positions, game_state)
                new_new_pos = next_position(unit, new_direction)
                # TODO: never figured out why this cost calculation would sometimes be None
                if unit.move_cost(game_state, direction) is not None and unit.action_queue_cost(game_state) is not None:
                    cost = unit.move_cost(game_state, direction) + unit.action_queue_cost(game_state)
                elif unit.unit_type == "LIGHT":
                    cost = 8
                else:
                    cost = 30
                if unit.power >= cost:
                    # TODO: update_actions should be called in light_actions and heavy_actions
                    # TODO: these methods should return a queue of actions
                    self.new_positions.append(new_new_pos)
                    queue = [unit.move(new_direction, repeat=0)]
                    self.update_actions(unit, queue)
                    return
                else:
                    unit.recharge(x=cost)
                    return
        if unit.move_cost(game_state, direction) is not None and unit.action_queue_cost(game_state) is not None:
            cost = unit.move_cost(game_state, direction) + unit.action_queue_cost(game_state)
        elif unit.unit_type == "LIGHT":
            cost = 8
        else:
            cost = 30

        if unit.power >= cost:
            # TODO: update_actions should be called in light_actions or heavy_actions
            # TODO: these methods should return a queue of actions
            self.new_positions.append(next_pos)
            queue = [unit.move(direction, repeat=0)]
            self.update_actions(unit, queue)
            return
        else:
            unit.recharge(x=cost)

    def attack_opp(self, unit, opp_lichen, game_state):
        # TODO: should return a queue of actions
        closest_lichen = closest_opp_lichen(opp_lichen, unit, self.player, self.opp_player, game_state)
        if np.all(closest_lichen == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
                digs = (unit.power - unit.action_queue_cost(game_state) - 30) // (unit.dig_cost(game_state))
                if digs > 20:
                    digs = 20
                elif digs < 1:
                    digs = 1
                queue = [unit.dig(n=digs)]
                self.update_actions(unit, queue)
        else:
            # TODO: replace this with path_to function
            queue = []
            rubble_map = game_state.board.rubble
            o_facto = [u.pos for u in game_state.factories[self.opp_player].values()]
            opp_factories = get_factory_tiles(o_facto)
            unit_positions = [u.pos for u in game_state.units[self.player].values() if u.unit_id != unit.unit_id]
            unit_positions.extend(opp_factories)
            unit_positions.extend(self.new_positions)
            path = []
            path_positions = dijkstras_path(unit.unit_type, rubble_map, unit.pos, closest_lichen, unit_positions)
            for i, pos in enumerate(path_positions):
                if i + 1 < len(path_positions):
                    path.append(self.path_to(unit, path_positions[i], path_positions[i + 1]))
            path = [act[0] for act in path]
            if len(path) > 20:
                path = path[:20]
            queue.extend(path)
            self.update_actions(unit, queue)
            # self.move_toward(closest_lichen, unit, game_state)

    def dig_rubble(self, unit, game_state, obs):
        # TODO: should return a queue of actions
        target_tile = closest_type_tile("rubble", unit, self.player, self.opp_player, game_state, obs)
        if np.all(target_tile == unit.pos):
            # TODO: Finding the number of digs can be it's own function: digs = dig_amount(unit, resource, game_state)
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
                digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                if digs > 20:
                    digs = 20
                queue = []
                for i in range(digs):
                    queue.append(unit.dig(n=1))
                    self.update_actions(unit, queue)
        else:
            # TODO: replace this with path_to function
            self.move_toward(target_tile, unit, game_state)

    def deliver_payload(self, unit, resource: int, amount: int, closest_f, game_state):
        # TODO: should return a queue of actions
        direction = direction_to(unit.pos, closest_f.pos)
        adjacent_to_factory = factory_adjacent(closest_f.pos, unit)
        if adjacent_to_factory:
            queue = [unit.transfer(direction, resource, amount, n=1)]
            self.update_actions(unit, queue)
        else:
            # TODO: replace this with path_to function
            if unit.unit_type == "HEAVY":
                self.move_toward(closest_f.pos, unit, game_state)
            else:
                self.move_toward(closest_f.pos, unit, game_state)

    def recharge(self, unit, home_f, game_state, desired_power=None):
        adjacent_to_factory = factory_adjacent(home_f.pos, unit)
        if adjacent_to_factory:
            if desired_power is not None:
                if desired_power > unit.power:
                    desired_power = unit.power + 50
                self.actions[unit.unit_id] = [unit.pickup(4, desired_power, n=1)]
                return
            if unit.unit_type == "LIGHT":
                pickup_amt = 150 - unit.power
            else:
                if home_f.power < 500:
                    pickup_amt = (home_f.power - 50)
                elif home_f.power < 1000:
                    pickup_amt = 500 + ((home_f.power - 500) // 2)
                elif home_f.power < 3000:
                    pickup_amt = 1000 + ((home_f.power - 1000) // 2)
                else:
                    pickup_amt = 2000 + (home_f.power % 1000)
            queue = [unit.pickup(4, pickup_amt, n=1)]
            self.update_actions(unit, queue)
        else:
            # TODO: replace this with path_to function
            self.move_toward(home_f.pos, unit, game_state)

    def retreat(self, unit, opp_unit, home_f, game_state):
        adjacent_to_factory = self.distance_to(unit.pos, home_f.pos)
        if adjacent_to_factory <= 1:
            desired_pwr = (opp_unit.power + 50) - unit.power
            self.actions[unit.unit_id] = [unit.pickup(4, desired_pwr, n=1)]
        else:
            direction = direction_to(unit.pos, home_f.pos)
            if unit.power >= unit.move_cost(game_state, direction) + unit.action_queue_cost(game_state):
                next_pos = next_position(unit, direction)
                self.new_positions.append(next_pos)
                queue = [unit.move(direction, repeat=0)]
                self.update_actions(unit, queue)

    # def mine_tile(self, the_tile, unit, game_state) -> None:
    #     # TODO: is this really necessary with the mining_queue functions?
    #     if np.all(the_tile == unit.pos):
    #         if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
    #             digs = ((unit.power - unit.action_queue_cost(game_state) - 10) // (unit.dig_cost(game_state)))
    #             if digs > 20:
    #                 digs = 20
    #             queue = [unit.dig(n=digs)]
    #             self.update_actions(unit, queue)
    #             return
    #     else:
    #         # TODO: replace this with path_to function
    #         self.move_toward(the_tile, unit, game_state)

    def path_to(self, unit, start, finish) -> list:
        # TODO: this is just a rise over run, not a true path
        # TODO: also it belongs in utils if it belongs anywhere
        # dy / dx
        y = finish[1] - start[1]
        x = finish[0] - start[0]

        path = []
        if y < 0:  # move up
            for i in range(abs(y)):
                path.append(unit.move(1, repeat=0))
        elif y > 0:  # move down
            for i in range(abs(y)):
                path.append(unit.move(3, repeat=0))
        if x < 0:  # move left
            for i in range(abs(x)):
                path.append(unit.move(4, repeat=0))
        elif x > 0:  # move right
            for i in range(abs(x)):
                path.append(unit.move(2, repeat=0))
        return path

    def distance_to(self, start, finish) -> int:
        # TODO: fine with this, but it belongs in utils
        # dy / dx
        y = finish[1] - start[1]
        x = finish[0] - start[0]
        return abs(x) + abs(y)

    def mining_queue(self, resource, unit, home_f, game_state, obs, sentry=False):
        # TODO: this is a mess, it's way too long and needs to be broken up
        queue = []
        path = []
        pickup_amt = 0
        if sentry is True or unit.unit_type == "LIGHT":
            mining_tile = closest_type_tile(resource, home_f, self.player, self.opp_player, game_state, obs,
                                            this_is_the_unit=unit)
        else:
            mining_tile = closest_type_tile(resource, home_f, self.player, self.opp_player, game_state, obs, heavy=True,
                                            this_is_the_unit=unit)
        # TODO: all this to find the closest factory tile. Maybe there is a better way?
        tile_locations = get_factory_tiles([home_f.pos])
        tile_distances = np.mean((tile_locations - mining_tile) ** 2, 1)
        factory_tile = tile_locations[np.argmin(tile_distances)]
        adjacent_to_factory = factory_adjacent(factory_tile, unit)
        on_factory = factory_adjacent(home_f.pos, unit)

        if resource == "ice":
            res_type = 0
            cargo = unit.cargo.ice
        else:
            res_type = 1
            cargo = unit.cargo.ore
        if cargo > 0 and adjacent_to_factory:
            direction = direction_to(unit.pos, factory_tile)
            transfer = unit.transfer(direction, res_type, cargo, n=1)
            queue.append(transfer)

        # TODO: This is basically just self.recharge(). Maybe self.recharge() returns a queue?
        if game_state.real_env_steps > 10 and on_factory:
            if unit.unit_type == "LIGHT" and unit.power < 100:
                pickup_amt = 150 - unit.power
            elif unit.unit_type == "HEAVY":
                if home_f.power < 1000:
                    pickup_amt = (home_f.power - 50)
                elif home_f.power < 3000:
                    pickup_amt = 1000 + ((home_f.power - 1000) // 2)
                else:
                    pickup_amt = 2000 + (home_f.power % 1000)
            if pickup_amt > 0:
                pickup = unit.pickup(4, pickup_amt, n=1)
                queue.append(pickup)

        # TODO: this is just short circuiting dijkstra's algorithm if the unit is a heavy.
        # TODO: either remove it altogether or implement it for all units traveling less than x tiles
        if unit.unit_type == "HEAVY":
            if unit.pos[0] != mining_tile[0] or unit.pos[1] != mining_tile[1]:
                path = self.path_to(unit, unit.pos, mining_tile)
                queue.extend(path)
            path_back = self.path_to(unit, mining_tile, factory_tile)

        # TODO: here is Dijkstra's algorithm. Returning the path queue needs to be it's own function
        # TODO: Also shouldn't have to crunch unit_positions once self.occupied_next_step() is implemented
        # TODO: this can go in pathing.py or similar(along with Dijkstra's and path_cost)
        else:
            rubble_map = game_state.board.rubble
            o_facto = [u.pos for u in game_state.factories[self.opp_player].values()]
            opp_factories = get_factory_tiles(o_facto)
            unit_positions = [u.pos for u in game_state.units[self.player].values() if u.unit_id != unit.unit_id]
            unit_positions.extend(opp_factories)
            unit_positions.extend(self.new_positions)
            if unit.pos[0] != mining_tile[0] or unit.pos[1] != mining_tile[1]:
                path = []
                path_positions = dijkstras_path(unit.unit_type, rubble_map, unit.pos, mining_tile, unit_positions)
                for i, pos in enumerate(path_positions):
                    if i + 1 < len(path_positions):
                        path.append(self.path_to(unit, path_positions[i], path_positions[i + 1]))
                path = [act[0] for act in path]
                queue.extend(path)

            path_back = []
            path_back_positions = dijkstras_path(unit.unit_type, rubble_map, mining_tile, factory_tile, unit_positions)

            for i, pos in enumerate(path_back_positions):
                if i + 1 < len(path_back_positions):
                    path_back.append(self.path_to(unit, path_back_positions[i], path_back_positions[i + 1]))
            path_back = [act[0] for act in path_back]

        # TODO: path_cost() should be it's own function. Can be kept in pathing.py
        if unit.unit_type == "LIGHT":
            path_cost = 0
            step = game_state.real_env_steps
            for i in range(len(path) + len(path_back)):
                if step % 50 >= 30:
                    path_cost += 1.1
                step += 1
            path_cost = floor(path_cost) + 31
        else:
            path_cost = (len(path) + len(path_back)) * 20

        # TODO: this can be it's own function. utils maybe pathing, idk
        num_digs = ((unit.power - unit.action_queue_cost(game_state) - path_cost + (pickup_amt // 2)) // (
            unit.dig_cost(game_state)))
        if num_digs > 20 - len(queue):
            num_digs = 20 - len(queue)
        for i in range(num_digs):
            queue.append(unit.dig(n=1))
        if num_digs < 1:
            queue = []

        # TODO: come up with some logic that makes sure the unit is not going to be stuck with no power
        if len(queue) > 20 - len(path_back):
            path_back_steps = 20 - len(queue)
            if path_back_steps > 0:
                path_back = path_back[:path_back_steps]
            else:
                path_back = []
        queue.extend(path_back)
        if len(queue) > 20:  # just in case
            queue = queue[:20]

        return queue

    def update_unit_positions(self, units, game_state):
        # TODO: we should update self.occupied_next_step() with the new positions that we get from self.current_actions
        # TODO: this occurs after resolving any collisions among queued actions
        for uid, act in self.prev_actions.items():
            if not isinstance(act, int) and len(act) > 0 and act[0][0] == 0:  # it's a move command
                unit = units.get(uid)
                if unit is not None:
                    new_pos = next_position(unit, act[0][1])
                    for pos in self.new_positions:
                        if new_pos[0] == pos[0] and new_pos[1] == pos[1]:
                            self.prev_actions[uid] = []
                            break
                    else:
                        self.new_positions.append(new_pos)

    def update_actions(self, unit, queue):
        # TODO: simple and effective. I like it, but prev_actions should be current_actions
        self.actions[unit.unit_id] = queue
        self.prev_actions[unit.unit_id] = queue

    def heavy_actions(self, unit, title, home_f, game_state, obs):
        adjacent_to_factory = factory_adjacent(home_f.pos, unit)
        # TODO: something is going ary with heavies running out of power. That needs to be fixed
        if unit.power < 30 and not adjacent_to_factory:
            self.prev_actions[unit.unit_id] = []
            return
        elif unit.power < 100:
            self.prev_actions[unit.unit_id] = []
            self.recharge(unit, home_f, game_state)
            return
        if game_state.board.rubble[unit.pos[0]][unit.pos[1]] > 0:
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                self.actions[unit.unit_id] = [unit.dig(n=1)]
                return
            else:
                return

        # TODO: mine some ore if in position to do so, this should be its own function
        factory_inv = len(self.inventory.factory_units[home_f.unit_id])
        step = game_state.real_env_steps
        if factory_inv <= 3 and home_f.cargo.water > 500 and home_f.cargo.metal < 40 and step < 700:
            closest_ore = closest_type_tile("ore", home_f, self.player, self.opp_player, game_state, obs,
                                            this_is_the_unit=unit)
            dist_to_ore = self.distance_to(home_f.pos, closest_ore)
            if dist_to_ore < 10 and len(self.prev_actions[unit.unit_id]) == 0:
                print(f"Step {game_state.real_env_steps}: {title} {unit.unit_id} is in position to mine ore",
                      file=sys.stderr)
                queue = self.mining_queue("ore", unit, home_f, game_state, obs)
                self.update_actions(unit, queue)
                return

        if home_f.cargo.water < 100 < unit.cargo.ice:  # didn't know you could do this chained comparison
            # TODO: I like the idea of everything returning a queue, then update_actions() is called.
            # TODO: should do that here. Also, makes it easier to debug/test
            self.deliver_payload(unit, 0, unit.cargo.ice, home_f, game_state)
            return

        if factory_adjacent(home_f.pos, unit) and unit.cargo.ore > 0:
            direction = direction_to(unit.pos, home_f.pos)
            transfer_ore = [unit.transfer(direction, 1, unit.cargo.ore, n=1)]
            self.update_actions(unit, transfer_ore)
            return

        elif unit.cargo.ice < 1000 and len(self.prev_actions[unit.unit_id]) == 0:
            # TODO: method returns a queue then update_actions() is called, this is what I'm talking about
            queue = self.mining_queue("ice", unit, home_f, game_state, obs)
            self.update_actions(unit, queue)
            return

        elif len(self.prev_actions[unit.unit_id]) == 0:
            factory_tiles = get_factory_tiles([home_f.pos])
            closest_factory_tile = closest_factory(factory_tiles, factory_tiles, unit, game_state)
            if factory_adjacent(closest_factory_tile, unit):
                queue = [unit.transfer(direction_to(unit.pos, home_f.pos), 0, unit.cargo.ice, n=1)]
                self.update_actions(unit, queue)
                return
            else:
                # TODO: should return a queue, then update_actions() is called
                self.deliver_payload(unit, 0, unit.cargo.ice, home_f, game_state)

    def light_actions(self, unit, title, home_f, game_state, obs):
        for pos in self.new_positions:
            if unit.pos[0] == pos[0] and unit.pos[1] == pos[1]:
                target_tile = closest_type_tile("rubble", unit, self.player, self.opp_player, game_state, obs)
                self.move_toward(target_tile, unit, game_state)
                return

        # COLLISION AVOIDANCE FOR QUEUED ACTIONS
        # TODO: all of this can be handled in update_unit_positions() and stored in self.occupied_next_step
        # TODO: then any unit that needs this logic will have already had it before it gets here
        if unit.unit_id in self.prev_actions.keys():
            if len(self.prev_actions[unit.unit_id]) > 0:
                next_act = self.prev_actions[unit.unit_id][0]
                if next_act[0] == [0]:  # it's a move command
                    direction = next_act[1]
                    new_pos = next_position(unit, direction)
                    unit_positions = [unit.pos for uid, unit in game_state.units[self.player].items() if
                                      uid != unit.unit_id]
                    unit_positions.extend(self.new_positions)
                    miner_count = 0
                    for pos in unit_positions:
                        if pos[0] == new_pos[0] and pos[1] == new_pos[1]:
                            # if title == "miner" and miner_count < 1:
                            #     miner_count += 1
                            #     continue
                            self.prev_actions[unit.unit_id] = []
                            return

        if unit.power < 8 and not factory_adjacent(home_f.pos, unit):
            # TODO: really need to dial in power handling
            return

        if unit.power < 30:
            if game_state.real_env_steps >= 900:
                factories = game_state.factories[self.player]
                factory_tiles = np.array([factory.pos for factory_id, factory in factories.items()])
                factory_units = [factory for factory_id, factory in factories.items()]
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_f = factory_units[np.argmin(factory_distances)]
                self.recharge(unit, closest_f, game_state)
                return
            self.recharge(unit, home_f, game_state)
            return

        if game_state.real_env_steps >= 900:
            # TODO: this should be it's own function that returns a queue, then update_actions() is called
            # TODO: if the queue is None, move on to the next decision check
            # If the closest rubble is not relevant at this point and opp has lichen, attack it
            if unit.power < 150 and factory_adjacent(home_f.pos, unit):
                pickup_amt = 150 - unit.power
                self.actions[unit.unit_id] = [unit.pickup(4, pickup_amt, n=1)]
                return
            opp_lichen, my_lichen = [], []
            for i in self.opp_strains:
                opp_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            for i in self.strains:
                my_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            if np.sum(opp_lichen) > np.sum(my_lichen) * 0.4:
                closest_rubble = closest_type_tile("rubble", unit, self.player, self.opp_player, game_state, obs)
                dist_to_close_rub = self.distance_to(unit.pos, closest_rubble)
                closest_lichen = closest_opp_lichen(opp_lichen, home_f, self.player, self.opp_player, game_state)
                if self.distance_to(unit.pos, closest_lichen) < 12 and dist_to_close_rub > 8 and len(
                        self.prev_actions[unit.unit_id]) == 0:
                    self.attack_opp(unit, opp_lichen, game_state)
                    return

        if unit.cargo.ore > 0 and title != "miner":
            # TODO: should return a queue, then update_actions() is called
            self.deliver_payload(unit, 1, unit.cargo.ore, home_f, game_state)
            return

        if title == "miner" and game_state.real_env_steps < 900:
            # TODO: this should be it's own function that returns a queue, then update_actions() is called
            # TODO: if the queue is None, move on to the next decision check
            home_unit_inv = len(self.inventory.factory_units[home_f.unit_id])
            if unit.cargo.ore >= 25 and home_unit_inv < 4:
                self.deliver_payload(unit, 1, unit.cargo.ore, home_f, game_state)
                return
            if unit.cargo.ore > 98:
                self.deliver_payload(unit, 1, unit.cargo.ore, home_f, game_state)
                return

            closest_ore = closest_type_tile("ore", home_f, self.player, self.opp_player, game_state, obs,
                                            this_is_the_unit=unit)
            dist_to_ore = self.distance_to(home_f.pos, closest_ore)
            if dist_to_ore < 16:  # and home_unit_inv < 7
                rubble_here = game_state.board.rubble[unit.pos[0]][unit.pos[1]]
                if rubble_here > 0:
                    digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                    if digs > 20:
                        digs = 20
                    queue = [unit.dig(n=digs)]
                    self.update_actions(unit, queue)
                    return
                elif unit.cargo.ore <= 98 and len(self.prev_actions[unit.unit_id]) == 0:
                    queue = self.mining_queue("ore", unit, home_f, game_state, obs)
                    # the first action will be popped off before the next step, so you need to add to self.new_positions
                    # now so that subsequent units on this step don't collide
                    if len(queue) > 0 and queue[0][0] == [0]:  # it's a move command
                        new_q_pos = next_position(unit, queue[0][1])
                        self.new_positions.append(new_q_pos)
                    self.update_actions(unit, queue)
                    return
                return

        if title == "digger" and game_state.real_env_steps > 700:
            # TODO: this should be it's own function that returns a queue, then update_actions() is called
            # TODO: if the queue is None, move on to the next decision check
            opp_lichen, my_lichen = [], []
            for i in self.opp_strains:
                opp_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            for i in self.strains:
                my_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            if np.sum(opp_lichen) > np.sum(my_lichen) * 0.4:
                closest_lichen = closest_opp_lichen(opp_lichen, home_f, self.player, self.opp_player, game_state)
                if self.distance_to(unit.pos, closest_lichen) < 12 and game_state.real_env_steps < 900:
                    self.attack_opp(unit, opp_lichen, game_state)
                    return
                elif game_state.real_env_steps >= 900:
                    self.attack_opp(unit, opp_lichen, game_state)
                    return

        if len(self.actions[unit.unit_id]) == 0:
            rubble_here = game_state.board.rubble[unit.pos[0]][unit.pos[1]]
            if rubble_here > 0:
                # TODO: dig_amount should be its own function that returns an int
                digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                if digs > 20:
                    digs = 20
                queue = [unit.dig(n=digs)]
                self.update_actions(unit, queue)
                return
            # TODO: this should be it's own function that returns a queue, then update_actions() is called
            self.dig_rubble(unit, game_state, obs)
            return

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # SETUP
        # TODO: this is where we reset self.actions_to_submit and self.occupied_next_step
        self.act_step += 1
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]  # all of your units
        factory_tiles = np.array([factory.pos for factory_id, factory in factories.items()])
        factory_units = [factory for factory_id, factory in factories.items()]
        self.new_positions = [list(f.pos) for fid, f in factories.items()]
        self.inventory.factory_types = dict()

        # STRAINS
        # TODO: only needs to be done once per game, this seems fine. Create a function for this and store in utils
        if game_state.real_env_steps == 1:
            opp_factories = game_state.factories[self.opp_player]
            for u_id, factory in opp_factories.items():
                self.opp_strains.append(factory.strain_id)
            for u_id, factory in factories.items():
                self.strains.append(factory.strain_id)

        # UPDATE ACTION QUEUE
        # TODO: we should be updating self.occupied_next_step and resolving collisions among queued actions
        # TODO: any unit that maintains a queue should skip the rest of decision making and just execute the next action
        # TODO: EXCEPT for deciding whether to evade from an enemy unit
        new_acts = dict()
        for uid, acts in self.prev_actions.items():
            if isinstance(acts, list):
                new_acts[uid] = acts[1:]
            elif isinstance(acts, int):
                continue
        self.prev_actions = new_acts
        self.update_unit_positions(units, game_state)
        self.actions = dict()

        # UNITS
        for unit_id, unit in units.items():
            # SETUP
            if unit.unit_id not in self.actions.keys():
                # TODO: this should be called something more descriptive. self.actions_to_submit?
                self.actions[unit.unit_id] = []
            if unit.unit_id not in self.prev_actions.keys():
                self.prev_actions[unit.unit_id] = []

            # TODO: since we calculate this for every unit, we should store it in a dict: self.closest_factory[unit_id]
            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            closest_f = factory_units[np.argmin(factory_distances)]
            if unit_id not in self.inventory.all_units:  # then it's new and needs to be added to inventory
                self.inventory.factory_units[closest_f.unit_id].append(unit_id)
                self.inventory.all_units.append(unit_id)

            # TODO: this is basically just "new unit" logic, or when a unit loses its home factory
            home_id = [f_id for f_id, inv in self.inventory.factory_units.items() if unit_id in inv]
            home_id = str(home_id[0])
            if home_id in factories.keys():
                home_factory = factories[home_id]
            else:
                home_id = closest_f.unit_id
                home_factory = closest_f

            # ATTACK EVASION
            # TODO: this should be its own function that returns a queue, then update_actions() if returns True
            evading = False
            for u_id, u in game_state.units[self.opp_player].items():
                o_facto = [op.pos.tolist() for op in game_state.factories[self.opp_player].values()]
                if unit.unit_type == "HEAVY" and unit.power >= 40:
                    attacker_dist = self.distance_to(unit.pos, u.pos)
                    if attacker_dist < 2 and u.unit_type == "HEAVY":
                        if u.pos.tolist() not in o_facto:  # if the enemy is not on a factory
                            if unit.power <= u.power:
                                self.retreat(unit, u, home_factory, game_state)
                                evading = True
                                break
                            # elif attacker_dist <= 1:
                            else:
                                self.move_toward(u.pos, unit, game_state, evading=True)
                                evading = True
                                break
                elif unit.unit_type == "LIGHT":
                    attacker_dist = self.distance_to(unit.pos, u.pos)
                    if attacker_dist <= 1:
                        if unit.power < u.power or unit.power < 20:
                            self.recharge(unit, home_factory, game_state)
                            evading = True
                            break
                        else:
                            if u.pos.tolist() not in o_facto:
                                self.move_toward(u.pos, unit, game_state, evading=True)
                                evading = True
                                break
            if evading:
                continue

            # HEAVY
            if unit.unit_type == "HEAVY":
                # TODO: more new unit logic, maybe make a separate function for this?
                if home_id not in self.inventory.factory_types.keys():
                    self.inventory.factory_types[home_id] = []
                home_homers = self.inventory.factory_types[home_id].count("homer")
                if home_homers == 0:
                    self.inventory.unit_title[unit_id] = "homer"
                    self.inventory.factory_types[home_id].append("homer")
                else:
                    self.inventory.unit_title[unit_id] = "sentry"
                    self.inventory.factory_types[home_id].append("sentry")

                title = self.inventory.unit_title[unit_id]
                self.heavy_actions(unit, title, home_factory, game_state, obs)

            # LIGHT
            elif unit.unit_type == "LIGHT":
                # TODO: bunch of new unit logic, again maybe make a separate function for this?
                if home_id not in self.inventory.factory_types.keys():
                    self.inventory.factory_types[home_id] = []
                home_helpers = self.inventory.factory_types[home_id].count("helper")
                home_miners = self.inventory.factory_types[home_id].count("miner")
                if home_miners < 1:
                    title = "miner"
                    self.inventory.unit_title[unit_id] = "miner"
                    self.inventory.factory_types[home_id].append("miner")
                    if unit_id not in self.inventory.factory_units[home_id]:
                        self.inventory.factory_units[home_id].append(unit_id)
                elif home_helpers < 2:
                    title = "helper"
                    self.inventory.unit_title[unit_id] = "helper"
                    self.inventory.factory_types[home_id].append("helper")
                    if unit_id not in self.inventory.factory_units[home_id]:
                        self.inventory.factory_units[home_id].append(unit_id)
                else:
                    title = "digger"
                    self.inventory.unit_title[unit_id] = "digger"
                    self.inventory.factory_types[home_id].append("digger")
                    if unit_id not in self.inventory.factory_units[home_id]:
                        self.inventory.factory_units[home_id].append(unit_id)
                self.light_actions(unit, title, home_factory, game_state, obs)

        # FACTORIES
        for unit_id, factory in factories.items():
            # TODO: this is basically just "new factory" logic, maybe make self.new_factory_setup() function?
            if unit_id not in self.inventory.factory_units.keys():
                self.inventory.factory_units[unit_id] = []
            if unit_id not in self.inventory.factory_types.keys():
                self.inventory.factory_types[unit_id] = []
            else:  # check if this factory's units exist in the game_state
                self.inventory.factory_units[unit_id] = [uid for uid in self.inventory.factory_units[unit_id] if
                                                         uid in units.keys()]

            number_of_homers = self.inventory.factory_types[unit_id].count("homer")
            if number_of_homers == 0:
                if factory.can_build_heavy(game_state):
                    self.actions[unit_id] = factory.build_heavy()
                    continue
            else:
                number_of_helpers = self.inventory.factory_types[unit_id].count("helper")
                number_of_diggers = self.inventory.factory_types[unit_id].count("digger")
                number_of_miners = self.inventory.factory_types[unit_id].count("miner")
                if number_of_helpers < 2 and self.act_step % 4 == 0:
                    if factory.can_build_light(game_state):
                        self.actions[unit_id] = factory.build_light()
                        continue
                elif number_of_miners < 1 and self.act_step % 4 == 0:
                    if factory.can_build_light(game_state):
                        self.actions[unit_id] = factory.build_light()
                        continue
                elif number_of_diggers < 3 and self.act_step % 4 == 0:
                    if factory.can_build_light(game_state):
                        self.actions[unit_id] = factory.build_light()
                        continue
                elif game_state.real_env_steps > 800 and factory.can_build_light(
                        game_state) and game_state.real_env_steps % 10 == 0:
                    self.actions[unit_id] = factory.build_light()
                    continue

            if factory.cargo.water > 50 and self.act_step > 780:
                self.actions[unit_id] = factory.water()

        # FINALIZE ACTIONS
        # TODO: self.actions could more accurately be called self.actions_to_submit or similar
        actions_to_submit = dict()
        for uid, acts in self.actions.items():
            if isinstance(acts, list):
                if len(acts) > 0:  # only submit actions that are not empty
                    actions_to_submit[uid] = acts
            else:  # it's an int and therefore a factory action, so submit it
                actions_to_submit[uid] = acts
        return actions_to_submit
