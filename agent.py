from lux.inventory import Inventory
from lux.kit import obs_to_game_state, EnvConfig
from lux.utils import *  # it's ok, these are just helper functions


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.act_step = 0
        self.old_units = []
        self.inventory = Inventory([], dict(), dict(), dict(), dict(), dict(), dict())
        self.homers = []
        self.fact_ice_tiles = dict()
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
                # TODO: place factories in a smart way
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
                m, w = 150, 150  # metal, water
                if factories_to_place == 1:
                    m, w = metal_left, water_left - 5
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

    # def mine_ice_heavy(self, resource, unit, closest_f, actions, game_state, obs) -> None:
    #     target_tile = closest_type_tile(resource, unit, self.player, self.opp_player, game_state, obs, heavy=True)
    #     if np.all(target_tile == unit.pos):
    #         if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
    #             digs = (unit.power - unit.action_queue_cost(game_state)) // (unit.dig_cost(game_state))
    #             actions[unit.unit_id] = [unit.dig(n=digs)]
    #     else:
    #         self.move_toward(target_tile, unit, actions, game_state)
    #
    # def mine_ice_light(self, resource, unit, actions, game_state, obs) -> None:
    #     target_tile = closest_type_tile(resource, unit, self.player, self.opp_player, game_state, obs)
    #     if np.all(target_tile == unit.pos):
    #         if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
    #             digs = (unit.power - unit.action_queue_cost(game_state)) // (unit.dig_cost(game_state))
    #             actions[unit.unit_id] = [unit.dig(n=digs)]
    #     else:
    #         self.move_toward(target_tile, unit, actions, game_state)

    def move_toward(self, target_tile, unit, game_state, evading=False) -> None:
        direction = direction_to(unit.pos, target_tile)
        o_facto = [u.pos for u in game_state.factories[self.opp_player].values()]
        opp_factories = get_factory_tiles(o_facto)
        unit_positions = [u.pos for u in game_state.units[self.player].values() if u.unit_id != unit.unit_id]
        unit_positions.extend(opp_factories)
        unit_positions.extend(self.new_positions)
        if not evading:
            unit_positions.extend([u.pos for u in game_state.units[self.opp_player].values()])
        next_pos = next_position(unit, direction)
        for u in unit_positions:
            if next_pos[0] == u[0] and next_pos[1] == u[1]:
                new_direction = find_new_direction(unit, unit_positions, game_state)
                new_new_pos = next_position(unit, new_direction)
                cost = unit.move_cost(game_state, new_direction) + unit.action_queue_cost(game_state)
                if unit.power >= cost:
                    self.new_positions.append(new_new_pos)
                    queue = [unit.move(new_direction, repeat=0)]
                    self.update_actions(unit, queue)
                    return
                else:
                    unit.recharge(x=cost)
                    return
        if direction != 0:
            cost = unit.move_cost(game_state, direction) + unit.action_queue_cost(game_state)
        else:
            cost = unit.action_queue_cost(game_state)
        if unit.power >= cost:
            self.new_positions.append(next_pos)
            queue = [unit.move(direction, repeat=0)]
            self.update_actions(unit, queue)
            return
        else:
            unit.recharge(x=cost)

    # def attack_opp(self, unit, opp_lichen, actions, game_state) -> int or None:
    #     target_tile = closest_opp_lichen(opp_lichen, unit, self.player, self.opp_player, game_state)
    #     if target_tile is None:
    #         return None
    #
    #     if np.all(target_tile == unit.pos):
    #         if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
    #             digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
    #             actions[unit.unit_id] = [unit.dig(repeat=True, n=digs)]
    #             return 1
    #     else:
    #         self.move_toward(target_tile, unit, actions, game_state)
    #         return 1

    def dig_rubble(self, unit, game_state, obs):
        target_tile = closest_type_tile("rubble", unit, self.player, self.opp_player, game_state, obs)
        if np.all(target_tile == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
                digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                if digs > 20:
                    digs = 20
                queue = []
                for i in range(digs):
                    queue.append(unit.dig(n=1))
                    self.update_actions(unit, queue)
        else:
            self.move_toward(target_tile, unit, game_state)

    def deliver_payload(self, unit, resource: int, amount: int, closest_f, game_state):
        direction = direction_to(unit.pos, closest_f.pos)
        adjacent_to_factory = factory_adjacent(closest_f.pos, unit)
        if adjacent_to_factory:
            queue = [unit.transfer(direction, resource, amount, n=1)]
            self.update_actions(unit, queue)
        else:
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
                if home_f.power < 500:
                    pickup_amt = 70
                elif home_f.power < 1000:
                    pickup_amt = 120
                elif home_f.power < 2000:
                    pickup_amt = 300
                else:
                    pickup_amt = 1000 + (home_f.power % 1000)
            else:
                if home_f.power < 1000:
                    pickup_amt = (home_f.power - 50)
                elif home_f.power < 3000:
                    pickup_amt = 1000 + ((home_f.power - 1000) // 2)
                else:
                    pickup_amt = 2000 + (home_f.power % 1000)
            queue = [unit.pickup(4, pickup_amt, n=1)]
            self.update_actions(unit, queue)
        else:
            self.move_toward(home_f.pos, unit, game_state)

    def retreat(self, unit, opp_unit, home_f):
        adjacent_to_factory = factory_adjacent(home_f.pos, unit)
        if adjacent_to_factory:
            desired_pwr = (opp_unit.power + 50) - unit.power
            self.actions[unit.unit_id] = [unit.pickup(4, desired_pwr, n=1)]
        else:
            direction = direction_to(unit.pos, home_f.pos)
            queue = [unit.move(direction, repeat=0)]
            self.update_actions(unit, queue)

    def mine_tile(self, the_tile, unit, game_state, obs, home_f=None) -> None:
        if the_tile == "ice":
            if home_f is not None:
                the_tile = closest_type_tile("ice", home_f, self.player, self.opp_player, game_state, obs, heavy=True,
                                             this_is_the_unit=unit)
            else:
                the_tile = closest_type_tile("ice", unit, self.player, self.opp_player, game_state, obs, heavy=True)
        if np.all(the_tile == unit.pos):
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                digs = ((unit.power - unit.action_queue_cost(game_state) - 10) // (unit.dig_cost(game_state)))
                if digs > 20:
                    digs = 20
                self.actions[unit.unit_id] = []
                queue = []
                for i in range(digs):
                    queue.append(unit.dig(n=1))
                self.update_actions(unit, queue)
                return
        else:
            self.move_toward(the_tile, unit, game_state)

    def mining_queue(self, unit, home_f, game_state, obs):
        pass

    def update_actions(self, unit, queue):
        self.actions[unit.unit_id] = queue
        self.prev_actions[unit.unit_id] = queue

    def heavy_actions(self, unit, title, home_f, game_state, obs):
        if 26 < game_state.real_env_steps % 50 < 31 and unit.power < 1200:
            dp = 1250 - unit.power
            self.recharge(unit, home_f, game_state, desired_power=dp)
            return

        adjacent_to_factory = factory_adjacent(home_f.pos, unit)
        direction_home = direction_to(unit.pos, home_f.pos)
        if direction_home == 0:
            return_cost = unit.action_queue_cost(game_state)
        else:
            return_cost = unit.move_cost(game_state, direction_home) + unit.action_queue_cost(game_state)

        if unit.power < return_cost and not adjacent_to_factory:
            unit.recharge(x=return_cost)
            return

        if unit.power < 80:
            self.recharge(unit, home_f, game_state)
            return

        if home_f.cargo.water < 100 and unit.cargo.ice > 0:
            self.deliver_payload(unit, 0, unit.cargo.ice, home_f, game_state)
            return

        elif unit.cargo.ice < 1000 and len(self.prev_actions[unit.unit_id]) == 0:
            self.mine_tile("ice", unit, game_state, obs, home_f=home_f)
            return

        elif len(self.prev_actions[unit.unit_id]) == 0:
            factory_tiles = get_factory_tiles([home_f.pos])
            closest_factory_tile = closest_factory(factory_tiles, factory_tiles, unit, game_state)
            if factory_adjacent(closest_factory_tile, unit):
                queue = [unit.transfer(direction_to(unit.pos, home_f.pos), 0, unit.cargo.ice, n=1)]
                self.update_actions(unit, queue)
                return
            else:
                self.deliver_payload(unit, 0, unit.cargo.ice, home_f, game_state)

    def light_actions(self, unit, title, home_f, game_state, obs):
        if title == "newb" and unit.power < 200:
            d_pow = 200 - unit.power
            self.recharge(unit, home_f, game_state, desired_power=d_pow)
            return
        else:
            if unit.power < 30:
                self.recharge(unit, home_f, game_state)
            elif len(self.actions[unit.unit_id]) == 0:
                self.dig_rubble(unit, game_state, obs)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # SETUP

        self.act_step += 1
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]  # all of your units
        factory_tiles = np.array([factory.pos for factory_id, factory in factories.items()])
        factory_units = [factory for factory_id, factory in factories.items()]
        self.new_positions = [f.pos for fid, f in factories.items()]
        new_acts = dict()
        for uid, acts in self.prev_actions.items():
            if isinstance(acts, list):
                new_acts[uid] = acts[1:]
            elif isinstance(acts, int):
                continue
        self.prev_actions = new_acts
        self.actions = dict()

        if self.act_step == 5:
            opp_factories = game_state.factories[self.opp_player]
            for u_id, factory in opp_factories.items():
                self.opp_strains.append(factory.strain_id)
            for u_id, factory in factories.items():
                self.strains.append(factory.strain_id)

        # UNITS
        for unit_id, unit in units.items():
            # SETUP
            if unit.unit_id not in self.actions.keys():
                self.actions[unit.unit_id] = []
            if unit.unit_id not in self.prev_actions.keys():
                self.prev_actions[unit.unit_id] = []

            factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
            closest_f = factory_units[np.argmin(factory_distances)]
            if unit_id not in self.inventory.all_units:  # then it's new and needs to be added to inventory
                self.inventory.factory_units[closest_f.unit_id].append(unit_id)
                self.inventory.all_units.append(unit_id)

            home_id = [f_id for f_id, inv in self.inventory.factory_units.items() if unit_id in inv]
            home_id = str(home_id[0])
            if home_id in factories.keys():
                home_factory = factories[home_id]
            else:
                home_id = closest_f.unit_id
                home_factory = closest_f

            # ATTACK EVASION
            evading = False
            for u_id, u in game_state.units[self.opp_player].items():
                o_facto = [op.pos for op in game_state.factories[self.opp_player].values()]
                if unit.unit_type == "HEAVY":
                    if np.linalg.norm(u.pos - unit.pos) <= 2 and u.unit_type == "HEAVY":
                        if not (u.pos & o_facto).all():
                            if unit.power <= u.power:
                                self.retreat(unit, u, home_factory)
                                evading = True
                                break
                            elif np.linalg.norm(u.pos - unit.pos) <= 1:
                                self.move_toward(u.pos, unit, game_state, evading=True)
                                evading = True
                                break

                elif unit.unit_type == "LIGHT":
                    if np.linalg.norm(u.pos - unit.pos) <= 1:
                        if not (u.pos & o_facto).all():
                            if unit.power < u.power:
                                self.recharge(unit, home_factory, game_state)
                                evading = True
                                break
                            else:
                                self.move_toward(u.pos, unit, game_state, evading=True)
                                evading = True
                                break
            if evading:
                continue

            # HEAVY
            if unit.unit_type == "HEAVY":
                home_homers = self.inventory.factory_types[home_id].count("homer")
                if home_homers == 0 and unit_id not in self.inventory.unit_title.keys():
                    self.inventory.unit_title[unit_id] = "homer"
                    self.inventory.factory_types[home_id].append("homer")
                elif unit_id not in self.inventory.unit_title.keys():
                    home_sentries = self.inventory.factory_types[home_id].count("sentry")
                    if home_sentries < 1:
                        self.inventory.factory_types[home_id].append("sentry")
                        self.inventory.unit_title[unit_id] = "sentry"
                title = self.inventory.unit_title[unit_id]
                self.heavy_actions(unit, title, home_factory, game_state, obs)

            # LIGHT
            elif unit.unit_type == "LIGHT":
                home_helpers = self.inventory.factory_types[home_id].count("helper")
                if home_helpers == 0:
                    self.inventory.unit_title[unit_id] = "helper"
                    self.inventory.factory_types[home_id].append("helper")
                    self.inventory.factory_units[home_id].append(unit_id)
                elif unit_id not in self.inventory.unit_title.keys():
                    home_diggers = self.inventory.factory_types[home_id].count("digger")
                    home_miners = self.inventory.factory_types[home_id].count("miner")
                    if home_diggers < 3:
                        self.inventory.factory_types[home_id].append("digger")
                        self.inventory.unit_title[unit_id] = "digger"
                        self.inventory.factory_units[home_id].append(unit_id)
                    elif home_miners < 1:
                        self.inventory.factory_types[home_id].append("miner")
                        self.inventory.unit_title[unit_id] = "miner"
                        self.inventory.factory_units[home_id].append(unit_id)
                    elif home_diggers < 5:
                        self.inventory.factory_types[home_id].append("digger")
                        self.inventory.unit_title[unit_id] = "digger"
                        self.inventory.factory_units[home_id].append(unit_id)
                    else:
                        nu = [[fid, len(unts)] for fid, unts in self.inventory.factory_units.items()]
                        print(f"nu: {nu}", file=sys.stderr)
                        fids = [n[0] for n in nu]
                        lens = [n[1] for n in nu]
                        new_home = fids[np.argmin(lens)]
                        if new_home in factories.keys():
                            home_id = new_home
                            home_factory = factories[home_id]
                        self.inventory.factory_types[home_id].append("newb")
                        self.inventory.unit_title[unit_id] = "newb"
                        self.inventory.factory_units[home_id].append(unit_id)

                title = self.inventory.unit_title[unit_id]
                self.light_actions(unit, title, home_factory, game_state, obs)

                # self.light_actions(unit, closest_f, actions, game_state, obs)

        # FACTORIES
        for unit_id, factory in factories.items():
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
                if number_of_helpers < 1:
                    if factory.can_build_light(game_state):
                        self.actions[unit_id] = factory.build_light()
                        continue
                elif number_of_diggers < 3 and self.act_step % 4 == 0:
                    if factory.can_build_light(game_state):
                        self.actions[unit_id] = factory.build_light()
                        continue
            if factory.cargo.water > 50 and self.act_step > 780:
                self.actions[unit_id] = factory.water()

        actions_to_submit = dict()
        for uid, acts in self.actions.items():
            if isinstance(acts, list):
                if len(acts) > 0:
                    actions_to_submit[uid] = acts
            else:
                actions_to_submit[uid] = acts
        return actions_to_submit
