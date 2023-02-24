from lib.actions import attack_opp, dig_rubble, deliver_payload, power_recharge, evade
from lib.inventory import Inventory
from lib.queue_builder import Queue
from lib.setup_factories import setup
from lux.kit import obs_to_game_state, EnvConfig
from lib.utils import *  # it's ok, these are just helper functions


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
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
        spawn_queue = setup(self, step, obs, remainingOverageTime)
        return spawn_queue

    # ------------------- Agent Behavior -------------------
    def update_action_queue(self):
        new_acts = dict()
        for uid, acts in self.prev_actions.items():
            if isinstance(acts, list):
                new_acts[uid] = acts[1:]
            elif isinstance(acts, int):
                continue
        self.prev_actions = new_acts

    def finalize_action_queue(self):
        actions_to_submit = dict()
        for uid, acts in self.actions.items():
            if isinstance(acts, list):
                if len(acts) > 0:  # only submit actions that are not empty
                    actions_to_submit[uid] = acts
            else:  # it's an int and therefore a factory action, so submit it
                actions_to_submit[uid] = acts
        return actions_to_submit

    def update_new_positions(self, units):
        for uid, act in self.prev_actions.items():
            if isinstance(act, list) and len(act) > 0 and uid in units.keys():
                unit = units.get(uid)
                if act[0][0] == 0:  # it's a move command
                    new_pos = next_position(unit, act[0][1])
                else:
                    new_pos = unit.pos

                append_to_new = True
                for pos in self.new_positions:
                    if new_pos[0] == pos[0] and new_pos[1] == pos[1]:
                        print(f"unit {unit.unit_id} tried to move to {new_pos} but it's already occupied", file=sys.stderr)
                        append_to_new = False
                        self.prev_actions[uid] = []
                        break
                if append_to_new:
                    self.new_positions.append(new_pos)

    def remove_new_position(self, unit):
        if unit.unit_id in self.prev_actions.keys() and len(self.prev_actions[unit.unit_id]) > 0:
            if self.prev_actions[unit.unit_id][0][0] == 0:
                new_pos = next_position(unit, self.prev_actions[unit.unit_id][0][1])
            else:
                new_pos = unit.pos

            for i, pos in enumerate(self.new_positions):
                if new_pos[0] == pos[0] and new_pos[1] == pos[1]:
                    del self.new_positions[i]
                    break

    def update_actions(self, unit, queue):
        if isinstance(queue, list) and len(queue) > 0:
            if queue[0][0] == 0:  # it's a move command
                new_pos = next_position(unit, queue[0][1])
                self.new_positions.append(new_pos)
            else:
                self.new_positions.append(unit.pos)

        self.actions[unit.unit_id] = queue
        self.prev_actions[unit.unit_id] = queue

    def heavy_actions(self, unit, title, home_f, game_state, obs):
        adjacent_to_factory = factory_adjacent(home_f.pos, unit)
        # TODO: something is going ary with heavies running out of power. That needs to be fixed
        if unit.power < 30 and not adjacent_to_factory:
            self.remove_new_position(unit)
            self.prev_actions[unit.unit_id] = []
            return
        elif unit.power < 100:
            self.remove_new_position(unit)
            self.prev_actions[unit.unit_id] = []
            queue = power_recharge(unit, home_f, self.player, self.opp_player, self.new_positions, game_state)
            self.update_actions(unit, queue)
            return
        if game_state.board.rubble[unit.pos[0]][unit.pos[1]] > 0:
            if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                self.actions[unit.unit_id] = [unit.dig(n=1)]
                return
            else:
                return

        factory_inv = len(self.inventory.factory_units[home_f.unit_id])
        step = game_state.real_env_steps
        if factory_inv <= 3 and home_f.cargo.water > 500 and home_f.cargo.metal < 40 and step < 700:
            closest_ore = closest_type_tile("ore", home_f, self.player, self.opp_player, game_state, obs,
                                            this_is_the_unit=unit)
            dist_to_ore = distance_to(home_f.pos, closest_ore)
            if dist_to_ore < 10 and len(self.prev_actions[unit.unit_id]) == 0:
                self.remove_new_position(unit)
                queue_builder = Queue(self)
                queue = queue_builder.build_mining_queue("ore", unit, home_f, game_state, obs)
                self.update_actions(unit, queue)
                return

        if home_f.cargo.water < 100 < unit.cargo.ice:  # didn't know you could do this chained comparison
            self.remove_new_position(unit)
            queue = deliver_payload(unit, 0, unit.cargo.ice, self.player, self.opp_player, self.new_positions, home_f,
                                    game_state)
            self.update_actions(unit, queue)
            return

        if factory_adjacent(home_f.pos, unit) and unit.cargo.ore > 0:
            self.remove_new_position(unit)
            direction = direction_to(unit.pos, home_f.pos)
            transfer_ore = [unit.transfer(direction, 1, unit.cargo.ore, n=1)]
            self.update_actions(unit, transfer_ore)
            return

        elif unit.cargo.ice < 1000 and len(self.prev_actions[unit.unit_id]) == 0:
            self.remove_new_position(unit)
            queue_builder = Queue(self)
            queue = queue_builder.build_mining_queue("ice", unit, home_f, game_state, obs)
            self.update_actions(unit, queue)
            return

        elif len(self.prev_actions[unit.unit_id]) == 0:
            if factory_adjacent(home_f.pos, unit):
                queue = [unit.transfer(direction_to(unit.pos, home_f.pos), 0, unit.cargo.ice, n=1)]
                self.update_actions(unit, queue)
                return
            else:
                queue = deliver_payload(unit, 0, unit.cargo.ice, self.player, self.opp_player, self.new_positions,
                                        home_f,
                                        game_state)
                self.update_actions(unit, queue)
                return

    def light_actions(self, unit, title, home_f, game_state, obs):
        if unit.power < 8 and not factory_adjacent(home_f.pos, unit):
            self.remove_new_position(unit)
            self.update_actions(unit, [])
            return

        if unit.power < 50:
            self.remove_new_position(unit)
            if game_state.real_env_steps >= 900:
                closest_f = get_closest_factory(game_state.factories[self.player], unit)
                queue = power_recharge(unit, closest_f, self.player, self.opp_player, self.new_positions, game_state)
                self.update_actions(unit, queue)
                return
            queue = power_recharge(unit, home_f, self.player, self.opp_player, self.new_positions, game_state)
            self.update_actions(unit, queue)
            return

        if game_state.real_env_steps >= 900:
            # If the closest rubble is not relevant at this point and opp has lichen, attack it
            if unit.power < 100 and factory_adjacent(home_f.pos, unit):
                self.remove_new_position(unit)
                pickup_amt = 150 - unit.power
                self.actions[unit.unit_id] = [unit.pickup(4, pickup_amt, n=1)]
                return
            opp_lichen = []
            for i in self.opp_strains:
                opp_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            if len(self.prev_actions[unit.unit_id]) == 0 and np.sum(opp_lichen) > 0:
                self.remove_new_position(unit)
                queue = attack_opp(unit, self.player, self.opp_player, self.opp_strains, self.new_positions, game_state, obs)
                self.update_actions(unit, queue)
                return

        if unit.cargo.ore > 0 and title != "miner":
            self.remove_new_position(unit)
            queue = deliver_payload(unit, 1, unit.cargo.ore, self.player, self.opp_player, self.new_positions, home_f,
                                    game_state)
            self.update_actions(unit, queue)
            return

        if title == "miner" and game_state.real_env_steps < 900:
            home_unit_inv = len(self.inventory.factory_units[home_f.unit_id])
            if unit.cargo.ore >= 25 and home_unit_inv < 4:
                self.remove_new_position(unit)
                queue = deliver_payload(unit, 1, unit.cargo.ore, self.player, self.opp_player, self.new_positions,
                                        home_f, game_state)
                self.update_actions(unit, queue)
                return
            if unit.cargo.ore > 98:
                self.remove_new_position(unit)
                queue = deliver_payload(unit, 1, unit.cargo.ore, self.player, self.opp_player, self.new_positions,
                                        home_f, game_state)
                self.update_actions(unit, queue)
                return

            closest_ore = closest_type_tile("ore", home_f, self.player, self.opp_player, game_state, obs, heavy=True,
                                            this_is_the_unit=unit)
            dist_to_ore = distance_to(home_f.pos, closest_ore)
            if dist_to_ore < 16:
                rubble_here = game_state.board.rubble[unit.pos[0]][unit.pos[1]]
                if rubble_here > 0:
                    self.remove_new_position(unit)
                    digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                    if digs > 20:
                        digs = 20
                    queue = []
                    for i in range(digs):
                        queue.append(unit.dig(n=1))
                    self.update_actions(unit, queue)
                    return
                elif unit.cargo.ore <= 98 and len(self.prev_actions[unit.unit_id]) == 0:
                    self.remove_new_position(unit)
                    queue_builder = Queue(self)
                    queue = queue_builder.build_mining_queue("ore", unit, home_f, game_state, obs)
                    if queue is None:
                        return
                    self.update_actions(unit, queue)
                    return
                return

        if title == "digger" and game_state.real_env_steps > 700:
            opp_lichen, my_lichen = [], []
            for i in self.opp_strains:
                opp_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            for i in self.strains:
                my_lichen.extend(np.argwhere((game_state.board.lichen_strains == i)))
            if np.sum(opp_lichen) > np.sum(my_lichen) * 0.4:
                closest_lichen = closest_opp_lichen(self.opp_strains, home_f, self.player, self.new_positions, game_state, obs)
                if distance_to(unit.pos, closest_lichen) < 12 and game_state.real_env_steps < 900:
                    self.remove_new_position(unit)
                    queue = attack_opp(unit, self.player, self.opp_player, self.opp_strains, self.new_positions, game_state, obs)
                    self.update_actions(unit, queue)
                    return
                elif game_state.real_env_steps >= 900:
                    self.remove_new_position(unit)
                    queue = attack_opp(unit, self.player, self.opp_player, self.opp_strains, self.new_positions, game_state, obs)
                    self.update_actions(unit, queue)
                    return

        if len(self.prev_actions[unit.unit_id]) == 0:
            rubble_here = game_state.board.rubble[unit.pos[0]][unit.pos[1]]
            if rubble_here > 0:
                diggable = True
                for pos in self.new_positions:
                    if unit.pos[0] == pos[0] and unit.pos[1] == pos[1]:
                        diggable = False
                        break
                if diggable:
                    digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
                    if digs > 20:
                        digs = 20
                    queue = []
                    for i in range(digs):
                        queue.append(unit.dig(n=1))
                    self.update_actions(unit, queue)
                    return
            queue = dig_rubble(unit, self.player, self.opp_player, self.new_positions, game_state, obs)
            self.update_actions(unit, queue)
            return

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # SETUP
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]
        opp_factories = game_state.factories[self.opp_player]
        my_factory_centers = [f.pos for i, f in factories.items()]
        opp_factory_centers = np.array([f.pos for i, f in opp_factories.items()])
        opp_factory_tiles = get_factory_tiles(opp_factory_centers)

        self.actions = dict()
        self.inventory.factory_types = dict()
        self.update_action_queue()
        self.new_positions = my_factory_centers
        self.new_positions.extend(opp_factory_tiles)
        self.update_new_positions(units)

        # STRAINS
        if game_state.real_env_steps == 1:
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

            closest_f = get_closest_factory(factories, unit)
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
            evading, evasion_queue = evade(unit, home_factory, self.player, self.opp_player, self.new_positions,
                                           game_state)
            if evading:
                self.remove_new_position(unit)
                self.update_actions(unit, evasion_queue)
                continue

            # HEAVY
            if unit.unit_type == "HEAVY":
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
                if home_id not in self.inventory.factory_types.keys():
                    self.inventory.factory_types[home_id] = []
                home_helpers = self.inventory.factory_types[home_id].count("helper")
                home_miners = self.inventory.factory_types[home_id].count("miner")
                if home_miners == 0:
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
                if number_of_miners < 1 and factory.can_build_light(game_state):
                    self.actions[unit_id] = factory.build_light()
                    continue
                elif number_of_helpers < 2 and factory.can_build_light(game_state):
                    self.actions[unit_id] = factory.build_light()
                    continue
                elif number_of_diggers < 3 and factory.can_build_light(game_state):
                    self.actions[unit_id] = factory.build_light()
                    continue
                elif game_state.real_env_steps > 800 and factory.can_build_light(
                        game_state) and game_state.real_env_steps % 10 == 0:
                    self.actions[unit_id] = factory.build_light()
                    continue

            if factory.cargo.water > 50 and game_state.real_env_steps > 780:
                self.actions[unit_id] = factory.water()

        # FINALIZE ACTIONS
        actions_to_submit = self.finalize_action_queue()
        return actions_to_submit
