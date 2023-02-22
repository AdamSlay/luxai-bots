from math import floor

from lib.actions import dig_rubble
from lib.dijkstra import dijkstras_path
from lib.pathing import path_to
from lib.utils import *


class Queue:
    def __init__(self, agent):
        self.agent = agent

    def build_mining_queue(self, resource, unit, home_f, game_state, obs, sentry=False):
        queue = []
        path = []
        pickup_amt = 0
        if sentry is True:
            mining_tile = closest_type_tile(resource, home_f, self.agent.player, self.agent.opp_player, game_state, obs,
                                            this_is_the_unit=unit)
        else:
            mining_tile = closest_type_tile(resource, home_f, self.agent.player, self.agent.opp_player, game_state, obs,
                                            heavy=True,
                                            this_is_the_unit=unit)
            print(f"Step {game_state.real_env_steps}: {unit.unit_id} is going to mine {mining_tile}",
                  file=sys.stderr)
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

        if unit.unit_type == "HEAVY":
            if unit.pos[0] != mining_tile[0] or unit.pos[1] != mining_tile[1]:
                path = path_to(unit, unit.pos, mining_tile)
                queue.extend(path)
            path_back = path_to(unit, mining_tile, factory_tile)

        else:
            rubble_map = game_state.board.rubble
            o_facto = [u.pos for u in game_state.factories[self.agent.opp_player].values()]
            opp_factories = get_factory_tiles(o_facto)
            unit_positions = [u.pos for u in game_state.units[self.agent.player].values() if u.unit_id != unit.unit_id]
            unit_positions.extend(opp_factories)
            unit_positions.extend(self.agent.new_positions)
            if unit.pos[0] != mining_tile[0] or unit.pos[1] != mining_tile[1]:
                path = []
                path_positions = dijkstras_path(unit.unit_type, rubble_map, unit.pos, mining_tile, unit_positions)
                print(
                    f"Step {game_state.real_env_steps}: {unit.unit_id} is following this path to the mining tile {path_positions}",
                    file=sys.stderr)
                for i, pos in enumerate(path_positions):
                    if i + 1 < len(path_positions):
                        path.append(path_to(unit, path_positions[i], path_positions[i + 1]))
                path = [act[0] for act in path]
                if len(path) > 0:
                    queue.extend(path)
                else:
                    queue = dig_rubble(unit, self.agent.player, self.agent.opp_player, self.agent.new_positions, game_state, obs)
                    return queue

            path_back = []
            path_back_positions = dijkstras_path(unit.unit_type, rubble_map, mining_tile, factory_tile,
                                                 unit_positions)
            print(
                f"Step {game_state.real_env_steps}: {unit.unit_id} is following this path back {path_back_positions}",
                file=sys.stderr)

            for i, pos in enumerate(path_back_positions):
                if i + 1 < len(path_back_positions):
                    path_back.append(path_to(unit, path_back_positions[i], path_back_positions[i + 1]))
            path_back = [act[0] for act in path_back]

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

    def build_transfer_queue(self):
        pass