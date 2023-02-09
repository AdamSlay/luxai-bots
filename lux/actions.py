import numpy as np
from lux.utils import closest_type_tile, direction_to


def deliver_to_factory(unit, game_state) -> None:
    pass


def mine_type(resource, unit, actions, game_state, player, obs) -> None:
    target_tile = closest_type_tile(resource, unit, game_state, player, obs)

    if np.all(target_tile == unit.pos):
        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
            digs = (unit.power - unit.action_queue_cost(game_state)) // (unit.dig_cost(game_state))
            actions[unit.unit_id] = [unit.dig(n=digs)]
    else:
        # TODO: path_toward(unit.pos, target_tile)
        direction = direction_to(unit.pos, target_tile)
        move_cost = unit.move_cost(game_state, direction)
        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
            actions[unit.unit_id] = [unit.move(direction, repeat=0)]
