import numpy as np

from lib.utils import closest_opp_lichen, closest_type_tile, direction_to, distance_to, factory_adjacent, get_target_tile
from lib.pathing import move_toward


def attack_opp(unit, player, opp_player, opp_lichen, new_positions, game_state):
    closest_lichen = closest_opp_lichen(opp_lichen, unit, player, opp_player, game_state)
    if np.all(closest_lichen == unit.pos):
        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
            digs = (unit.power - unit.action_queue_cost(game_state) - 30) // (unit.dig_cost(game_state))
            if digs > 20:
                digs = 20
            elif digs < 1:
                digs = 1
            return [unit.dig(n=digs)]
    else:
        return move_toward(closest_lichen, unit, player, opp_player, new_positions, game_state)


def dig_rubble(unit, player, opp_player, new_positions, game_state, obs):
    # target_tile = closest_type_tile("rubble", unit, player, opp_player, game_state, obs)
    target_tile = get_target_tile("rubble", unit, player, new_positions, game_state, obs)
    if np.all(target_tile == unit.pos):
        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state) + 20:
            digs = (unit.power - unit.action_queue_cost(game_state) - 20) // (unit.dig_cost(game_state))
            if digs > 20:
                digs = 20
            queue = []
            for i in range(digs):
                queue.append(unit.dig(n=1))
            return queue
    else:
        queue = move_toward(target_tile, unit, player, opp_player, new_positions, game_state)
        return queue


def deliver_payload(unit, resource: int, amount: int, player, opp_player, new_positions, closest_f, game_state):
    direction = direction_to(unit.pos, closest_f.pos)
    adjacent_to_factory = factory_adjacent(closest_f.pos, unit)
    if adjacent_to_factory:
        return [unit.transfer(direction, resource, amount, n=1)]
    else:
        if unit.unit_type == "HEAVY":
            return move_toward(closest_f.pos, unit, player, opp_player, new_positions, game_state)
        else:
            return move_toward(closest_f.pos, unit, player, opp_player, new_positions, game_state)


def power_recharge(unit, home_f, player, opp_player, new_positions, game_state):
    adjacent_to_factory = factory_adjacent(home_f.pos, unit)
    if adjacent_to_factory:
        can_pickup = True
        for pos in new_positions:
            if unit.pos[0] == pos[0] and unit.pos[1] == pos[1]:
                can_pickup = False
                break
        if can_pickup:
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
            return [unit.pickup(4, pickup_amt, n=1)]

    return move_toward(home_f.pos, unit, player, opp_player, new_positions, game_state)


def retreat(unit, opp_unit, home_f, game_state):
    adjacent_to_factory = distance_to(unit.pos, home_f.pos)
    if adjacent_to_factory <= 1:
        desired_pwr = (opp_unit.power + 50) - unit.power
        return [unit.pickup(4, desired_pwr, n=1)]
    else:
        direction = direction_to(unit.pos, home_f.pos)
        if unit.power >= unit.move_cost(game_state, direction) + unit.action_queue_cost(game_state):
            return [unit.move(direction, repeat=0)]


def evade(unit, home_factory, player, opp_player, new_positions, game_state):
    for u_id, u in game_state.units[opp_player].items():
        o_facto = [op.pos.tolist() for op in game_state.factories[opp_player].values()]
        if unit.unit_type == "HEAVY" and unit.power >= 40:
            attacker_dist = distance_to(unit.pos, u.pos)
            if attacker_dist < 2 and u.unit_type == "HEAVY":
                if u.pos.tolist() not in o_facto:  # if the enemy is not on a factory
                    if unit.power <= u.power:
                        queue = retreat(unit, u, home_factory, game_state)
                        evading = True
                        return evading, queue
                    else:
                        queue = move_toward(u.pos, unit, player, opp_player, new_positions, game_state, evading=True)
                        evading = True
                        return evading, queue
        elif unit.unit_type == "LIGHT":
            attacker_dist = distance_to(unit.pos, u.pos)
            if attacker_dist <= 1:
                if unit.power <= u.power or unit.power < 20 or u.unit_type == "HEAVY":
                    queue = power_recharge(unit, home_factory, player, opp_player, new_positions, game_state)
                    evading = True
                    return evading, queue
                elif unit.power > u.power:
                    if u.pos.tolist() not in o_facto:
                        queue = move_toward(u.pos, unit, player, opp_player, new_positions, game_state, evading=True)
                        evading = True
                        return evading, queue
    return False, []