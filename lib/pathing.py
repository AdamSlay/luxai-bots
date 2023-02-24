from lib.utils import direction_to, next_position, find_new_direction


def move_toward(target_tile, unit, player, opp_player, new_positions, game_state, evading=False) -> list:
    unit_positions = [u.pos for u in game_state.units[player].values() if u.unit_id != unit.unit_id]
    unit_positions.extend(new_positions)

    direction = direction_to(unit.pos, target_tile)
    if not evading:
        unit_positions.extend([u.pos for u in game_state.units[opp_player].values()])
    next_pos = next_position(unit, direction)
    for u in unit_positions:
        if next_pos[0] == u[0] and next_pos[1] == u[1]:
            new_direction = find_new_direction(unit, unit_positions, game_state)
            if unit.move_cost(game_state, direction) is not None and unit.action_queue_cost(game_state) is not None:
                cost = unit.move_cost(game_state, direction) + unit.action_queue_cost(game_state)
            elif unit.unit_type == "LIGHT":
                cost = 8
            else:
                cost = 30
            if unit.power >= cost:
                return [unit.move(new_direction, repeat=0)]
            else:
                return [unit.recharge(x=cost)]
    if unit.move_cost(game_state, direction) is not None and unit.action_queue_cost(game_state) is not None:
        cost = unit.move_cost(game_state, direction) + unit.action_queue_cost(game_state)
    elif unit.unit_type == "LIGHT":
        cost = 8
    else:
        cost = 30

    if unit.power >= cost:
        # new_positions.append(next_pos)
        return [unit.move(direction, repeat=0)]
    else:
        return [unit.recharge(x=cost)]


def path_to(unit, start, finish) -> list:
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
