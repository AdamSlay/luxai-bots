import numpy as np

from lux.kit import obs_to_game_state
from lib.utils import manhattan_dist_to_nth_closest, my_turn_to_place_factory, closest_type_tile, distance_to
from lib.spawn import SpawnSpot


def setup(self, step: int, obs, remainingOverageTime: int = 60):
    if step == 0:
        return dict(faction="TheBuilders", bid=5)
    else:
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        water_left = game_state.teams[self.player].water
        metal_left = game_state.teams[self.player].metal

        factories_to_place = game_state.teams[self.player].factories_to_place
        my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)

        if factories_to_place > 0 and my_turn_to_place:
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
                dist_to_ore = distance_to(spot.pos, closest_ore)
                if dist_to_ore <= 10:
                    m, w = 145, 145
            if factories_to_place == 1:
                m, w = metal_left, water_left
            return dict(spawn=spawn_loc, metal=m, water=w)
        return dict()
