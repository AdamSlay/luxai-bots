from dataclasses import dataclass
import numpy as np

# TODO: the whole point this exists is to be able to pass spawn location into closest_type_tile. This is a hack.
@dataclass
class SpawnSpot:
    pos: tuple
