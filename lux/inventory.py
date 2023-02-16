from dataclasses import dataclass


@dataclass
class Inventory:
    all_units: list
    factory_units: dict
    factory_types: dict
    unit_title: dict
    ice_tiles: dict
    ore_tiles: dict
    homer_helpers: dict
