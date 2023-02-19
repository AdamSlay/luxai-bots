from dataclasses import dataclass

'''
Inventory is a dataclass that holds the inventory of each factory
'''
@dataclass
class Inventory:
    all_units: list  # all of my units. I *think* this is redundant
    factory_units: dict  # units beloning to each factory
    factory_types: dict  # this should be more discriptive. factory_unit_types?
    unit_title: dict  # not sure that I need this anymore since changing title each step is more flexible
