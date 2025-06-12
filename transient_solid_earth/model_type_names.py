"""
Because all models have to be identified.
"""

from typing import Type

from .generic_rheology_model import MODEL, GenericRheologyModel
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel

MODEL_TYPE_NAMES: dict[str, Type[MODEL]] = {
    "love_numbers": SolidEarthFullNumericalModel,
    "interpolate_love_numbers": GenericRheologyModel,
}
