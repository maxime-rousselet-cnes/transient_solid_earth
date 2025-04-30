"""
Because all models have to be identified.
"""

from typing import Type

from .generic_rheology_model import GenericRheologyModel
from .model import MODEL
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel
from .test_models import TestModel

MODEL_TYPE_NAMES: dict[str, Type[MODEL]] = {
    "test_models": TestModel,
    "love_numbers": SolidEarthFullNumericalModel,
    "interpolate_test_models": TestModel,
    "interpolate_love_numbers": GenericRheologyModel,
}
