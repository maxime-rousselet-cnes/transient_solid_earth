"""
Because all models have to be identified.
"""

from typing import Type

from .model import MODEL
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel
from .test_models import TestModel

MODEL_TYPE_NAMES: dict[str, Type[MODEL]] = {
    "test_models": TestModel,
    "love_numbers": SolidEarthFullNumericalModel,
}
