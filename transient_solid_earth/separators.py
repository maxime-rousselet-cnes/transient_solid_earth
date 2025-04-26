"""
Name separators configuration.
"""

LAYER_NAMES_SEPARATOR = "__"
SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR = 5 * "_"
SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR = 4 * "_"
LAYERS_SEPARATOR = "___"
VALUES_SEPARATOR = "__"

UNUSED_MODEL_PART_DEFAULT_NAME = "unused"


def is_elastic(model_id: str) -> bool:
    """
    Check if the model is elastic given its ID.
    """
    return model_id.count(UNUSED_MODEL_PART_DEFAULT_NAME) == 2
