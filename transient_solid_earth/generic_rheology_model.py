"""
Contains both the elastic and the anelastic models informations, preprocessed.
The variables of this preprocessed class are still independent on frequency.
"""

import dataclasses
from typing import Optional

from .model import Model
from .parameters import DEFAULT_SOLID_EARTH_PARAMETERS, SolidEarthParameters
from .paths import SolidEarthModelPart
from .separators import (
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
)


def solid_earth_full_numerical_model_id_from_part_names(
    elasticity_name: str, long_term_anelasticity_name: str, short_term_anelasticity_name: str
) -> str:
    """
    Builds an id for a solid Earth full numerical model given the names of its components.
    """

    return SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(
        (
            part_name.replace("/", SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR)
            for part_name in (
                elasticity_name,
                long_term_anelasticity_name,
                short_term_anelasticity_name,
            )
        )
    )


@dataclasses.dataclass
class GenericRheologyModel(Model):
    """
    Because we sometimes need model names without building the models.
    """

    solid_earth_parameters: SolidEarthParameters

    def __init__(
        self,
        solid_earth_parameters: SolidEarthParameters = DEFAULT_SOLID_EARTH_PARAMETERS,
        rheology: Optional[dict[SolidEarthModelPart, Optional[str]]] = None,
    ) -> None:

        super().__init__(
            model_id=solid_earth_full_numerical_model_id_from_part_names(
                elasticity_name=rheology[SolidEarthModelPart.ELASTICITY],
                long_term_anelasticity_name=rheology[SolidEarthModelPart.LONG_TERM_ANELASTICITY],
                short_term_anelasticity_name=rheology[SolidEarthModelPart.SHORT_TERM_ANELASTICITY],
            )
        )

        self.solid_earth_parameters = solid_earth_parameters
