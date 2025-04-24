"""
Arborescence configuration.
"""

from enum import Enum
from pathlib import Path


class SolidEarthModelPart(Enum):
    """
    Available model parts.
    """

    ELASTICITY = "elasticity"
    LONG_TERM_ANELASTICITY = "long_term_anelasticity"
    SHORT_TERM_ANELASTICITY = "short_term_anelasticity"


# Contains both inputs and outputs.
data_path = Path("data")

## Inputs.
inputs_path = data_path.joinpath("inputs")

### Input data.
grace_data_path = inputs_path.joinpath("grace")
gmsl_data_path = inputs_path.joinpath("gmsl_data")
pole_data_path = inputs_path.joinpath("pole_data")
masks_data_path = inputs_path.joinpath("masks")

### Solid Earth model descriptions.
solid_earth_model_descriptions_root_path = inputs_path.joinpath("solid_earth_model_descriptions")
solid_earth_model_descriptions_path: dict[SolidEarthModelPart, Path] = {
    model_part: solid_earth_model_descriptions_root_path.joinpath(model_part.value)
    for model_part in SolidEarthModelPart
}

### Solid Earth numerical models.
solid_earth_numerical_models_base_path = data_path.joinpath("solid_earth_numerical_models")
solid_earth_numerical_models_path: dict[SolidEarthModelPart, Path] = {
    model_part: solid_earth_numerical_models_base_path.joinpath(model_part.value)
    for model_part in SolidEarthModelPart
}
solid_earth_full_numerical_models_path = solid_earth_numerical_models_base_path.joinpath(
    "solid_earth_full_numerical_models"
)

## Outputs.
outputs_path = data_path.joinpath("outputs")

### Tables to manage results.
tables_path = outputs_path.joinpath("tables")

### Love numbers.
love_numbers_path = outputs_path.joinpath("love_numbers")

### Load numerical models.
loads_path = outputs_path.joinpath("loads")

#### Input load signal numerical models.
input_load_signal_models_path = loads_path.joinpath("input_load_signal_models")

#### Output load signal trends.
output_load_signal_trends_path = loads_path.joinpath("output_load_signal_trends")

##### Degree one inversion components.
harmonic_geoid_trends_path = output_load_signal_trends_path.joinpath("geoid_terms")
harmonic_radial_displacement_trends_path = output_load_signal_trends_path.joinpath(
    "radial_displacement_terms"
)
harmonic_residual_trends_path = output_load_signal_trends_path.joinpath("residuals")

##### Anelastic load signals.
anelastic_load_signals_path = output_load_signal_trends_path.joinpath("load_signal_trends")

### SLURM logs.
logs_path = outputs_path.joinpath("logs")

#### Love numbers job array logs.
logs_subpaths = {sub_path: logs_path.joinpath(sub_path) for sub_path in ["love_numbers"]}
