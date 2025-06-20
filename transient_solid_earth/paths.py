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
grace_data_path = data_path.joinpath("grace")

### Input data.
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

#### Input load numerical models.
elastic_load_models_path = loads_path.joinpath("elastic_load_models")

#### Output load model trends.
anelastic_load_model_trends_path = loads_path.joinpath("anelastic_load_model_trends")

anelastic_pole_motion_path = anelastic_load_model_trends_path.joinpath("pole_motions")

##### Degree one inversion components.
harmonic_geoid_deformation_trends_path = anelastic_load_model_trends_path.joinpath(
    "geoid_deformations"
)
harmonic_vertical_displacement_trends_path = anelastic_load_model_trends_path.joinpath(
    "vertical_displacements"
)
harmonic_residual_trends_path = anelastic_load_model_trends_path.joinpath("residuals")

anelastic_load_models_path = anelastic_load_model_trends_path.joinpath("load_models")

### SLURM logs.
logs_path = outputs_path.joinpath("logs")

#### Love numbers job array logs.
sub_paths = [
    "love_numbers",
    "interpolate_love_numbers",
    "generate_elastic_load_models",
    "invert_degree_one",
]
logs_subpaths = {sub_path: logs_path.joinpath(sub_path) for sub_path in sub_paths}


#### Intermediate preprocessing/post-processing results.
worker_information_subpaths = {
    sub_path: log_subpath.joinpath("worker_informations")
    for sub_path, log_subpath in logs_subpaths.items()
}

INTERMEDIATE_RESULT_STRING = "intermediate_results"
intermediate_result_subpaths = {
    sub_path: log_subpath.joinpath(INTERMEDIATE_RESULT_STRING)
    for sub_path, log_subpath in logs_subpaths.items()
}

elastic_load_model_parameters_subpath = intermediate_result_subpaths[
    "generate_elastic_load_models"
].joinpath("parameters")


def get_love_numbers_subpath(model_id: str, n: int, period: float) -> Path:
    """
    Generates the path to save Y_i integration results for a given model.
    """
    return (
        intermediate_result_subpaths["love_numbers"]
        .joinpath(model_id)
        .joinpath(str(n))
        .joinpath(str(period))
    )


def interpolated_love_numbers_path(periods_id: str, rheological_model_id: str) -> Path:
    """
    Gets the path for Love numbers of a given rheological model interpolated on given periods.
    """

    return (
        intermediate_result_subpaths["interpolate_love_numbers"]
        .joinpath(periods_id)
        .joinpath(rheological_model_id)
    )
