"""
Main call for Love numbers computing loop on rheologies.
"""

import shutil
from time import time

from transient_solid_earth import (
    SolidEarthModelPart,
    adaptative_step_parallel_computing_loop,
    create_all_model_variations,
    generate_degrees_list,
    interpolate_on_grid_parallel_computing_loop,
    load_parameters,
    logs_subpaths,
)

if __name__ == "__main__":

    # Clears.
    if logs_subpaths["love_numbers"].exists():
        shutil.rmtree(logs_subpaths["love_numbers"])

    parameters = load_parameters()

    models: list[tuple[dict[SolidEarthModelPart, str], list[dict[SolidEarthModelPart, str]]]] = (
        create_all_model_variations(variable_parameters=parameters.solid_earth_variabilities)
    )

    for elastic_model, anelastic_models in models:

        t_0 = time()

        rheologies = [elastic_model] + anelastic_models

        adaptative_step_parallel_computing_loop(
            rheologies=rheologies,
            function_name="love_numbers",
            fixed_parameter_list=generate_degrees_list(
                degree_thresholds=parameters.solid_earth.degree_discretization.thresholds,
                degree_steps=parameters.solid_earth.degree_discretization.steps,
            ),
            parameters=parameters,
        )

        t_1 = time()
        print(t_1 - t_0)

        interpolate_on_grid_parallel_computing_loop(
            function_name="love_numbers", rheologies=rheologies, parameters=parameters
        )

        print(time() - t_1)
