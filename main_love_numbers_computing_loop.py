"""
Main call for Love numbers computing loop on rheologies.
"""

import os
import shutil
from time import time

import numpy

from transient_solid_earth import (
    SolidEarthModelPart,
    adaptative_step_parallel_computing_loop,
    asymptotic_degree_value,
    asymptotic_love_numbers_computing_loop,
    create_all_model_variations,
    generate_degrees_list,
    interpolate_parallel_computing_loop,
    load_parameters,
    logs_subpaths,
)

CLEAR_COMPUTING = False
CLEAR_INTERPOLATING = True
CLEAR_ASYMPTOTIC = True
CLEAR_GREEN = True

if __name__ == "__main__":

    # Clears.
    if CLEAR_COMPUTING and logs_subpaths["love_numbers"].exists():
        shutil.rmtree(logs_subpaths["love_numbers"])
    if CLEAR_INTERPOLATING and logs_subpaths["interpolate_love_numbers"].exists():
        shutil.rmtree(logs_subpaths["interpolate_love_numbers"])
    if CLEAR_ASYMPTOTIC and logs_subpaths["asymptotic_love_numbers"].exists():
        shutil.rmtree(logs_subpaths["asymptotic_love_numbers"])
    if CLEAR_GREEN and logs_subpaths["green_functions"].exists():
        shutil.rmtree(logs_subpaths["green_functions"])

    # Initializes.
    parameters = load_parameters()

    models: list[tuple[dict[SolidEarthModelPart, str], list[dict[SolidEarthModelPart, str]]]] = (
        create_all_model_variations(variable_parameters=parameters.solid_earth_variabilities)
    )

    for elastic_model, anelastic_models in models:

        rheologies = [elastic_model]  # + anelastic_models

        t_0 = time()

        # Processes.
        adaptative_step_parallel_computing_loop(
            rheologies=rheologies,
            function_name="love_numbers",
            fixed_parameter_list=generate_degrees_list(
                degree_thresholds=parameters.solid_earth.degree_discretization.thresholds,
                degree_steps=parameters.solid_earth.degree_discretization.steps,
            ),
            parameters=parameters,
        )

        # Interpolates on periods.
        interpolate_parallel_computing_loop(
            function_name="love_numbers", rheologies=rheologies, parameters=parameters
        )

        # Interpolates on degrees.
        interpolate_parallel_computing_loop(
            function_name="love_numbers",
            rheologies=rheologies,
            parameters=parameters,
            fixed_parameter_new_values=numpy.arange(
                start=1,
                stop=asymptotic_degree_value(parameters=parameters),
            ).tolist(),
        )

        # Computes asymptotic Love numbers.
        asymptotic_love_numbers_computing_loop(rheologies=rheologies, parameters=parameters)

        # Computes Green functions.

        adaptative_step_parallel_computing_loop(
            rheologies=rheologies,
            function_name="green_functions",
            parameters=parameters,
        )

        t_1 = time()
        print(t_1 - t_0)

    os._exit(status=0)
