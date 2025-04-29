"""
Main call for Love numbers computing loop on rheologies.
"""

from time import time

from transient_solid_earth import (
    SolidEarthModelPart,
    adaptative_step_parallel_computing_loop,
    create_all_model_variations,
    generate_degrees_list,
    load_parameters,
)

if __name__ == "__main__":

    parameters = load_parameters()

    models: list[tuple[dict[SolidEarthModelPart, str], list[dict[SolidEarthModelPart, str]]]] = (
        create_all_model_variations(variable_parameters=parameters.solid_earth_variabilities)
    )

    t_0 = time()

    for elastic_model, anelastic_models in models:
        adaptative_step_parallel_computing_loop(
            rheologies=[elastic_model] + anelastic_models,
            function_name="love_numbers",
            fixed_parameter_list=generate_degrees_list(
                degree_thresholds=parameters.solid_earth.degree_discretization.thresholds,
                degree_steps=parameters.solid_earth.degree_discretization.steps,
            ),
            parameters=parameters,
        )

    print(time() - t_0)
