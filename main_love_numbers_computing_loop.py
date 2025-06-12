"""
Main to call for:
I   - Elastic load model preprocessings.
II  - Love number computing for all rheological models.
III - Self-coherent anelastic load re-estimation for all elastic loads and rheological models.
"""

import os
import shutil
from pathlib import Path

import numpy

from transient_solid_earth import (
    SolidEarthModelPart,
    adaptative_step_parallel_computing_loop,
    anelastic_load_model_re_estimation_processing_loop,
    create_all_model_variations,
    elastic_load_models_path,
    elastic_polar_tide_correction_back,
    generate_degrees_list,
    generate_elastic_load_models_parallel_loop,
    get_period_interpolation_basis,
    interpolate_parallel_computing_loop,
    interpolated_love_numbers_path,
    load_complex_array,
    load_parameters,
    logs_subpaths,
    tables_path,
)

CLEAR = {
    "love_numbers": False,
    "generate_elastic_load_models": False,
    "interpolate_love_numbers": True,
}
CLEAR_TABLES = CLEAR["generate_elastic_load_models"]


def clear_path(path: Path) -> None:
    """
    Clears the given path if it exists.
    """

    if path.exists():
        shutil.rmtree(path)


if __name__ == "__main__":

    # Eventually clears the directories.
    for path_to_clear, to_clear in CLEAR.items():
        if to_clear and logs_subpaths[path_to_clear].exists():
            clear_path(path=logs_subpaths[path_to_clear])
    if CLEAR_TABLES:
        clear_path(path=tables_path)
        if CLEAR["generate_elastic_load_models"]:
            clear_path(path=elastic_load_models_path)

    # Loads parameters and rheological models.
    parameters = load_parameters()
    rheological_models: list[
        tuple[dict[SolidEarthModelPart, str], list[dict[SolidEarthModelPart, str]]]
    ] = create_all_model_variations(variable_parameters=parameters.solid_earth_variabilities)

    # Preprocesses the elastic load models.
    generate_elastic_load_models_parallel_loop(parameters=parameters)
    (
        period_new_values_per_id,
        interpolation_basis_ids,
        elastic_load_models,
        interpolation_timeout,
        load_model_ids_per_interpolation_basis_ids,
    ) = get_period_interpolation_basis(parameters=parameters)

    # Pre/Post-interpolation degrees for Love numbers.
    degree_new_values = numpy.arange(start=1, stop=parameters.load.model.signature.n_max)
    degree_list = generate_degrees_list(
        degree_thresholds=parameters.solid_earth.degree_discretization.thresholds,
        degree_steps=parameters.solid_earth.degree_discretization.steps,
        n_max=parameters.load.model.signature.n_max,
    )

    # Loops on elastic rheologies. Usually, PREM only.
    for elastic_model, anelastic_models in rheological_models:

        rheologies = [elastic_model] + anelastic_models

        # Processes all Love numbers.
        rheological_model_ids, elastic_model_id = adaptative_step_parallel_computing_loop(
            rheologies=rheologies,
            degree_list=degree_list,
            parameters=parameters,
        )

        # Interpolates the Love numbers on adequate periods/degrees for load computing.
        interpolate_parallel_computing_loop(
            rheologies=rheologies,
            period_new_values_per_id=period_new_values_per_id,
            # Has to come after periods since n_max is redefined.
            degree_new_values=degree_new_values,
            parameters=parameters,
            timeout=interpolation_timeout,
        )

        # Loops on interpolated elastic load models.
        for (
            periods_id,
            elastic_load_model_ids,
        ) in load_model_ids_per_interpolation_basis_ids.items():

            # Memorizes elastic Love numbers.
            elastic_love_numbers = load_complex_array(
                path=interpolated_love_numbers_path(
                    periods_id=periods_id, rheological_model_id=elastic_model_id
                )
            )

            for elastic_load_model_id in elastic_load_model_ids:

                elastic_load_model = elastic_load_models[elastic_model_id]
                elastic_polar_tide_correction_back(
                    elastic_load_model=elastic_load_model, elastic_love_numbers=elastic_love_numbers
                )

            # Loops on rheological models.
            for rheological_model_id in rheological_model_ids:

                # Main anelastic self-coherent re-estimation for the given list of elastic load
                # models.
                anelastic_load_model_re_estimation_processing_loop(
                    elastic_load_models=[
                        elastic_load_model
                        for elastic_load_model_id, elastic_load_model in elastic_load_models.items()
                        if elastic_load_model_id in elastic_load_model_ids
                    ],
                    elastic_love_numbers=elastic_love_numbers,
                    periods_id=periods_id,
                    rheological_model_id=rheological_model_id,
                )

    # Wait for processes to end naturally.
    os._exit(status=0)
