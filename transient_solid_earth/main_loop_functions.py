"""
Sub-functions for a readable main loop.
"""

import shutil
from pathlib import Path

import numpy
from scipy.fft import fft, ifft

from .constants import EMPIRICAL_INTERPOLATION_TIEMOUT_FACTOR, BoundaryCondition, Direction
from .database import (
    add_result_to_table,
    extract_terminal_attributes,
    is_in_table,
    load_complex_array,
    save_base_model,
    save_complex_array,
)
from .degree_one import period_dependent_degree_one_inversion
from .elastic_load_models import ElasticLoadModel, load_elastic_load_model
from .formating import (
    make_grid,
    stack_period_dependent_harmonics,
    unstack_period_dependent_harmonics,
)
from .leakage_correction import forward_modeling_leakage_correction
from .parameters import ParallelComputingParameters, Parameters, generate_hash
from .paths import (
    anelastic_load_models_path,
    anelastic_pole_motion_path,
    elastic_load_models_path,
    harmonic_geoid_deformation_trends_path,
    harmonic_residual_trends_path,
    harmonic_vertical_displacement_trends_path,
    interpolated_love_numbers_path,
)
from .polar_tide import polar_motion_correction
from .trends import get_ocean_mean_trend, get_trend_from_period_dependent_harmonic_model

STEPS_TO_POST_PROCESS = [0, 1, 3]


def clear_path(path: Path) -> None:
    """
    Clears the given path if it exists.
    """

    if path.exists():
        shutil.rmtree(path)


def get_interpolation_timeout(period_new_values_per_id: dict[str, numpy.ndarray[float]]) -> float:
    """
    Empirical timeout parallel Love number interpolation.
    """

    return max(
        sum(period_new_values.shape) / EMPIRICAL_INTERPOLATION_TIEMOUT_FACTOR
        for period_new_values in period_new_values_per_id.values()
    )


def get_period_interpolation_basis(
    parameters: Parameters,
) -> tuple[
    dict[str, numpy.ndarray[float]],
    dict[str, str],
    dict[str, ElasticLoadModel],
    float,
    dict[str, list[str]],
]:
    """
    Returns:
        - A dictionary giving the interpolation basis by ID.
        - A dictionary giving the interpolation basis IDs by elastic load model ID.
        - A dictionary giving the elastic load models by ID.
        - An estimated timeout for parallel interpolations
        - A dictionary giving the elastic load model IDs by interpolation basis ID.
    """

    # Gets interpolation basis IDs:
    interpolation_basis_ids = {}
    interpolation_basis = {}
    parameters.load_model.signature.n_max = 0
    load_models = {}

    for file in elastic_load_models_path.glob("*.json"):

        load_model = load_elastic_load_model(model_id=file.name)
        interpolation_basis_ids[file.name] = (
            load_model.load_model_parameters.interpolation_basis_id()
        )
        interpolation_basis[load_model.load_model_parameters.interpolation_basis_id()] = (
            load_model.base_products.temporal_products.periods
        )
        parameters.load_model.signature.n_max = max(
            parameters.load_model.signature.n_max, load_model.load_model_parameters.signature.n_max
        )
        load_models[file.name] = load_model

    period_new_values_per_id = {
        str(interpolation_basis_id): interpolation_basis[interpolation_basis_id]
        for interpolation_basis_id in numpy.unique(list(interpolation_basis_ids.values()))
    }

    return (
        period_new_values_per_id,
        interpolation_basis_ids,
        load_models,
        get_interpolation_timeout(period_new_values_per_id=period_new_values_per_id),
        {
            interpolation_basis_id: [
                load_model_id
                for load_model_id, load_interpolation_basis_id in interpolation_basis_ids.items()
                if load_interpolation_basis_id == interpolation_basis_id
            ]
            for interpolation_basis_id in period_new_values_per_id
        },
    )


def renormalize_past_temporal_component(
    elastic_load_model: ElasticLoadModel, past_trend_ratio: float
) -> None:
    """
    Normalizes the past part of the temporal component by a given unitless ratio.
    The trend on recent time period still corresponds to 1.0 (unitless).
    """

    last_past_time_period_value = elastic_load_model.base_products.time_dependent_component[
        elastic_load_model.side_products.past_trend_indices[-1]
    ]
    last_recent_time_period_value = elastic_load_model.base_products.time_dependent_component[
        elastic_load_model.side_products.recent_trend_indices[-1]
    ]
    additive_constant = last_past_time_period_value * (1.0 - 1.0 / past_trend_ratio)

    # Multiplicative factor on past time period.
    elastic_load_model.base_products.time_dependent_component[
        elastic_load_model.side_products.past_trend_indices
    ] /= past_trend_ratio
    elastic_load_model.base_products.time_dependent_component[
        -elastic_load_model.side_products.past_trend_indices
    ] /= past_trend_ratio

    # Additive constant on recent time period to maintain continuity.
    elastic_load_model.base_products.time_dependent_component[
        elastic_load_model.side_products.recent_trend_indices
    ] -= additive_constant
    elastic_load_model.base_products.time_dependent_component[
        -elastic_load_model.side_products.recent_trend_indices
    ] += additive_constant

    # Multiplicative factor on antisymmetric interpolation spline to maintain continuity.
    elastic_load_model.base_products.time_dependent_component[
        elastic_load_model.side_products.recent_trend_indices[-1]
        + 1 : -elastic_load_model.side_products.recent_trend_indices[-1]
    ] *= (1.0 - additive_constant / last_recent_time_period_value)


def harmonic_load_model_re_estimation(
    elastic_love_numbers: numpy.ndarray,
    anelastic_love_numbers: numpy.ndarray,
    period_dependent_harmonic_load_model_step_2: numpy.ndarray,
) -> numpy.ndarray:
    """
    Anelastic self-coherent load model re-estimation using the potential load Love number on all
    degrees.
    """

    # (1 + k_el) / (1 + k_anel) * {C, S}.
    return stack_period_dependent_harmonics(  # Stacks harmonics back.
        period_dependent_harmonics=numpy.multiply(
            numpy.concatenate(
                (  # Adds a line of one values for degree zero. => Signal's mean value unmodified.
                    numpy.ones(shape=(anelastic_love_numbers.shape[0], 1)),
                    (
                        1.0
                        + elastic_love_numbers[
                            :, :, BoundaryCondition.LOAD.value, Direction.POTENTIAL.value
                        ]
                    )
                    / (
                        1.0
                        + anelastic_love_numbers[
                            :, :, BoundaryCondition.LOAD.value, Direction.POTENTIAL.value
                        ]
                    ),
                ),
                axis=1,
                # Love numbers: (periods, degrees).
                # After creation of new axes: Ok with unstacked.
            )[:, None, :, None],
            # Unstacks harmonics before straightforward multiplication.
            unstack_period_dependent_harmonics(
                dense_period_dependent_harmonics=period_dependent_harmonic_load_model_step_2
            ),
        ),
    )


def anelastic_load_model_re_estimation_processing_steps(
    elastic_love_numbers: numpy.ndarray,
    anelastic_love_numbers: numpy.ndarray,
    c_2_1_pt_se_complex: numpy.ndarray,
    s_2_1_pt_se_complex: numpy.ndarray,
    save_informations: tuple[ElasticLoadModel, str, ParallelComputingParameters],
) -> tuple[list[float], dict[str, float]]:
    """
    In a Newton's method loop to normalize the past barystatic trend, performs:
        -   Anelastic self-coherent time-dependent polar tide correction.
        -   Anelastic self-coherent load model re-estimation using the potential load Love number
            on all degrees.
        -   Anelastic self-coherent degree one inversion.
        -   Leakage correction.
    Iterates on the past trend normalization factor until the post-processed past trend fits to the
    barystatic load model history.
    """

    (
        elastic_load_model,
        anelastic_load_model_id,
        parallel_computing_parameters,
    ) = save_informations

    # Initializes to enter the loop.
    past_trend = numpy.inf
    temporal_products = elastic_load_model.base_products.temporal_products

    past_trend_ratio = (
        1.0
        if anelastic_love_numbers.shape[0] == 1  # Rheological model is actually the elastic one.
        else elastic_load_model.load_model_parameters.numerical_parameters.initial_past_trend_factor
    )

    # Iterates on past_trend_ratio for convergence on past_trend.
    while (
        abs(past_trend - temporal_products.target_past_trend)
        > elastic_load_model.load_model_parameters.numerical_parameters.past_trend_error
    ):

        renormalize_past_temporal_component(
            elastic_load_model=elastic_load_model, past_trend_ratio=past_trend_ratio
        )

        period_dependent_harmonic_load_model_steps = numpy.zeros(
            shape=numpy.concatenate(
                (
                    [4],  # For the 5 steps. Last step is not period dependent.
                    elastic_load_model.base_products.time_dependent_component.shape,
                    elastic_load_model.base_products.load_model_harmonic_component.shape,
                ),
                axis=0,
            ),
            dtype=numpy.complex64,
        )

        # Step 1: unmodified signal.
        period_dependent_harmonic_load_model_steps[0] = numpy.tensordot(
            a=fft(elastic_load_model.base_products.time_dependent_component),
            b=elastic_load_model.base_products.load_model_harmonic_component,
            axes=0,
        )

        # Step 2: Signal corrected from the polar tide.
        period_dependent_harmonic_load_model_steps[1] = numpy.array(
            object=period_dependent_harmonic_load_model_steps[0]
        ).copy()

        if elastic_load_model.load_model_parameters.history.pole.use:

            period_dependent_harmonic_load_model_steps[1][
                :,
                2,
                1,
            ] -= c_2_1_pt_se_complex
            period_dependent_harmonic_load_model_steps[1][:, -3, -2] -= s_2_1_pt_se_complex

        # Step 3: All degrees re-estimated using the potential load Love number.
        period_dependent_harmonic_load_model_steps[2] = harmonic_load_model_re_estimation(
            elastic_love_numbers=elastic_love_numbers[
                :, : elastic_load_model.load_model_parameters.signature.n_max
            ],
            anelastic_love_numbers=anelastic_love_numbers[
                :, : elastic_load_model.load_model_parameters.signature.n_max
            ],
            period_dependent_harmonic_load_model_step_2=period_dependent_harmonic_load_model_steps[
                1
            ],
        )

        # Gets trend as a preprocessing for degree one inversion.
        harmonic_load_model_trend_step_3: numpy.ndarray = (
            get_trend_from_period_dependent_harmonic_model(
                period_dependent_harmonic_model=period_dependent_harmonic_load_model_steps[2],
                elastic_load_model=elastic_load_model,
            )
        )

        # Step 4: Degree one inversion.
        (
            period_dependent_harmonic_load_model_steps[3],
            period_dependent_inversion_components,
        ) = period_dependent_degree_one_inversion(
            love_numbers=anelastic_love_numbers[
                :, : elastic_load_model.load_model_parameters.signature.n_max
            ],
            period_dependent_harmonic_load_model=numpy.array(
                object=period_dependent_harmonic_load_model_steps[2]
            ).copy(),
            harmonic_load_model_trend=harmonic_load_model_trend_step_3,
            elastic_load_model=elastic_load_model,
            chunks=parallel_computing_parameters.degree_one_inversion_chunks,
        )

        # Gets step 4 past trends for step 5 computing.
        harmonic_load_model_past_trend: numpy.ndarray = (
            get_trend_from_period_dependent_harmonic_model(
                period_dependent_harmonic_model=period_dependent_harmonic_load_model_steps[3],
                elastic_load_model=elastic_load_model,
                recent_trend=False,
            )
        )

        # Step 5: Leakage correction on past trends.
        harmonic_load_model_past_trend = forward_modeling_leakage_correction(
            harmonic_load_model_trend=harmonic_load_model_past_trend,
            elastic_load_model=elastic_load_model,
            recent=False,
        )

        # Evaluates ocean mean past trend to check for normalization convergence.
        past_trend = get_ocean_mean_trend(
            harmonic_load_model_trend=harmonic_load_model_past_trend,
            elastic_load_model=elastic_load_model,
            recent_trend=False,
        )

        past_trend_ratio = (
            past_trend / elastic_load_model.base_products.temporal_products.target_past_trend
        )

    # Eventually saves intermediate products.
    return post_process_intermediate_load_model_products(
        period_dependent_harmonic_load_model_steps=period_dependent_harmonic_load_model_steps,
        period_dependent_inversion_components=period_dependent_inversion_components,
        save_informations=(elastic_load_model, anelastic_load_model_id),
        harmonic_load_model_trend_step_3=harmonic_load_model_trend_step_3,
    )


def post_process_intermediate_load_model_products(
    period_dependent_harmonic_load_model_steps: numpy.ndarray,
    period_dependent_inversion_components: dict[str, numpy.ndarray],
    save_informations: tuple[ElasticLoadModel, str],
    harmonic_load_model_trend_step_3: numpy.ndarray,
) -> tuple[list[float], dict[str, float]]:
    """
    Sub-function for readability. Computes trends and ocean mean trends and eventually saves
    spherical harmonics into (.JSON) files.
    """

    elastic_load_model, anelastic_load_model_id = save_informations

    # Performs all trend computations for output.
    harmonic_load_model_trend_steps = numpy.zeros(
        shape=period_dependent_harmonic_load_model_steps.shape[1:]
    )
    harmonic_load_model_trend_steps[2] = harmonic_load_model_trend_step_3

    for i_step in STEPS_TO_POST_PROCESS:

        harmonic_load_model_trend_steps[i_step] = get_trend_from_period_dependent_harmonic_model(
            period_dependent_harmonic_model=period_dependent_harmonic_load_model_steps[i_step],
            elastic_load_model=elastic_load_model,
        )

    harmonic_load_model_trend_steps[4] = forward_modeling_leakage_correction(
        harmonic_load_model_trend=harmonic_load_model_trend_steps[3],
        elastic_load_model=elastic_load_model,
    )

    geoid_deformation_harmonic_trends = get_trend_from_period_dependent_harmonic_model(
        period_dependent_harmonic_model=period_dependent_inversion_components["geoid_deformation"],
        elastic_load_model=elastic_load_model,
    )

    vertical_displacement_harmonic_trends = get_trend_from_period_dependent_harmonic_model(
        period_dependent_harmonic_model=period_dependent_inversion_components[
            "vertical_displacement"
        ],
        elastic_load_model=elastic_load_model,
    )

    # Eventually post-processes residuals.
    residuals_harmonic_trends = (
        None
        if not elastic_load_model.load_model_parameters.options.compute_residuals
        else get_trend_from_period_dependent_harmonic_model(
            period_dependent_harmonic_model=period_dependent_inversion_components["residuals"],
            elastic_load_model=elastic_load_model,
        )
    )

    # Eventually saves the processing steps.
    for i_step in range(5):

        if (
            elastic_load_model.load_model_parameters.options.save_options.all
            or elastic_load_model.load_model_parameters.options.save_options.steps[i_step]
        ):

            save_base_model(
                obj=make_grid(
                    harmonics=harmonic_load_model_trend_steps[i_step],
                    n_max=elastic_load_model.load_model_parameters.signature.n_max,
                ),
                name=anelastic_load_model_id,
                path=anelastic_load_models_path,
            )

    # Eventually saves the degree one inversion comonents.
    if elastic_load_model.load_model_parameters.options.save_options.inversion_components:

        save_base_model(
            obj=make_grid(
                harmonics=geoid_deformation_harmonic_trends,
                n_max=elastic_load_model.load_model_parameters.signature.n_max,
            ),
            name=anelastic_load_model_id,
            path=harmonic_geoid_deformation_trends_path,
        )

        save_base_model(
            obj=make_grid(
                harmonics=vertical_displacement_harmonic_trends,
                n_max=elastic_load_model.load_model_parameters.signature.n_max,
            ),
            name=anelastic_load_model_id,
            path=harmonic_vertical_displacement_trends_path,
        )

    # Eventually saves the residuals
    if elastic_load_model.load_model_parameters.options.compute_residuals:

        save_base_model(
            obj=make_grid(
                harmonics=residuals_harmonic_trends,
                n_max=elastic_load_model.load_model_parameters.signature.n_max,
            ),
            name=anelastic_load_model_id,
            path=harmonic_residual_trends_path,
        )

    # Computes ocean mean trends for output.
    load_model_ocean_trend_steps = []

    for i_step in range(5):

        load_model_ocean_trend_steps += [
            get_ocean_mean_trend(
                harmonic_load_model_trend=harmonic_load_model_trend_steps[i_step],
                elastic_load_model=elastic_load_model,
            )
        ]

    return (
        load_model_ocean_trend_steps,
        {
            "geoid_deformation_ocean_mean_trend": get_ocean_mean_trend(
                harmonic_load_model_trend=geoid_deformation_harmonic_trends,
                elastic_load_model=elastic_load_model,
            ),
            "vertical_deformation_ocean_mean_trend": get_ocean_mean_trend(
                harmonic_load_model_trend=vertical_displacement_harmonic_trends,
                elastic_load_model=elastic_load_model,
            ),
            "residuals_ocean_mean_trend": (
                None
                if not elastic_load_model.load_model_parameters.options.compute_residuals
                else get_ocean_mean_trend(
                    harmonic_load_model_trend=residuals_harmonic_trends,
                    elastic_load_model=elastic_load_model,
                )
            ),
        },
    )


def anelastic_load_model_re_estimation_processing_loop(
    elastic_load_models: list[ElasticLoadModel],
    elastic_love_numbers: numpy.ndarray,
    periods_id: str,
    rheological_model_id: str,
    parallel_computing_parameters: ParallelComputingParameters,
) -> None:
    """
    Given a rheological model and a list of elastic load model sharing periods, re-estimates a
    self-coherent load model assuming anelastic rheology.
    Needs the elastic load models to be preprocessed and the Love numbers and their corresponding
    purely elastic version to be already computed and coherently interpolated on the frequencies
    describing the given elastic load models.
    """

    anelastic_love_numbers = load_complex_array(
        path=interpolated_love_numbers_path(
            periods_id=periods_id, rheological_model_id=rheological_model_id
        )
    )

    for elastic_load_model in elastic_load_models:

        anelastic_load_model_id = generate_hash(
            input_dict={
                "rheological_model_id": rheological_model_id,
                "elastic_load_model_id": elastic_load_model.load_model_parameters.model_id(),
            }
        )

        if not is_in_table(table_name="anelastic_load_models", id_to_check=anelastic_load_model_id):

            # Memorizes the anelastic polar tide correction series before the normalization loop.
            c_2_1_pt_se_complex, s_2_1_pt_se_complex = polar_motion_correction(
                m_1=elastic_load_model.side_products.time_dependent_m_1,
                m_2=elastic_load_model.side_products.time_dependent_m_2,
                love_numbers=anelastic_love_numbers,
            )

            # Saves the anelastic polar tide correction time series for post-processing purposes.
            for complex_signal, name in zip(
                [c_2_1_pt_se_complex, s_2_1_pt_se_complex], ["C_2_1", "S_2_1"]
            ):

                save_complex_array(
                    obj=numpy.array(object=ifft(complex_signal), dtype=numpy.complex64),
                    name=name,
                    path=anelastic_pole_motion_path.joinpath(anelastic_load_model_id),
                )

            # Saves intermediate files and post-processes ocean mean trends.
            (
                load_model_ocean_trend_steps,
                degree_one_inversion_component_ocean_mean_trends,
            ) = anelastic_load_model_re_estimation_processing_steps(
                elastic_love_numbers=elastic_love_numbers,
                anelastic_love_numbers=anelastic_love_numbers,
                c_2_1_pt_se_complex=c_2_1_pt_se_complex,
                s_2_1_pt_se_complex=s_2_1_pt_se_complex,
                save_informations=(
                    elastic_load_model,
                    anelastic_load_model_id,
                    parallel_computing_parameters,
                ),
            )

            # Saves ocean mean trends to the main table.
            add_result_to_table(
                table_name="anelastic_load_models",
                dictionary=extract_terminal_attributes(obj=elastic_load_model.load_model_parameters)
                | {"rheological_model_id": rheological_model_id}
                | {
                    "ocean_mean_trend_step_" + str(i_step + 1): load_model_ocean_trend_step
                    for i_step, load_model_ocean_trend_step in enumerate(
                        load_model_ocean_trend_steps
                    )
                }
                | degree_one_inversion_component_ocean_mean_trends
                | {"ID": anelastic_load_model_id},
            )
