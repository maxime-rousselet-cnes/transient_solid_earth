"""
Sub-functions for a readable main loop.
"""

import numpy
from pandas import read_csv
from scipy.fft import ifft

from .constants import EMPIRICAL_INTERPOLATION_TIEMOUT_FACTOR
from .database import load_complex_array, save_complex_array
from .load_signal_model import ElasticLoadModel, load_elastic_load_model
from .parameters import Parameters, generate_hash
from .paths import (
    anelastic_load_signals_path,
    anelastic_pole_motion_path,
    elastic_load_models_path,
    harmonic_geoid_trends_path,
    harmonic_radial_displacement_trends_path,
    harmonic_residual_trends_path,
    interpolated_love_numbers_path,
    tables_path,
)
from .polar_tide import polar_motion_correction
from .separators import is_elastic


def get_interpolation_timeout(period_new_values_per_id: dict[str, numpy.ndarray[float]]) -> float:
    """
    Empirical timeout parallel Love number interpolation.
    """

    return max(
        len(period_new_values) / EMPIRICAL_INTERPOLATION_TIEMOUT_FACTOR
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
    parameters.load.model.signature.n_max = 0
    load_models = {}

    for file in elastic_load_models_path.glob("*.json"):

        load_model = load_elastic_load_model(model_id=file.name)
        interpolation_basis_ids[file.name] = (
            load_model.load_model_parameters.interpolation_basis_id()
        )
        interpolation_basis[load_model.load_model_parameters.interpolation_basis_id()] = (
            load_model.elastic_load_model_base_products.elastic_load_model_temporal_products.periods
        )
        parameters.load.model.signature.n_max = max(
            parameters.load.model.signature.n_max, load_model.load_model_parameters.signature.n_max
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


def is_in_table(table_name: str, id_to_check: str) -> bool:
    """
    Verify if a given ID is in a (.CSV) table.
    """

    file = tables_path.joinpath(table_name + ".csv")

    if file.exists():

        df = read_csv(file)

        return id_to_check in df["ID"]

    return False


def renormalize_past_temporal_component(
    elastic_load_model: ElasticLoadModel, past_trend_ratio: float
) -> None:
    """
    Normalizes the past part of the temporal component by a given unitless ratio.
    The trend on recent time period still corresponds to 1.0 (unitless).
    """

    last_past_time_period_value = (
        elastic_load_model.elastic_load_model_base_products.time_dependent_component[
            elastic_load_model.elastic_load_model_side_products.past_trend_indices[-1]
        ]
    )
    last_recent_time_period_value = (
        elastic_load_model.elastic_load_model_base_products.time_dependent_component[
            elastic_load_model.elastic_load_model_side_products.recent_trend_indices[-1]
        ]
    )
    additive_constant = last_past_time_period_value * (1.0 - 1.0 / past_trend_ratio)

    # Multiplicative factor on past time period.
    elastic_load_model.elastic_load_model_base_products.time_dependent_component[
        elastic_load_model.elastic_load_model_side_products.past_trend_indices
    ] /= past_trend_ratio
    elastic_load_model.elastic_load_model_base_products.time_dependent_component[
        -elastic_load_model.elastic_load_model_side_products.past_trend_indices
    ] /= past_trend_ratio

    # Additive constant on recent time period to maintain continuity.
    elastic_load_model.elastic_load_model_base_products.time_dependent_component[
        elastic_load_model.elastic_load_model_side_products.recent_trend_indices
    ] -= additive_constant
    elastic_load_model.elastic_load_model_base_products.time_dependent_component[
        -elastic_load_model.elastic_load_model_side_products.recent_trend_indices
    ] += additive_constant

    # Multiplicative factor on antisymmetric interpolation spline to maintain continuity.
    elastic_load_model.elastic_load_model_base_products.time_dependent_component[
        elastic_load_model.elastic_load_model_side_products.recent_trend_indices[-1]
        + 1 : -elastic_load_model.elastic_load_model_side_products.recent_trend_indices[-1]
    ] *= (1.0 - additive_constant / last_recent_time_period_value)


def anelastic_load_model_re_estimation_processing_steps(
    elastic_load_model: ElasticLoadModel, rheological_model_id: str
) -> None:
    """
    In a Newton's method loop to normalize the past barystatic trend, performs:
        -   Anelastic self-coherent time-dependent polar tide correction.
        -   Anelastic self-coherent load model re-estimation using the potential load Love number
            on all degrees.
        -   Anelastic self-coherent degree one inversion.
        -   Leakage correction.
    Iterates on the past trend normalization factor until the post-processed past trend fits to the
    barystatic load signal history.
    """

    # Initializes to enter the loop.
    past_trend = numpy.inf
    elastic_load_model_temporal_products = (
        elastic_load_model.elastic_load_model_base_products.elastic_load_model_temporal_products
    )

    past_trend_ratio = (
        1.0
        if is_elastic(model_id=rheological_model_id)
        else elastic_load_model.load_model_parameters.numerical_parameters.initial_past_trend_factor
    )

    # Iterates on past_trend_ratio for convergence on past_trend.
    while (
        abs(past_trend - elastic_load_model_temporal_products.target_past_trend)
        > elastic_load_model.load_model_parameters.numerical_parameters.past_trend_error
    ):

        renormalize_past_temporal_component(
            elastic_load_model=elastic_load_model, past_trend_ratio=past_trend_ratio
        )

    # TODO:
    # - - Renormalize past signal by dividing by past_trend_ratio.
    # - - Get step 1 to 5 and degree one inversion components. Dummy.
    # - - Compute ocean mean past trend.
    # - - Overwrite past_trend_ratio by past trend over target_past_trend.
    # - Compute needed means.


def anelastic_load_model_re_estimation_processing_loop(
    elastic_load_models: list[ElasticLoadModel],
    elastic_love_numbers: numpy.ndarray,
    periods_id: str,
    rheological_model_id: str,
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
            c_2_1_complex, s_2_1_complex = polar_motion_correction(
                m_1=elastic_load_model.elastic_load_model_side_products.time_dependent_m_1,
                m_2=elastic_load_model.elastic_load_model_side_products.time_dependent_m_2,
                love_numbers=anelastic_love_numbers,
            )

            # Saves the anelastic polar tide correction time series for post-processing purposes.
            for complex_signal, name in zip([c_2_1_complex, s_2_1_complex], ["C_2_1", "S_2_1"]):

                save_complex_array(
                    obj=numpy.array(object=ifft(complex_signal), dtype=complex),
                    name=name,
                    path=anelastic_pole_motion_path.joinpath(anelastic_load_model_id),
                )

            anelastic_load_model_re_estimation_processing_steps(
                elastic_load_model=elastic_load_model, rheological_model_id=rheological_model_id
            )

    # TODO:
    # - Save to table.
    # - Eventually save step trends and degree one inversion component trends.
    # Test, lint and back.
    # Rewrite:
    # - Step 2: Polar tide correction.
    # - Step 3: Anelastic re-estimation.
    # - Step 4: Degree one inversion.
    # - Step 5: Leakage correction.
