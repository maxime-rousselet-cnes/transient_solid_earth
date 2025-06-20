"""
Defines a leakage correction procedure from a spherical harmonics load model (EWH; mm/yr).
"""

import numpy
from pandas import DataFrame

# pylint: disable=no-name-in-module
from pyGFOToolbox.processing.filter.filter_ddk import _pool_apply_DDK_filter

from .elastic_load_models import ElasticLoadModel
from .formating import make_grid, make_harmonics, stack_harmonics, unstack_harmonics
from .trends import get_ocean_mean_trend


def collection_sh_data_from_grid(grid: numpy.ndarray[float], n_max: int) -> DataFrame:
    """
    Converts a grid (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1) into a Collection of Spherical
    harmonics instance from pyGFOToolbox.
    """

    degrees = numpy.arange(n_max + 1)
    fake_frequencies_mesh, degrees_mesh, orders_mesh = numpy.meshgrid(
        [0], degrees, degrees, indexing="ij"
    )
    harmonic_load_signal = unstack_harmonics(dense_harmonics=make_harmonics(grid=grid, n_max=n_max))
    c: numpy.ndarray[float] = harmonic_load_signal[0]
    s: numpy.ndarray[float] = harmonic_load_signal[1]
    data = numpy.array(
        object=[
            fake_frequencies_mesh.flatten(),
            degrees_mesh.flatten(),
            orders_mesh.flatten(),
            c.flatten(),
            s.flatten(),
        ]
    ).T
    collection_data = DataFrame(data, columns=["date", "degree", "order", "C", "S"])
    for column in ["date", "degree", "order"]:
        collection_data[column] = collection_data[column].astype(int)
    collection_data = collection_data.set_index(["date", "degree", "order"]).sort_index()

    return collection_data


def grid_from_collection_sh_data(
    collection_data: DataFrame,
    n_max: int,
) -> numpy.ndarray[float]:  # (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1) - shaped.
    """
    Converts a Collection of Spherical harmonics instance from pyGFOToolbox into a
    (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1) grid.
    """

    return make_grid(
        harmonics=stack_harmonics(
            harmonics=[
                numpy.array(object=coeffs).reshape((n_max + 1, n_max + 1))
                for coeffs in collection_data.to_numpy().T
            ]
        ),
        n_max=n_max,
    )


def forward_modeling_leakage_correction(
    harmonic_load_model_trend: numpy.ndarray[float],
    elastic_load_model: ElasticLoadModel,
    recent: bool = True,
) -> numpy.ndarray[float]:
    """
    Performs a leakage correction procedure via forward modeling.
    """

    load_model_parameters = elastic_load_model.load_model_parameters

    # Gets the input in spatial domain.
    grid: numpy.ndarray[complex] = make_grid(
        harmonics=harmonic_load_model_trend,
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )

    # Oceanic true level.
    ocean_true_level = get_ocean_mean_trend(
        harmonic_load_model_trend=harmonic_load_model_trend,
        elastic_load_model=elastic_load_model,
        recent_trend=recent,
    )

    # Iterates a leakage correction procedure as many times as asked for.
    for _ in range(load_model_parameters.numerical_parameters.leakage_correction_iterations):

        mask_non_oceanic_signal = (
            elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask
            * (
                abs(grid)
                > (
                    load_model_parameters.numerical_parameters.ewh_threshold
                    if recent
                    else load_model_parameters.numerical_parameters.ewh_threshold_past
                )
            )
            # Continental mask as complementary.
            + (1 - elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask)
        )

        # Leakage input.
        ewh_2_prime: numpy.ndarray[float] = (
            ocean_true_level * (1 - mask_non_oceanic_signal) + grid * mask_non_oceanic_signal
        )
        ewh_2_third: numpy.ndarray[float] = ocean_true_level * (
            1 - mask_non_oceanic_signal
        ) + grid * (1 - elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask)

        # Computes continental leakage on oceans.
        ewh_2_second: numpy.ndarray[float] = grid_from_collection_sh_data(
            collection_data=_pool_apply_DDK_filter(
                grace_monthly_sh=collection_sh_data_from_grid(
                    grid=ewh_2_prime,
                    n_max=load_model_parameters.signature.n_max,
                ),
                ddk_filter_level=load_model_parameters.numerical_parameters.ddk_filter_level,
            ),
            n_max=load_model_parameters.signature.n_max,
        )

        # Applies correction.
        differential_term: numpy.ndarray[float] = ewh_2_second - ewh_2_third
        grid += (
            differential_term
            * (1 - elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask)
            - differential_term
            * elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask
        )

    # Gets the result back in spherical harmonics domain.
    return make_harmonics(grid=grid, n_max=load_model_parameters.signature.n_max)
