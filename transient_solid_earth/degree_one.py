"""
Defines period dependent degree one inversion procedure from a spherical harmonics load model
(EWH; mm/yr).
"""

from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy
from scipy.linalg import lstsq

from .constants import DENSITY_RATIO, BoundaryCondition, Direction
from .elastic_load_models import ElasticLoadModel
from .formating import (
    make_grid,
    make_grid_from_unstacked,
    make_harmonics,
    stack_period_dependent_harmonics,
    surface_ponderation,
    unstack_period_dependent_harmonics,
)
from .functions import chunkify_array


def period_dependent_high_degrees_component(
    period_dependent_harmonic_load_model: numpy.ndarray,
    love_numbers: numpy.ndarray,
    degrees: numpy.ndarray,
    direction: Direction = Direction.POTENTIAL,
) -> numpy.ndarray:
    """
    Builds the sea level equation components in the Spherical harmonics domain.
    """

    one_line = (
        numpy.ones(shape=(love_numbers.shape[0], 1), dtype=numpy.complex64)
        if direction == Direction.POTENTIAL
        else numpy.zeros(shape=(love_numbers.shape[0], 1), dtype=numpy.complex64)
    )

    divisor = (2 * degrees[None, :] + 1).astype(numpy.complex64)

    arr = numpy.concatenate(
        (
            one_line,
            (
                (1.0 if direction == Direction.POTENTIAL else 0.0)
                + love_numbers[:, :, BoundaryCondition.LOAD.value, direction.value]
            ),
        ),
        axis=1,
    )

    scaled = arr / divisor  # shape: (periods, degrees)

    return (
        3
        * DENSITY_RATIO
        * numpy.multiply(
            scaled[:, None, :, None],
            unstack_period_dependent_harmonics(
                dense_period_dependent_harmonics=period_dependent_harmonic_load_model
            ),
        )
    )


def make_low_degree_polynomials(ocean_mask_indices: numpy.ndarray, n_max: int) -> numpy.ndarray:
    """
    Creates low degree spherical harmonic polynomials flattened and masked.
    """

    p_1_0 = make_grid_from_unstacked(
        harmonics=numpy.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]), n_max=n_max
    )
    p_1_1_c = make_grid_from_unstacked(
        harmonics=numpy.array([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]), n_max=n_max
    )
    p_1_1_s = make_grid_from_unstacked(
        harmonics=numpy.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]), n_max=n_max
    )
    p_0 = make_grid_from_unstacked(
        harmonics=numpy.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]), n_max=n_max
    )
    p_2_0 = make_grid_from_unstacked(
        harmonics=numpy.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        n_max=n_max,
    )

    low_degrees_polynomials = numpy.array(
        [
            p_1_0.flatten()[ocean_mask_indices],
            p_1_1_c.flatten()[ocean_mask_indices],
            p_1_1_s.flatten()[ocean_mask_indices],
            p_0.flatten()[ocean_mask_indices],
            p_2_0.flatten()[ocean_mask_indices],
        ],
        dtype=numpy.float32,
    )
    return low_degrees_polynomials


def parallel_period_dependent_degree_one_inversion(
    period_dependent_harmonic_load_model_chunk: numpy.ndarray,
    love_numbers_chunk: numpy.ndarray,
    elastic_load_model: ElasticLoadModel,
    harmonic_load_model_trend: numpy.ndarray,
) -> tuple[
    numpy.ndarray,  # Contains c_1_0, c_1_1 and s_1_1.
    Optional[numpy.ndarray],
    numpy.ndarray,
    numpy.ndarray,
    Optional[numpy.ndarray],
]:
    """
    Degree one inversion job for a chunk of the load model, parallelized over periods.
    """

    mask: (
        numpy.ndarray
    ) = elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask.astype(bool) & (
        numpy.abs(
            make_grid(
                harmonics=harmonic_load_model_trend,
                n_max=elastic_load_model.load_model_parameters.signature.n_max,
            )
        )
        < elastic_load_model.load_model_parameters.numerical_parameters.ewh_threshold
    )
    ocean_mask_indices = mask.flatten()

    least_square_weights = (
        surface_ponderation(
            mask=mask,
            latitudes=elastic_load_model.elastic_load_model_spatial_products.latitudes,
        ).flatten()[ocean_mask_indices]
        ** 0.5
    ).astype(numpy.complex64)

    if elastic_load_model.load_model_parameters.options.compute_residuals:
        lat_idx = numpy.arange(
            len(elastic_load_model.elastic_load_model_spatial_products.latitudes), dtype=numpy.int32
        )
        lon_idx = numpy.arange(
            len(elastic_load_model.elastic_load_model_spatial_products.longitudes),
            dtype=numpy.int32,
        )
        lat_mesh, lon_mesh = numpy.meshgrid(lat_idx, lon_idx, indexing="ij")
        lat_mesh_ocean = lat_mesh.flatten()[ocean_mask_indices]
        lon_mesh_ocean = lon_mesh.flatten()[ocean_mask_indices]

    degrees = numpy.arange(
        stop=elastic_load_model.load_model_parameters.signature.n_max + 1, dtype=numpy.int32
    )

    # Compute geoid deformation and vertical displacement.
    period_dependent_geoid_deformation = period_dependent_high_degrees_component(
        period_dependent_harmonic_load_model=period_dependent_harmonic_load_model_chunk,
        love_numbers=love_numbers_chunk,
        degrees=degrees,
    )

    period_dependent_vertical_displacement = period_dependent_high_degrees_component(
        period_dependent_harmonic_load_model=period_dependent_harmonic_load_model_chunk,
        love_numbers=love_numbers_chunk,
        degrees=degrees,
        direction=Direction.VERTICAL,
    )

    period_dependent_right_hand_side = (
        period_dependent_geoid_deformation
        - period_dependent_vertical_displacement
        - unstack_period_dependent_harmonics(
            dense_period_dependent_harmonics=period_dependent_harmonic_load_model_chunk
        )
    )

    low_degrees_polynomials = make_low_degree_polynomials(
        ocean_mask_indices=ocean_mask_indices,
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )

    invert_for_j2 = elastic_load_model.load_model_parameters.options.invert_for_J2
    n_chunk = period_dependent_harmonic_load_model_chunk.shape[0]

    # Precompute basis vectors for stacking along last axis
    basis_vectors = [
        -period_dependent_right_hand_side[:, 0, 1, 0],
        -period_dependent_right_hand_side[:, 0, 1, 1],
        -period_dependent_right_hand_side[:, 1, 1, 1],
        numpy.ones(n_chunk, dtype=numpy.complex64),
    ]

    if invert_for_j2:

        basis_vectors.append(-period_dependent_right_hand_side[:, 0, 2, 0].astype(numpy.complex64))

    period_dependent_left_hand_side = (
        least_square_weights[None, :]
        * low_degrees_polynomials[: 5 if invert_for_j2 else 4, :]  # (1, n_points, n_poly)
    ).T[None, :, :] * numpy.array(basis_vectors).T[
        :, None, :  # (n_chunk, 1, n_poly).
    ]  # (n_chunk, n_points, n_poly)

    # Resets degree 1 harmonics in right hand side to zero.
    period_dependent_right_hand_side[:, :, :2, :] = 0.0

    period_dependent_degree_one = numpy.zeros((n_chunk, 2, 2), dtype=numpy.complex64)

    if invert_for_j2:

        period_dependent_right_hand_side[:, 0, 2, 0] = 0.0  # Try without
        j_2 = numpy.zeros(n_chunk, dtype=numpy.complex64)

    else:

        j_2 = None

    if elastic_load_model.load_model_parameters.options.compute_residuals:

        residuals = numpy.zeros(
            period_dependent_harmonic_load_model_chunk.shape, dtype=numpy.complex64
        )
        grid = numpy.zeros(lat_mesh.shape, dtype=numpy.complex64)

    else:

        residuals = None

    left_hand_side: numpy.ndarray
    harmonic_right_hand_side: numpy.ndarray

    for i_period, (left_hand_side, harmonic_right_hand_side) in enumerate(
        zip(period_dependent_left_hand_side, period_dependent_right_hand_side)
    ):

        spatial_right_hand_side_real = make_grid_from_unstacked(
            harmonics=harmonic_right_hand_side.real,
            n_max=elastic_load_model.load_model_parameters.signature.n_max,
        )
        spatial_right_hand_side_imag = make_grid_from_unstacked(
            harmonics=harmonic_right_hand_side.imag,
            n_max=elastic_load_model.load_model_parameters.signature.n_max,
        )
        spatial_right_hand_side = spatial_right_hand_side_real + 1j * spatial_right_hand_side_imag
        right_hand_side = (
            least_square_weights
            * spatial_right_hand_side.astype(numpy.complex64).flatten()[ocean_mask_indices]
        )[:, None]

        # Solves the least squares.
        solution_vector, _, _, _ = lstsq(left_hand_side, right_hand_side)

        period_dependent_degree_one[i_period, :, :] = numpy.array(
            [[solution_vector[0, 0], solution_vector[1, 0]], [0.0, solution_vector[2, 0]]],
            dtype=numpy.complex64,
        )

        if invert_for_j2:

            j_2[i_period] = solution_vector[4, 0]

        if elastic_load_model.load_model_parameters.options.compute_residuals:

            grid[lat_mesh_ocean, lon_mesh_ocean] = (
                left_hand_side @ solution_vector - right_hand_side
            ).flatten()
            residuals[i_period, :, :] = make_harmonics(
                grid=grid, n_max=elastic_load_model.load_model_parameters.signature.n_max
            )

    # Post-process degree one inversion components.
    period_dependent_geoid_deformation[:, :, 1, :2] = (
        DENSITY_RATIO
        * (1.0 + love_numbers_chunk[:, 0, BoundaryCondition.LOAD.value, Direction.POTENTIAL.value])
        * period_dependent_degree_one
    )
    period_dependent_vertical_displacement[:, :, 1, :2] = (
        DENSITY_RATIO
        * love_numbers_chunk[:, 0, BoundaryCondition.LOAD.value, Direction.VERTICAL.value]
        * period_dependent_degree_one
    )

    if invert_for_j2:

        period_dependent_geoid_deformation[:, 0, 2, 0] = (
            3
            / 5
            * DENSITY_RATIO
            * (
                1.0
                + love_numbers_chunk[:, 1, BoundaryCondition.LOAD.value, Direction.POTENTIAL.value]
            )
            * j_2
        )
        period_dependent_vertical_displacement[:, 0, 2, 0] = (
            3
            / 5
            * DENSITY_RATIO
            * love_numbers_chunk[:, 1, BoundaryCondition.LOAD.value, Direction.VERTICAL.value]
            * j_2
        )

    return (
        period_dependent_degree_one,
        j_2,
        stack_period_dependent_harmonics(
            period_dependent_geoid_deformation.astype(numpy.complex64)
        ),
        stack_period_dependent_harmonics(
            period_dependent_vertical_displacement.astype(numpy.complex64)
        ),
        residuals,
    )


def period_dependent_degree_one_inversion(
    love_numbers: numpy.ndarray,
    period_dependent_harmonic_load_model: numpy.ndarray,
    harmonic_load_model_trend: numpy.ndarray,
    elastic_load_model: ElasticLoadModel,
    chunks: int,
) -> tuple[numpy.ndarray, dict[str, numpy.ndarray]]:
    """
    Performs a degree one inversion for all periods.
    Distributes the inversion along the period axis.
    Returns:
        - The period dependent load model with inverted degree one in the harmonic domain.
        - The period dependent inversion components as a dictionary:
            - geoid deformation.
            - vertical displacement.
            - residuals (if computed).
    """

    period_dependent_harmonic_load_model[:, 1, 0] = 1.0
    period_dependent_harmonic_load_model[:, 1, 1] = 1.0
    period_dependent_harmonic_load_model[:, -2, -1] = 1.0

    if elastic_load_model.load_model_parameters.options.invert_for_J2:

        period_dependent_harmonic_load_model[:, 2, 0] = 1.0

    i_period = 0
    geoid_deformation_chunks = []
    vertical_displacement_chunks = []
    residuals_chunks = (
        [] if elastic_load_model.load_model_parameters.options.compute_residuals else None
    )

    if love_numbers.shape[0] == 1:  # Purely elastic case.

        love_number_chunks = [love_numbers] * chunks

    else:

        love_number_chunks = chunkify_array(complex_array=love_numbers, chunks=chunks)

    model_chunks = chunkify_array(complex_array=period_dependent_harmonic_load_model, chunks=chunks)

    with Pool(processes=cpu_count()) as p:

        results = p.starmap(
            parallel_period_dependent_degree_one_inversion,
            [
                (model_chunk, love_chunk, elastic_load_model, harmonic_load_model_trend)
                for model_chunk, love_chunk in zip(model_chunks, love_number_chunks)
            ],
        )

    for (
        inverted_degree_one,
        j_2,
        geoid_deformation_chunk,
        vertical_displacement_chunk,
        residuals_chunk,
    ) in results:

        n_periods = len(inverted_degree_one)
        period_dependent_harmonic_load_model[i_period : i_period + n_periods, 1, 0] = (
            inverted_degree_one[:, 0, 0]
        )
        period_dependent_harmonic_load_model[i_period : i_period + n_periods, 1, 1] = (
            inverted_degree_one[:, 0, 1]
        )
        period_dependent_harmonic_load_model[i_period : i_period + n_periods, -2, -1] = (
            inverted_degree_one[:, 1, 1]
        )

        if j_2 is not None:

            period_dependent_harmonic_load_model[i_period : i_period + n_periods, 2, 0] = j_2

        geoid_deformation_chunks.append(geoid_deformation_chunk)
        vertical_displacement_chunks.append(vertical_displacement_chunk)

        if residuals_chunk is not None:

            residuals_chunks.append(residuals_chunk)

        i_period += n_periods

    geoid_deformation = numpy.vstack(geoid_deformation_chunks)
    vertical_displacement = numpy.vstack(vertical_displacement_chunks)

    if residuals_chunks is not None:

        residuals = numpy.vstack(residuals_chunks)
    else:

        residuals = None

    return (
        period_dependent_harmonic_load_model,
        {
            "geoid_deformation": geoid_deformation,
            "vertical_displacement": vertical_displacement,
            "residuals": residuals,
        },
    )
