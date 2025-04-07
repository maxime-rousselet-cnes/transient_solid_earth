"""
Base mathematical functions.
"""

from typing import Callable, Optional

import numpy
from pyshtools import SHCoeffs
from pyshtools.shclasses import DHRealGrid
from scipy import interpolate

from .constants import MASK_DECIMALS


def precise_curvature(
    x_initial_values: numpy.ndarray,
    f: Callable[[float], numpy.ndarray[complex]],
    max_tol: float,
    decimals: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Finds a sufficiently precise sampling of x axis for f function representation.
    The criteria takes curvature into account is the error between the first and second orders.
    """

    # Initializes.
    x_new_values = x_initial_values
    x_values = numpy.array(object=[], dtype=float)
    f_values = numpy.array(object=[], dtype=complex)

    # Loops while there are still added abscissas.
    while len(x_new_values) != 0:
        # Calls f for new values only.
        f_new_values = f(x_new_values)

        # Updates.
        x_values = numpy.concatenate((x_values, x_new_values))
        f_values = (
            f_new_values if len(f_values) == 0 else numpy.concatenate((f_values, f_new_values))
        )
        order = numpy.argsort(x_values)
        x_values = x_values[order]
        f_values = f_values[order]
        x_new_values = numpy.array(object=[], dtype=float)

        # Iterates on new sampling.
        x_new_values = verify_curvature_condition(
            x_values=x_values, f_values=f_values, max_tol=max_tol, x_new_values=x_new_values
        )

        # Keeps only values that are not already taken into account.
        x_new_values = numpy.setdiff1d(
            ar1=numpy.unique(numpy.round(a=x_new_values, decimals=decimals)), ar2=x_values
        )

    return x_values, f_values


def verify_curvature_condition(
    x_values: numpy.ndarray, f_values: numpy.ndarray, max_tol: float, x_new_values: numpy.ndarray
) -> numpy.ndarray:
    """
    Re-sample the axis is the curvature is above a given threshold.
    """
    for f_left, f_x, f_right, x_left, x, x_right in zip(
        f_values[:-2],
        f_values[1:-1],
        f_values[2:],
        x_values[:-2],
        x_values[1:-1],
        x_values[2:],
    ):
        # For maximal curvature: finds where the error is above maximum threshold parameter and
        # adds median values.
        condition: numpy.ndarray = numpy.abs(
            (f_right - f_left) / (x_right - x_left) * (x - x_left) + f_left - f_x
        ) > max_tol * numpy.max(a=numpy.abs([f_left, f_x, f_right]), axis=0)
        if condition.any():
            # Updates sampling.
            x_new_values = numpy.concatenate(
                (x_new_values, [(x + x_left) / 2.0, (x + x_right) / 2.0])
            )
    return x_new_values


def interpolate_array(
    x_values: numpy.ndarray, y_values: numpy.ndarray, new_x_values: numpy.ndarray
) -> numpy.ndarray:
    """
    1D-Interpolates the given data on its first axis, whatever its shape is.
    """

    # Flattens all other dimensions.
    shape = y_values.shape
    y_values.reshape((shape[0], -1))
    components = y_values.shape[1]

    # Initializes
    function_values = numpy.zeros(shape=(len(new_x_values), components), dtype=complex)

    # Loops on components
    for i_component, component in enumerate(y_values.transpose()):

        # Creates callable (linear).
        function = interpolate.interp1d(x=x_values, y=component, kind="linear")

        # Calls linear interpolation on new x values.
        function_values[:, i_component] = function(x=new_x_values)

    #  Converts back into initial other dimension shapes.
    function_values.reshape((len(new_x_values), *shape[1:]))
    return function_values


def interpolate_all(
    x_values_per_component: list[numpy.ndarray],
    function_values: list[numpy.ndarray],
    x_shared_values: numpy.ndarray,
) -> numpy.ndarray:
    """
    Interpolate several function values on shared abscissas.
    """
    return numpy.array(
        object=(
            function_values
            if len(x_shared_values) == 1
            and x_shared_values[0] == numpy.inf  # Manages elastic case.
            else [
                interpolate_array(
                    x_values=x_tab,
                    y_values=function_values_tab,
                    new_x_values=x_shared_values,
                )
                for x_tab, function_values_tab in zip(x_values_per_component, function_values)
            ]
        )
    )


def get_degrees_indices(degrees: list[int], degrees_to_plot: list[int]) -> list[int]:
    """
    Returns the indices of the wanted degrees in the list of degrees.
    """
    return [list(degrees).index(degree) for degree in degrees_to_plot]


def signal_trend(
    trend_dates: numpy.ndarray[float], signal: numpy.ndarray[float]
) -> tuple[float, float]:
    """
    Returns signal's trend: mean slope and additive constant during last years (LSE).
    """
    # Assemble matrix A.
    a_matrix = numpy.vstack(
        [
            trend_dates,
            numpy.ones(len(trend_dates)),
        ]
    ).T
    # Direct least square regression using pseudo-inverse.
    result: numpy.ndarray = numpy.linalg.pinv(a_matrix).dot(signal[:, numpy.newaxis])
    return result.flatten()  # Turn the signal into a column vector. (slope, additive_constant)


def map_normalizing(
    map_array: numpy.ndarray,
) -> numpy.ndarray:
    """
    Sets global mean as zero and max as one by homothety.
    """
    n_t = numpy.prod(map_array.shape)
    sum_map = sum(map_array.flatten())
    max_map = numpy.max(map_array.flatten())
    return map_array / (max_map - sum_map / n_t) + sum_map / (sum_map - max_map * n_t)


def surface_ponderation(
    mask: numpy.ndarray[float], latitudes: numpy.ndarray[float]
) -> numpy.ndarray[float]:
    """
    Gets the surface of a (latitude * longitude) array.
    """
    return mask * numpy.expand_dims(a=numpy.cos(latitudes * numpy.pi / 180.0), axis=1)


def mean_on_mask(
    mask: numpy.ndarray[float],
    latitudes: numpy.ndarray[float],
    signal_threshold: float,
    harmonics: Optional[numpy.ndarray[float]] = None,
    grid: Optional[numpy.ndarray[float]] = None,
) -> float:
    """
    Computes mean value over a given surface. Uses a given mask.
    """
    if grid is None:
        grid: numpy.ndarray[float] = make_grid(harmonics=harmonics)
    surface = surface_ponderation(
        mask=mask * (numpy.abs(grid) < signal_threshold), latitudes=latitudes
    )
    weighted_values = grid * surface
    return numpy.round(
        a=sum((weighted_values).flatten()) / sum(surface.flatten()), decimals=MASK_DECIMALS
    )


def make_grid(
    harmonics: numpy.ndarray[float],
    n_max: Optional[int] = None,
) -> numpy.ndarray[float]:
    """
    Computes the 2D grid corresponding to the given spherical harmonics.
    """
    if n_max is None:
        n_max = harmonics.shape[1] - 1
    result: DHRealGrid = SHCoeffs.from_array(harmonics, lmax=n_max).expand(extend=True, lmax=n_max)
    return result.data


def closest_index(array: numpy.ndarray, value: float) -> int:
    """
    Returns the index of the closest element in array to value.
    """
    return numpy.argmin(numpy.abs(array - value))
