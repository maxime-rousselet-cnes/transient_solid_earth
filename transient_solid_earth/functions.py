"""
Base mathematical functions.
"""

from typing import Optional

import numpy
from pyshtools import SHCoeffs
from pyshtools.shclasses import DHRealGrid
from scipy import interpolate

from .constants import MASK_DECIMALS


def round_value(t: float, rounding: int) -> float:
    """
    Truncature in scientific notation.
    """

    x = numpy.asarray(t)
    sign = numpy.sign(x)
    x_abs = numpy.abs(x)
    nonzero = x_abs != 0
    exp = numpy.zeros_like(x_abs, dtype=int)
    exp[nonzero] = numpy.floor(numpy.log10(x_abs[nonzero])).astype(int)
    factor = 10.0 ** (exp - rounding + 1)
    truncated = numpy.floor(x_abs / factor) * factor
    return sign * truncated


def add_sorted(
    result_dict: dict[str, numpy.ndarray], x: float, values: numpy.ndarray
) -> dict[str, numpy.ndarray]:
    """
    Insert a 'x' and 'values' in result_dict["x"] and result_dict["values"] respectively according
    to the order of result_dict["x"] elements.
    """

    position = 0
    while (position < len(result_dict["x"])) and (result_dict["x"][position] < x):
        position += 1
    return {
        "x": numpy.array(
            object=result_dict["x"][:position].tolist() + [x] + result_dict["x"][position:].tolist()
        ),
        "values": numpy.array(
            object=result_dict["values"][:position].tolist()
            + [values]
            + result_dict["values"][position:].tolist()
        ),
    }

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
