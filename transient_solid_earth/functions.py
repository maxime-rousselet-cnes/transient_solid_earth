"""
Base mathematical functions.
"""

import numpy


def generate_n_factor(degrees: numpy.ndarray) -> numpy.ndarray:
    """
    Because nl_n and nk_n are better to interpolate.
    """

    n_factor = numpy.ones(shape=(len(degrees), 1, 1, 3))
    n_factor[:, :, :, 1] = numpy.array(object=degrees)[:, None, None]
    n_factor[:, :, :, 2] = n_factor[:, :, :, 1]

    return n_factor


def sum_lists(lists: list[list]) -> list:
    """
    Concatenates lists. Needed to iterate on parameter variations.
    """

    concatenated_list = []

    for elt in lists:

        for sub_elt in elt:

            concatenated_list += sub_elt

    return concatenated_list


def round_value(t: float, rounding: int) -> float:
    """
    Truncature in scientific notation.
    """

    x = numpy.asarray(t)
    sign = numpy.sign(x)
    x_abs = numpy.abs(x)
    exp = numpy.zeros_like(x_abs, dtype=int)

    # Valid values are finite and non-zero.
    valid = numpy.isfinite(x_abs) & (x_abs != 0)

    with numpy.errstate(divide="ignore", invalid="ignore"):

        exp[valid] = numpy.floor(numpy.log10(x_abs[valid])).astype(int)

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


def get_degrees_indices(degrees: list[int], degrees_to_plot: list[int]) -> list[int]:
    """
    Returns the indices of the wanted degrees in the list of degrees.
    """

    return [list(degrees).index(degree) for degree in degrees_to_plot]


def trend(trend_dates: numpy.ndarray[float], signal: numpy.ndarray[float]) -> tuple[float, float]:
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
    result: numpy.ndarray = numpy.linalg.pinv(a_matrix).dot(signal[:, None])

    return result.flatten()  # (slope, additive_constant).


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


def closest_index(array: numpy.ndarray, value: float) -> int:
    """
    Returns the index of the closest element in array to value.
    """

    return numpy.argmin(numpy.abs(array - value))


def chunkify_array(complex_array: numpy.ndarray, chunks: int) -> list[numpy.ndarray]:
    """
    Chunkifies a complex array for parallel computing.
    """

    n_periods = len(complex_array)

    chunk_length = int(numpy.ceil(n_periods // chunks))

    return [
        complex_array[
            chunk_id * chunk_length : min(n_periods, (chunk_id + 1) * chunk_length)
        ].copy()
        for chunk_id in range(chunks)
    ]
