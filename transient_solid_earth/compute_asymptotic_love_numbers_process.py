"""
To be called for parallel processing.
"""

import numpy

from .database import load_base_model, save_complex_array
from .functions import generate_n_factor
from .paths import intermediate_result_subpaths
from .worker_parser import WorkerInformation


def fit_limit_inverse_model(
    n_values: numpy.ndarray | list, y_values: numpy.ndarray | list
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Fits a model f(n) = l + a / n using linear least squares.
    Supports real/complex and multi-dimensional y_values.
    """

    n_values = numpy.asarray(n_values)
    y_values = numpy.asarray(y_values)

    # Fits a model f(n) = l + a / n.
    x = numpy.column_stack((numpy.ones_like(n_values), 1.0 / n_values))
    x = x.astype(numpy.result_type(x, y_values))  # Matches dtype (real or complex).

    y_shape = y_values.shape
    n_samples = y_shape[0]
    y_values_flat = y_values.reshape(n_samples, -1)

    coeffs, *_ = numpy.linalg.lstsq(x, y_values_flat, rcond=None)

    output_shape = y_shape[1:]  # Shape after axis 0.
    l: numpy.ndarray = coeffs[0]
    a: numpy.ndarray = coeffs[1]
    return l.reshape(output_shape), a.reshape(output_shape)


def compute_asymptotic_love_numbers(worker_information: WorkerInformation) -> None:
    """
    Computes a Richardson equivalent of Love numbers for a given rheology.
    """

    save_path = intermediate_result_subpaths["asymptotic_love_numbers"].joinpath(
        worker_information.model_id
    )

    # Checks whether the task has already been computed.
    if save_path.exists():

        return

    n_min_for_asymptotic_behavior = worker_information.fixed_parameter
    load_path = intermediate_result_subpaths["interpolate_love_numbers"].joinpath(
        worker_information.model_id
    )
    love_numbers = load_base_model(name="real", path=load_path)
    periods = love_numbers["variable_parameter"]
    degrees = love_numbers["fixed_parameter"]
    min_index_for_asymptotic_behavior = numpy.searchsorted(
        degrees, n_min_for_asymptotic_behavior, side="left"
    )
    values = numpy.array(object=love_numbers["values"]) + 1.0j * numpy.array(
        object=load_base_model(name="imag", path=load_path)["values"]
    )

    for i_period, period in enumerate(periods):

        # Computes the asymptotic Love numbers.
        asymptotic_love_numbers_order_0, asymptotic_love_numbers_order_1 = fit_limit_inverse_model(
            n_values=degrees[min_index_for_asymptotic_behavior:],
            y_values=values[min_index_for_asymptotic_behavior:, i_period]
            * generate_n_factor(fixed_parameter_values=degrees)[
                min_index_for_asymptotic_behavior:, 0
            ],
        )

        # Saves the results.
        path = save_path.joinpath(str(period))
        save_complex_array(
            obj=asymptotic_love_numbers_order_0,
            path=path,
            name="order_0",
        )
        save_complex_array(
            obj=asymptotic_love_numbers_order_1,
            path=path,
            name="order_1",
        )
