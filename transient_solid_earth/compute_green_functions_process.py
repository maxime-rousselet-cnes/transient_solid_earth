"""
To be called for parallel processing.
"""

from os import utime
from time import sleep

import numpy
from numpy.polynomial.legendre import legval

from .constants import EARTH_MASS, EARTH_RADIUS
from .database import load_complex_array, save_complex_array
from .paths import INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME, intermediate_result_subpaths
from .separators import is_elastic
from .worker_parser import WorkerInformation


def x_from_theta(theta: float) -> float:
    """
    Computes the x value from theta.
    """
    return numpy.sqrt(2.0 * (1.0 - numpy.cos(theta)))


def legendre_polynomials_sum_order_0(theta: float) -> float:
    """
    Computes the series of P_n(cos(theta)) using explicit formula.
    """

    return 1.0 / x_from_theta(theta=theta)


def legendre_polynomials_sum_order_1(theta: float) -> float:
    """
    Computes the series of (1/n) * P_n(cos(theta)) using explicit formula.
    """

    return numpy.log(2.0 / (x_from_theta(theta=theta) + 1.0 - numpy.cos(theta)))


def legendre_polynomial_derivatives_sum_order_1(theta: float) -> float:
    """
    Computes the series of d/dtheta(P_n(cos(theta))) using explicit formula.
    """

    x = x_from_theta(theta=theta)
    return -numpy.sin(theta) * (1.0 / x + 1.0) / (x + 1.0 - numpy.cos(theta))


def legendre_polynomial_derivatives_sum_order_2(theta: float) -> float:
    """
    Computes the series of (1/n) * d/dtheta(P_n(cos(theta))) using explicit formula.
    """

    x = x_from_theta(theta=theta)
    cos = numpy.cos(theta)
    sin = numpy.sin(theta)
    temp = x - 1.0 + cos
    return 1.0 / sin * numpy.log(
        (2.0 - x) ** 2.0 * (1.0 - cos) / (2 * temp * (1.0 + cos))
    ) - cos / sin * numpy.log(sin**2.0 / (2 * temp))


def compute_green_functions(worker_information: WorkerInformation) -> None:
    """
    Computes Green functions for a given rheology.
    """

    fixed_parameter = (
        numpy.inf
        if is_elastic(model_id=worker_information.model_id)
        else worker_information.fixed_parameter
    )

    save_path = (
        intermediate_result_subpaths["green_functions"]
        .joinpath(worker_information.model_id)
        .joinpath(str(fixed_parameter))
        .joinpath(str(worker_information.variable_parameter))
    )

    # Checks whether the task has already been computed.
    if save_path.exists():

        utime(path=save_path.joinpath("imag.json"))

    else:

        load_path = (
            intermediate_result_subpaths["interpolate_love_numbers"]
            .joinpath(worker_information.model_id)
            .joinpath(INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME)
        )

        def get_love_numbers() -> numpy.ndarray:
            """
            Intermediate function for concurrent I/O.
            """
            try:
                return load_complex_array(
                    path=load_path.joinpath(str(worker_information.fixed_parameter))
                )
            except FileNotFoundError:
                sleep(0.1)
                return get_love_numbers()

        love_numbers = get_love_numbers()

        asymptotic_love_numbers_order_0 = load_complex_array(
            path=intermediate_result_subpaths["asymptotic_love_numbers"]
            .joinpath(worker_information.model_id)
            .joinpath(str(fixed_parameter)),
            name="order_0",
        )
        asymptotic_love_numbers_order_1 = load_complex_array(
            path=intermediate_result_subpaths["asymptotic_love_numbers"]
            .joinpath(worker_information.model_id)
            .joinpath(str(fixed_parameter)),
            name="order_1",
        )

        degrees = numpy.arange(len(love_numbers)) + 1
        theta = numpy.pi / 180.0 * worker_information.variable_parameter

        u = (
            EARTH_RADIUS
            / EARTH_MASS
            * (
                asymptotic_love_numbers_order_0[:, 0]
                * legendre_polynomials_sum_order_0(theta=theta)
                + asymptotic_love_numbers_order_1[:, 1]
                * legendre_polynomials_sum_order_1(theta=theta)
                + legval(
                    x=numpy.cos(theta),
                    c=numpy.concatenate(
                        (
                            [[0.0, 0.0, 0.0]],
                            love_numbers[:, :, 0]
                            - asymptotic_love_numbers_order_0[:, 0][None, :]
                            - asymptotic_love_numbers_order_1[:, 0][None, :] / degrees[:, None],
                        )
                    ),
                )
            )
        )
        save_complex_array(obj=numpy.concatenate(([u],)), path=save_path)
