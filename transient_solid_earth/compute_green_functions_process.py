"""
To be called for parallel processing.
"""

from os import utime

import numpy
from pyshtools.legendre import legendre_lm

from .database import load_base_model, load_complex_array, save_complex_array
from .paths import INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME, intermediate_result_subpaths
from .separators import is_elastic
from .worker_parser import WorkerInformation


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

    print(
        worker_information.model_id,
        worker_information.fixed_parameter,
        worker_information.variable_parameter,
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
        periods: list = load_base_model(name="variable_parameter_values", path=load_path.parent)
        love_numbers = load_complex_array(path=load_path)[
            :, periods.index(worker_information.fixed_parameter)
        ]
        degrees = load_base_model(name="fixed_parameter_new_values", path=load_path.parent.parent)

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

        # TODO.
        save_complex_array(obj=numpy.zeros(shape=(3, 2)), path=save_path)
