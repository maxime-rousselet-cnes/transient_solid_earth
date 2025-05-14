"""
To be called for parallel processing.
"""

from os import utime

from pyshtools.legendre import legendre_lm

from .database import load_base_model, load_complex_array
from .paths import intermediate_result_subpaths
from .separators import is_elastic
from .worker_parser import WorkerInformation


def compute_green_functions(worker_information: WorkerInformation) -> None:
    """
    Computes Green functions for a given rheology.
    """

    # TODO.

    save_path = (
        intermediate_result_subpaths["green_functions"]
        .joinpath(worker_information.model_id)
        .joinpath(
            "inf"
            if is_elastic(model_id=worker_information.model_id)
            else str(worker_information.fixed_parameter)
        )
        .joinpath(str(worker_information.variable_parameter))
    )

    # Checks whether the task has already been computed.
    if save_path.exists():

        utime(path=save_path.joinpath("imag.json"))

    else:

        load_path = intermediate_result_subpaths["interpolate_love_numbers"].joinpath(
            worker_information.model_id
        )
        love_numbers = load_complex_array(path=load_path)
        periods = load_base_model(name="variable_parameter_values", path=load_path)
        degrees = load_base_model(name="fixed_parameter_values", path=load_path.parent)
