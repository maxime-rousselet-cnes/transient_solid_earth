"""
To be called for parallel processing.
"""

from os import utime

from .paths import intermediate_result_subpaths
from .separators import is_elastic
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel
from .solid_earth_time_dependent_numerical_model import SolidEarthTimeDependentNumericalModel
from .worker_parser import WorkerInformation


def worker_compute_love_numbers(worker_information: WorkerInformation) -> None:
    """
    Love number integration function for a given rheology, degree and period.
    """

    path = (
        intermediate_result_subpaths["love_numbers"]
        .joinpath(worker_information.model_id)
        .joinpath(str(worker_information.fixed_parameter))
    ).joinpath(
        "inf"
        if is_elastic(model_id=worker_information.model_id)
        else str(worker_information.variable_parameter)
    )

    # Checks whether the task has already been computed.
    if path.exists():

        utime(path=path.joinpath("imag.json"))

    else:

        solid_earth_full_numerical_model = SolidEarthFullNumericalModel(
            model_id=worker_information.model_id, load_numerical_model=True
        )

        # Loads the full numerical model, processes and saves.
        solid_earth_time_dependent_numerical_model = SolidEarthTimeDependentNumericalModel(
            solid_earth_full_numerical_model=solid_earth_full_numerical_model,
            period=worker_information.variable_parameter,
            n=worker_information.fixed_parameter,
        )

        solid_earth_time_dependent_numerical_model.integrate_y_i_systems()
