"""
To be called for parallel processing.
"""

from os import utime

from .database import load_base_model, save_base_model
from .paths import intermediate_result_subpaths, logs_subpaths
from .test_models import TestModel
from .worker_parser import WorkerInformation


def compute_test_models(worker_information: WorkerInformation) -> None:
    """
    Test models dummy function for a given rheology, degree and period.
    """

    path = (
        intermediate_result_subpaths["test_models"]
        .joinpath(worker_information.model_id)
        .joinpath(str(worker_information.fixed_parameter))
        .joinpath(str(worker_information.variable_parameter))
    )

    # Checks whether the task has already been computed.
    if path.exists():

        utime(path=path.joinpath("imag.json"))

    else:

        # Processes and saves.
        model: TestModel = load_base_model(
            name=worker_information.model_id,
            path=logs_subpaths["test_models"].joinpath("models"),
            base_model_type=TestModel,
        )
        save_base_model(
            obj=model.process(
                fixed_parameter=worker_information.fixed_parameter,
                variable_parameter=worker_information.variable_parameter,
            ),
            name="real",
            path=path,
        )
        save_base_model(
            obj=model.process(
                fixed_parameter=worker_information.fixed_parameter,
                variable_parameter=worker_information.variable_parameter,
            ),
            name="imag",
            path=path,
        )
