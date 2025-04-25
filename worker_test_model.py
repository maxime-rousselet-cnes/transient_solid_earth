"""
Defines a process/job for test model computing.
"""

from transient_solid_earth.adaptative_step_parallel_computing import WorkerInformation
from transient_solid_earth.database import load_base_model, save_base_model
from transient_solid_earth.paths import (
    intermediate_result_subpaths,
    logs_subpaths,
    worker_information_subpaths,
)
from transient_solid_earth.test_model import TestModel
from transient_solid_earth.worker_parser import parse_worker_args

if __name__ == "__main__":
    args = parse_worker_args()
    job_array_file_base_name = args.job_array_file_base_name
    job_array_max_file_size = args.job_array_max_file_size
    job_id = args.job_id

    # Gets all the worker's needed informations.
    worker_information = WorkerInformation(
        **load_base_model(
            name=str(job_id // job_array_max_file_size),
            path=worker_information_subpaths["test_model"].joinpath(job_array_file_base_name),
        )[job_id % job_array_max_file_size]
    )
    path = intermediate_result_subpaths["test_model"].joinpath(worker_information.model_id)

    # Check whether the task has already been computed.
    if not path.joinpath(str(worker_information.variable_parameter) + ".json").exists():

        # Processes and saves.
        model: TestModel = load_base_model(
            name=worker_information.model_id,
            path=logs_subpaths["test_model"].joinpath("models"),
            base_model_type=TestModel,
        )
        save_base_model(
            obj=model.process(variable_parameter=worker_information.variable_parameter),
            name=str(worker_information.variable_parameter),
            path=path,
        )
