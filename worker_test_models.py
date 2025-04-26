"""
Defines a process/job for test models computing.
"""

from transient_solid_earth import (
    TestModels,
    WorkerInformation,
    intermediate_result_subpaths,
    load_base_model,
    logs_subpaths,
    parse_worker_information,
    save_base_model,
)

if __name__ == "__main__":
    worker_information: WorkerInformation = parse_worker_information(function_name="test_models")
    path = intermediate_result_subpaths["test_models"].joinpath(worker_information.model_id)
    file: str = str(worker_information.variable_parameter)

    # Check whether the task has already been computed.
    if not path.joinpath(file + ".json").exists():

        # Processes and saves.
        model: TestModels = load_base_model(
            name=worker_information.model_id,
            path=logs_subpaths["test_models"].joinpath("models"),
            base_model_type=TestModels,
        )
        save_base_model(
            obj=model.process(variable_parameter=worker_information.variable_parameter),
            name=file,
            path=path,
        )
