"""
Defines a process/job for Love number computing at a given frequency, a given degree for a given
model.
"""

import sys

from transient_solid_earth.database import load_base_model, save_base_model
from transient_solid_earth.love_numbers_computing import WorkerInformation
from transient_solid_earth.test_model import Model, test_models_path

if __name__ == "__main__":
    job_array_file_base_name = sys.argv[1]
    job_array_max_file_size = int(sys.argv[2])
    job_id = int(sys.argv[3])

    worker_information = WorkerInformation(
        **load_base_model(
            name=str(job_id // job_array_max_file_size),
            path=test_models_path.joinpath("worker_informations").joinpath(
                job_array_file_base_name
            ),
        )[job_id % job_array_max_file_size]
    )
    model: Model = load_base_model(
        name=worker_information.name,
        path=test_models_path.joinpath("models"),
        base_model_type=Model,
    )
    path = test_models_path.joinpath("intermediate_results").joinpath(model.name)
    if not path.joinpath(str(worker_information.t) + ".json").exists():
        save_base_model(
            obj=model.process(t=worker_information.t),
            name=str(worker_information.t),
            path=path,
        )
