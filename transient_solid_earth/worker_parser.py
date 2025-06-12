"""
For all worker types to have same signature.
"""

import argparse
import dataclasses

from pydantic import BaseModel

from transient_solid_earth.database import load_base_model
from transient_solid_earth.paths import worker_information_subpaths


@dataclasses.dataclass
class WorkerArguments:
    """
    Needed information for a single worker to process its jobs.
    """

    job_array_file_base_name: str
    job_array_max_file_size: int
    job_id: int


def parse_worker_args() -> WorkerArguments:
    """
    Parses the command line arguments for a worker of any specific kind.
    """

    parser = argparse.ArgumentParser(description="Process worker job array parameters.")
    parser.add_argument(
        "job_array_file_base_name", type=str, help="Base name of the job array file path."
    )
    parser.add_argument("job_array_max_file_size", type=int, help="Maximum job array file size.")
    parser.add_argument("job_id", type=int, help="Job ID within the array.")

    return WorkerArguments(**vars(parser.parse_args()))


class WorkerInformation(BaseModel):
    """
    Describes the informations a worker needs to process its task.
    """

    model_id: str
    fixed_parameter: float  # Usually, degree or flag.
    variable_parameter: float | str  # Usually, period or flag.


def parse_worker_information(function_name: str) -> WorkerInformation:
    """
    Parses the command line arguments for a worker of any specific kind.
    """

    args = parse_worker_args()

    # Gets all the worker's needed informations.
    return WorkerInformation(
        **load_base_model(
            name=str(args.job_id // args.job_array_max_file_size),
            path=worker_information_subpaths[function_name].joinpath(args.job_array_file_base_name),
        )[args.job_id % args.job_array_max_file_size]
    )
