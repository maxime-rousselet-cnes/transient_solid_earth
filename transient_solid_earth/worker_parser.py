"""
For all worker types to have same signature.
"""

import argparse
import dataclasses


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
