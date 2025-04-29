"""
Needed to monitor parallel computing.
"""

import threading
from multiprocessing import cpu_count

from .database import save_base_model
from .jobs import run_job_array
from .parameters import ParallelComputingParameters
from .paths import worker_information_subpaths
from .worker_parser import WorkerInformation


class ProcessCatalog:
    """
    Recap of all processes to be launched and in process for a parallel computing loop.
    """

    to_process: set[tuple[str, float, float]] = set()
    in_process: dict[tuple[str, float], set[float]] = {}
    parallel_computing_parameters: ParallelComputingParameters
    semaphore: threading.Semaphore
    i_job_array: int = 0
    function_name: str

    def __init__(
        self,
        function_name: str,
        parallel_computing_parameters: ParallelComputingParameters,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        self.function_name = function_name
        self.parallel_computing_parameters = parallel_computing_parameters
        self.semaphore = threading.Semaphore(
            value=parallel_computing_parameters.cpu_buffer_factor * cpu_count()
        )

    def update_in_process(
        self, model_id: str, fixed_parameter: float, variable_parameter: float
    ) -> None:
        """
        Updates the 'in_process' attribute.
        """

        if (model_id, fixed_parameter) in self.in_process:
            self.in_process[(model_id, fixed_parameter)].add(variable_parameter)
        else:
            self.in_process[(model_id, fixed_parameter)] = set([variable_parameter])

    def schedule_jobs(self) -> None:
        """
        Schedules jobs for a job array. Empties 'to_process' and updates 'in_process'.
        """

        i_worker_informations_file, n_jobs = 0, len(self.to_process)
        worker_informations: list[WorkerInformation] = []

        # Create files for workers informations.
        for i_job in range(1, n_jobs + 1):

            model_id, fixed_parameter, variable_parameter = self.to_process.pop()

            worker_informations += [
                WorkerInformation(
                    model_id=model_id,
                    fixed_parameter=fixed_parameter,
                    variable_parameter=variable_parameter,
                )
            ]

            # Create a file for 'job_array_max_file_size' worker informations
            if (i_job % self.parallel_computing_parameters.job_array_max_file_size == 0) or (
                i_job == n_jobs
            ):

                save_base_model(
                    obj=worker_informations,
                    name=str(i_worker_informations_file),
                    path=worker_information_subpaths[self.function_name].joinpath(
                        str(self.i_job_array)
                    ),
                )
                worker_informations = []
                i_worker_informations_file += 1

            # Updates 'in_process'.
            self.update_in_process(
                model_id=model_id,
                fixed_parameter=fixed_parameter,
                variable_parameter=variable_parameter,
            )

        # Submits the job array.
        run_job_array(
            n_jobs=n_jobs,
            function_name=self.function_name,
            job_array_name=str(self.i_job_array),
            job_array_max_file_size=self.parallel_computing_parameters.job_array_max_file_size,
            semaphore=self.semaphore,
        )

        # Updates the job array ID.
        self.i_job_array += 1
