# pylint: disable=consider-using-with
"""
Needed to monitor parallel computing.
"""

import os
import shutil
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from itertools import islice
from multiprocessing import cpu_count, get_context
from typing import Iterable, Optional

from .database import save_base_model
from .parallel_processing_functions import functions
from .parameters import ParallelComputingParameters
from .paths import intermediate_result_subpaths, logs_subpaths, worker_information_subpaths
from .worker_parser import WorkerInformation


def chunked(iterable: Iterable, size: int) -> Iterable:
    """
    Splits an iterable into chunks of a given maximum size.
    """

    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


def submit_local_job(function_name: str, worker_information: WorkerInformation) -> None:
    """
    To avoid resource-allocation explosion.
    """

    functions[function_name](worker_information=worker_information)


def submit_local_jobs(
    function_name: str,
    max_concurrent_processes_factor: int,
    job_array_worker_informations: list[list[WorkerInformation]],
) -> None:
    """
    Local parallel processing.
    """
    flat_worker_infos = sum(job_array_worker_informations, [])
    submit_fn = partial(submit_local_job, function_name)  # Partially applies the function name.

    with ProcessPoolExecutor(
        max_workers=cpu_count()
        // (
            1  # Does not limit the number of prcesses if the loop is straightforward.
            if ("interpolate" in function_name) or ("asymptotic" in function_name)
            else max_concurrent_processes_factor
        ),
        mp_context=get_context("fork"),
    ) as executor:
        list(executor.map(submit_fn, flat_worker_infos))


def is_slurm_available() -> bool:
    """
    Verifies if the scheduler is available for HPC computing.
    """

    slurm_path = shutil.which("sbatch")

    if not slurm_path:
        return False

    try:
        result = subprocess.run(
            ["scontrol", "ping"], capture_output=True, text=True, timeout=2.0, check=True
        )
        return "Slurmctld" in result.stdout and "responding" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.CalledProcessError):
        return False


def submit_slurm_jobs(
    n_jobs: int,
    function_name: str,
    job_array_name: str,
    job_array_max_file_size: int,
) -> None:
    """
    Submits a job array using sbatch.
    """

    os.makedirs(logs_subpaths[function_name], exist_ok=True)

    subprocess.Popen(
        [
            "sbatch",
            f"--array=0-{n_jobs - 1}",
            "job_array.sh",
            function_name,
            job_array_name,
            str(job_array_max_file_size),
        ]
    )


class ProcessCatalog:
    """
    Recap of all processes to be launched and in process for a parallel computing loop.
    """

    to_process: set[tuple[str, float, float]] = set()
    in_process: dict[tuple[str, float], set[float]] = {}
    parallel_computing_parameters: ParallelComputingParameters
    slurm: bool
    thread_semaphore: threading.Semaphore
    i_job_array: int = 0
    i_job_array_lock: threading.Lock
    function_name: str
    threads: list[threading.Thread] = []

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
        self.slurm = is_slurm_available()
        self.thread_semaphore = threading.Semaphore(
            value=cpu_count() // parallel_computing_parameters.max_concurrent_threads_factor
        )
        self.i_job_array_lock = threading.Lock()

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

    def run_job_array(
        self,
        n_jobs: int,
        job_array_worker_informations: list[list[WorkerInformation]],
    ) -> threading.Thread:
        """
        Runs parallel computing without blocking.
        """

        def thread_target():

            with self.i_job_array_lock:
                i_job_array_snapshot = self.i_job_array

            with self.thread_semaphore:  # Limits concurent threads.
                if self.slurm:

                    # Writes worker informations in files so that separate jobs can read them.
                    for i_worker_informations_file, worker_informations in enumerate(
                        job_array_worker_informations[:-1]
                    ):
                        save_base_model(
                            obj=worker_informations,
                            name=str(i_worker_informations_file),
                            path=worker_information_subpaths[self.function_name].joinpath(
                                str(i_job_array_snapshot)
                            ),
                        )

                    submit_slurm_jobs(
                        n_jobs=n_jobs,
                        function_name=self.function_name,
                        job_array_name=i_job_array_snapshot,
                        job_array_max_file_size=(
                            self.parallel_computing_parameters.job_array_max_file_size
                        ),
                    )

                else:
                    submit_local_jobs(
                        function_name=self.function_name,
                        max_concurrent_processes_factor=(
                            self.parallel_computing_parameters.max_concurrent_processes_factor
                        ),
                        job_array_worker_informations=job_array_worker_informations,
                    )

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
        return thread

    def schedule_jobs(self) -> None:
        """
        Schedules jobs for a job array. Empties 'to_process' and updates 'in_process'.
        """

        jobs = list(self.to_process)
        self.to_process.clear()
        n_jobs = len(jobs)

        worker_info_list = [
            WorkerInformation(
                model_id=model_id,
                fixed_parameter=fixed_parameter,
                variable_parameter=variable_parameter,
            )
            for model_id, fixed_parameter, variable_parameter in jobs
        ]

        for model_id, fixed_parameter, variable_parameter in jobs:
            self.update_in_process(
                model_id=model_id,
                fixed_parameter=fixed_parameter,
                variable_parameter=variable_parameter,
            )

        job_array_worker_informations = list(
            chunked(
                worker_info_list,
                self.parallel_computing_parameters.job_array_max_file_size,
            )
        )

        self.threads.append(
            self.run_job_array(
                n_jobs=n_jobs,
                job_array_worker_informations=job_array_worker_informations,
            )
        )

        with self.i_job_array_lock:
            self.i_job_array += 1

    def wait(self, timeout: Optional[float] = None) -> None:
        """
        Waits for all jobs to finish, or timeout
        """

        if not timeout:
            timeout = self.parallel_computing_parameters.timeout
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=timeout)

    def wait_for_jobs(
        self, subpath_name: Optional[str] = None, timeout: Optional[float] = None
    ) -> None:
        """
        Waits for all jobs to finish.
        """

        while self.in_process:
            for (model_id, _), __ in deepcopy(self.in_process).items():
                path = intermediate_result_subpaths[self.function_name].joinpath(model_id)
                if subpath_name:
                    path = path.joinpath(subpath_name)
                if path.exists():
                    self.in_process.pop((model_id, _))
                    self.wait(timeout=timeout)
