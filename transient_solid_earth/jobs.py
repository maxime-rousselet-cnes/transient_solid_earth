# pylint: disable=consider-using-with
"""
Describes how to launch non-blocking parallel computing either with sbatch or locally.
"""

import multiprocessing
import os
import shutil
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor

from .paths import logs_subpaths


def submit_slurm_jobs(
    n_jobs: int,
    function_name: str,
    job_array_name: str,
    job_array_max_file_size: int,
):
    """
    Submits a job array using sbatch.
    """

    # Ensures the logs directory exists.
    os.makedirs(logs_subpaths[function_name], exist_ok=True)

    # Uses Popen to avoid blocking.
    subprocess.Popen(
        [
            "sbatch",
            f"--array=0-{n_jobs - 1}",
            "_job_array.sh",
            function_name,
            job_array_name,
            str(job_array_max_file_size),
        ]
    )


def run_local_job(
    function_name: str, job_array_name: str, job_array_max_file_size: int, job_id: int
):
    """
    Runs locally a single job in background.
    """

    # Uses Popen to avoid blocking.
    subprocess.Popen(
        [
            "python",
            "worker_" + function_name + ".py",
            job_array_name,
            str(job_array_max_file_size),
            str(job_id),
        ]
    )


def run_local_job_array_in_background(
    n_jobs: int,
    function_name: str,
    job_array_name: str,
    job_array_max_file_size: int,
    max_parallel: int,
):
    """
    Runs locally a job array in background.
    """

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        for i in range(n_jobs):
            executor.submit(
                run_local_job, function_name, job_array_name, job_array_max_file_size, i
            )


def is_slurm_available() -> bool:
    """
    Verifies if the scheduler is available for HPC computing.
    """

    slurm_path = shutil.which("sbatch")

    if not slurm_path:
        return False

    try:
        result = subprocess.run(
            ["scontrol", "ping"], capture_output=True, text=True, timeout=2, check=True
        )
        return "Slurmctld" in result.stdout and "responding" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.CalledProcessError):
        return False


def run_job_array(
    n_jobs: int,
    function_name: str,
    job_array_name: str,
    job_array_max_file_size: int,
) -> None:
    """
    Runs parallel computing without blocking.
    """

    if is_slurm_available():
        # Run SLURM job submission in a background thread
        thread = threading.Thread(
            target=submit_slurm_jobs,
            args=(n_jobs, function_name, job_array_name, job_array_max_file_size),
            daemon=True,
        )
        thread.start()
    else:
        thread = threading.Thread(
            target=run_local_job_array_in_background,
            args=(
                n_jobs,
                function_name,
                job_array_name,
                job_array_max_file_size,
                multiprocessing.cpu_count(),  # Gets the number of available cores.
            ),
            daemon=True,
        )
        thread.start()
