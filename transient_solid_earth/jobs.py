# pylint: disable=consider-using-with
"""
Describes how to launch non-blocking parallel computing either with sbatch or locally.
"""

import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor  # Use. threads, not processes.

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

    os.makedirs(logs_subpaths[function_name], exist_ok=True)

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


def run_local_job_with_semaphore(
    semaphore: threading.Semaphore,
    function_name: str,
    job_array_name: str,
    job_array_max_file_size: int,
    job_id: int,
):
    """
    Runs a single job while respecting a global semaphore limit.
    """
    with semaphore:
        process = subprocess.Popen(
            [
                "python",
                "worker_" + function_name + ".py",
                job_array_name,
                str(job_array_max_file_size),
                str(job_id),
            ]
        )
        process.wait()  # Waits until the subprocess completes before releasing the slot.


def run_local_job_array_in_background(
    n_jobs: int,
    function_name: str,
    job_array_name: str,
    job_array_max_file_size: int,
    semaphore: threading.Semaphore,
):
    """
    Runs locally a job array in background with a concurrency limit.
    """

    with ThreadPoolExecutor() as executor:
        for i in range(n_jobs):
            executor.submit(
                run_local_job_with_semaphore,
                semaphore,
                function_name,
                job_array_name,
                job_array_max_file_size,
                i,
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
            ["scontrol", "ping"], capture_output=True, text=True, timeout=0.1, check=True
        )
        return "Slurmctld" in result.stdout and "responding" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.CalledProcessError):
        return False


def run_job_array(
    n_jobs: int,
    function_name: str,
    job_array_name: str,
    job_array_max_file_size: int,
    semaphore: threading.Semaphore,
) -> threading.Thread:
    """
    Runs parallel computing without blocking.
    """

    if is_slurm_available():
        thread = threading.Thread(
            target=submit_slurm_jobs,
            args=(n_jobs, function_name, job_array_name, job_array_max_file_size),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=run_local_job_array_in_background,
            args=(
                n_jobs,
                function_name,
                job_array_name,
                job_array_max_file_size,
                semaphore,
            ),
            daemon=True,
        )
    thread.start()
    return thread
