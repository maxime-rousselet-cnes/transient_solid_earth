"""
Describes the loop using adaptatives step for all considered rheologies.
"""

import math
import threading
from copy import deepcopy
from multiprocessing import cpu_count
from typing import Type

import numpy

from .database import load_base_model, save_base_model
from .file_creation_observer import FileCreationObserver
from .jobs import run_job_array
from .model import MODEL
from .parameters import (
    DEFAULT_PARAMETERS,
    DiscretizationParameters,
    ParallelComputingParameters,
    Parameters,
    SolidEarthParameters,
)
from .paths import intermediate_result_subpaths, worker_information_subpaths
from .worker_parser import WorkerInformation


def add_sorted(
    result_dict: dict[str, numpy.ndarray], x: float, values: numpy.ndarray
) -> dict[str, numpy.ndarray]:
    """
    Insert a 'x' and 'values' in result_dict["x"] and result_dict["values"] respectively according
    to the order of result_dict["x"] elements.
    """

    position = 0
    while (position < len(result_dict["x"])) and (result_dict["x"][position] < x):
        position += 1
    return {
        "x": numpy.array(
            object=result_dict["x"][:position].tolist() + [x] + result_dict["x"][position:].tolist()
        ),
        "values": numpy.array(
            object=result_dict["values"][:position].tolist()
            + [values]
            + result_dict["values"][position:].tolist()
        ),
    }


class ProcessCatalog:
    """
    Recap of all processes to be launched, in process, just processed and whose results have been
    stored. Manages the whole adaptative step parallel computing loop.
    """

    # To memorize.
    function_name: str
    parallel_computing_parameters: ParallelComputingParameters
    discretization_parameters: DiscretizationParameters

    # Actual process logs.
    i_job_array: int = 0
    file_creation_observer: FileCreationObserver
    semaphore: threading.Semaphore = threading.Semaphore(value=cpu_count())

    to_process: set[tuple[str, float, float]] = set()
    in_process: dict[tuple[str, float], set[float]] = {}
    just_processed: set[tuple[str, float, float]] = set()
    processed: dict[tuple[str, float], dict[str, numpy.ndarray]] = {}  # "x" tab in log scale.

    def __init__(
        self,
        fixed_parameter_list: list[float],
        rheologies: list[dict],
        function_name: str,
        model: Type[MODEL],
        solid_earth_parameters: SolidEarthParameters,
        parallel_computing_parameters: ParallelComputingParameters,
        discretization_parameters: DiscretizationParameters,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        self.function_name = function_name
        self.parallel_computing_parameters = parallel_computing_parameters
        self.discretization_parameters = discretization_parameters

        self.file_creation_observer = FileCreationObserver(
            base_path=intermediate_result_subpaths[self.function_name]
        )

        # Generates the loop's initial discretization.
        initial_variable_parameter_list = (
            discretization_parameters.exponentiation_base
            ** numpy.linspace(
                start=math.log(
                    discretization_parameters.value_min,
                    discretization_parameters.exponentiation_base,
                ),
                stop=math.log(
                    discretization_parameters.value_max,
                    discretization_parameters.exponentiation_base,
                ),
                num=discretization_parameters.n_0,
            )
        )

        for rheology in rheologies:
            model_id = model(
                solid_earth_parameters=deepcopy(solid_earth_parameters), rheology=rheology
            ).model_id
            for fixed_parameter in fixed_parameter_list:
                self.processed[(model_id, fixed_parameter)] = {
                    "x": numpy.array(object=[]),
                    "values": numpy.array(object=[]),
                }
                for x in initial_variable_parameter_list:
                    self.to_process.add((model_id, fixed_parameter, x))

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
            if (model_id, fixed_parameter) in self.in_process:
                self.in_process[(model_id, fixed_parameter)].add(variable_parameter)
            else:
                self.in_process[(model_id, fixed_parameter)] = set([variable_parameter])

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

    def get_results(self) -> None:
        """
        Loads the already computed values and updates the 'in_process' and 'processed' attributes.
        Updates 'just_processed' and 'processed'.
        """

        for path in self.file_creation_observer.get_created_file_paths():

            variable_parameter = float(path.name)
            fixed_parameter = float(path.parent.name)
            model_id = path.parent.parent.name

            if (model_id, fixed_parameter) in self.in_process:

                if variable_parameter in self.in_process[(model_id, fixed_parameter)]:

                    # Updates 'in_process'.
                    # print(model_id, fixed_parameter)
                    # print(self.in_process[(model_id, fixed_parameter)])
                    self.in_process[(model_id, fixed_parameter)].remove(variable_parameter)
                    if len(self.in_process[(model_id, fixed_parameter)]) == 0:
                        del self.in_process[(model_id, fixed_parameter)]

                    # Updates 'just_processed'.
                    self.just_processed.add((model_id, fixed_parameter, variable_parameter))

                    # Updates 'processed'.
                    self.processed[(model_id, fixed_parameter)] = add_sorted(
                        result_dict=self.processed[model_id, fixed_parameter],
                        x=math.log(
                            variable_parameter, self.discretization_parameters.exponentiation_base
                        ),
                        values=numpy.array(object=load_base_model(name="real", path=path))
                        + 1.0j * numpy.array(object=load_base_model(name="imag", path=path)),
                    )

    def refine_discretization(self) -> None:
        """
        Refines the discretization on the variable parameter for a rheology and a fixed_parameter.
        Stops when the result is approximated everywhere by its linear interpolation.
        """

        to_postprocess: set[tuple[str, float]] = set()
        while self.just_processed:
            model_id, fixed_parameter, _ = self.just_processed.pop()
            if (model_id, fixed_parameter) not in self.in_process:
                to_postprocess.add((model_id, fixed_parameter))

        # Don't test for the stop criterion if processes are still running for the same model.
        for model_id, fixed_parameter in to_postprocess:

            # Verifies the stop criterion on the whole sequence.
            f = self.processed[(model_id, fixed_parameter)]["values"]  # shape (T, *S)
            x = self.processed[(model_id, fixed_parameter)]["x"]  # shape (T,)

            # Reshapes time arrays to broadcast with f.
            x_0 = x[:-2].reshape(-1, *([1] * (f.ndim - 1)))  # shape (T-2, 1, ..., 1)
            x_1 = x[1:-1].reshape(-1, *([1] * (f.ndim - 1)))
            x_2 = x[2:].reshape(-1, *([1] * (f.ndim - 1)))
            x_span = x_2 - x_0

            # Linear interpolation estimate at midpoints.
            mask = numpy.any(
                numpy.abs(((x_1 - x_0) * f[2:] + (x_2 - x_1) * f[:-2]) / x_span - f[1:-1])
                > self.discretization_parameters.maximum_tolerance
                * (numpy.abs(f[2:] - f[:-2]) + numpy.abs(f[-1] - f[0])),
                axis=tuple(range(1, f.ndim)),
            )

            # Inserts midpoints of x where error is large.
            new_x_values = numpy.unique(
                numpy.concatenate(
                    [(x[:-2][mask] + x[1:-1][mask]) / 2.0, (x[1:-1][mask] + x[2:][mask]) / 2.0]
                )
            )

            if len(new_x_values) != 0:

                # Updates 'to_process'.
                for x in new_x_values:
                    self.to_process.add(
                        (
                            model_id,
                            fixed_parameter,
                            self.discretization_parameters.exponentiation_base**x,
                        )
                    )

            else:

                # Saves results for the whole model.
                variable_parameter_tab = (
                    self.discretization_parameters.exponentiation_base
                    ** self.processed[model_id, fixed_parameter]["x"]
                )
                path = (
                    intermediate_result_subpaths[self.function_name]
                    .joinpath(model_id)
                    .joinpath(str(fixed_parameter))
                )
                save_base_model(
                    obj={
                        "variable_parameter": variable_parameter_tab,
                        "values": self.processed[model_id, fixed_parameter]["values"].real,
                    },
                    name="real",
                    path=path,
                )
                save_base_model(
                    obj={
                        "variable_parameter": variable_parameter_tab,
                        "values": self.processed[model_id, fixed_parameter]["values"].imag,
                    },
                    name="imag",
                    path=path,
                )


def adaptative_step_parallel_computing_loop(
    rheologies: list[dict],
    model: Type[MODEL],
    function_name: str,
    fixed_parameter_list: list[float],
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> None:
    """
    For every rheologies, uses an adaptative step.
    """

    # Initializes data structures.
    process_catalog = ProcessCatalog(
        fixed_parameter_list=fixed_parameter_list,
        rheologies=rheologies,
        function_name=function_name,
        model=model,
        solid_earth_parameters=parameters.solid_earth,
        parallel_computing_parameters=parameters.parallel_computing,
        discretization_parameters=parameters.discretization[function_name],
    )

    try:

        # Loops until the stop criterion is verified for every rheologies and fixed_parameter.
        while process_catalog.to_process or process_catalog.in_process:

            if process_catalog.to_process:

                # Launches jobs if needed.
                process_catalog.schedule_jobs()

            if process_catalog.file_creation_observer.file_has_been_created():

                # Gets results if they are finished.
                process_catalog.get_results()

                # Tests the stop cirterion on newly available results.
                process_catalog.refine_discretization()

    finally:

        process_catalog.file_creation_observer.stop()
