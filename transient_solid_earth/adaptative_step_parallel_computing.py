"""
Describes the loop using adaptatives step for all considered rheologies.
"""

import math
import multiprocessing
import threading
from copy import deepcopy
from itertools import product
from typing import Optional, Type

import numpy

from .database import load_base_model, save_base_model
from .jobs import run_job_array
from .model import MODEL
from .parameters import DEFAULT_PARAMETERS, Parameters, SolidEarthParameters
from .paths import intermediate_result_subpaths, worker_information_subpaths
from .worker_parser import WorkerInformation


def add_sorted(result_dict: dict[str, list], x: float, values: list[float]) -> dict[str, list]:
    """
    Insert a 'x' and 'values' in result_dict["x"] and result_dict["values"] respectively according
    to the order of result_dict["x"] elements.
    """

    position = 0
    while (position < len(result_dict["x"])) and (result_dict["x"][position] < x):
        position += 1
    return {
        "x": numpy.array(
            object=list(result_dict["x"][:position]) + [x] + list(result_dict["x"][position:])
        ),
        "values": numpy.array(
            object=list(result_dict["values"][:position])
            + [values]
            + list(result_dict["values"][position:])
        ),
    }


class ProcessCatalog:
    """
    Recap of all processes to be launched, in process, just processed and whose results have been
    stored. Manages the whole adaptative step parallel computing loop.
    """

    # Actual process logs.
    to_process: set[tuple[tuple[tuple, float], float]]
    in_process: dict[tuple[tuple, float], list[float]]
    just_processed: set[tuple[tuple, float]]
    processed: dict[
        tuple,  # Rheology as a tuple of parameters (model part names).
        dict[float, dict[str, numpy.ndarray]],
    ]

    # To memorize.
    function_name: str
    rheologies: list[dict]
    model_ids: dict[tuple[tuple, float], str]

    def __init__(
        self,
        fixed_parameter_list: list[float],
        rheologies: list[dict],
        initial_variable_parameter_list: numpy.ndarray[float] | list[float],
        function_name: str,
    ) -> None:

        self.to_process: set[tuple[tuple[tuple, float], float]] = set()
        for rheology in rheologies:
            for fixed_parameter in fixed_parameter_list:
                for x in initial_variable_parameter_list:
                    self.to_process.add(((tuple(rheology.values()), fixed_parameter), x))
        self.in_process: dict[tuple[tuple, float], list[float]] = {}
        self.just_processed: set[tuple[tuple, float]] = set()
        self.processed: dict[
            tuple,  # Rheology as a tuple of parameters (model part names).
            dict[float, dict[str, numpy.ndarray]],
        ] = {
            tuple(rheology.values()): {
                fixed_parameter: {"x": [], "values": []} for fixed_parameter in fixed_parameter_list
            }
            for rheology in rheologies
        }
        self.function_name = function_name
        self.rheologies = rheologies
        self.model_ids: dict[tuple[tuple, float], str] = {}

    def generate_rheologies(
        self,
        model: Type[MODEL],
        fixed_parameter_list: list[float],
        solid_earth_parameters: SolidEarthParameters,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        for rheology, fixed_parameter in product(self.rheologies, fixed_parameter_list):
            self.model_ids[(tuple(rheology.values()), fixed_parameter)] = model(
                solid_earth_parameters=deepcopy(solid_earth_parameters), rheology=rheology
            ).model_id

    def schedule_jobs(
        self,
        i_job_array: int,
        job_array_max_file_size: int,
        exponentiation_base: float,
        global_semaphore: threading.Semaphore,
    ) -> int:
        """
        Schedules jobs for a job array. Empties 'to_process' and updates 'in_process'.
        """

        i_worker_informations_file, i_job, n_jobs = 0, 0, len(self.to_process)
        worker_informations: list[WorkerInformation] = []

        # Create files for workers informations.
        while self.to_process:

            (rheology, fixed_parameter), x = self.to_process.pop()

            worker_informations += [
                WorkerInformation(
                    model_id=self.model_ids[(rheology, fixed_parameter)],
                    fixed_parameter=fixed_parameter,
                    variable_parameter=exponentiation_base**x,
                )
            ]
            i_job += 1

            # Create a file for 'job_array_max_file_size' worker informations
            if (i_job % job_array_max_file_size == 0) or (i_job == n_jobs):

                save_base_model(
                    obj=worker_informations,
                    name=str(i_worker_informations_file),
                    path=worker_information_subpaths[self.function_name].joinpath(str(i_job_array)),
                )
                worker_informations = []
                i_worker_informations_file += 1

            # Updates 'in_process'.
            if (rheology, fixed_parameter) in self.in_process:
                self.in_process[(rheology, fixed_parameter)] += [x]
            else:
                self.in_process[(rheology, fixed_parameter)] = [x]

        # Submits the job array.
        run_job_array(
            n_jobs=n_jobs,
            function_name=self.function_name,
            job_array_name=str(i_job_array),
            job_array_max_file_size=job_array_max_file_size,
            semaphore=global_semaphore,
        )

        # Updates the job array ID.
        i_job_array += 1

        return i_job_array

    def get_results(self, exponentiation_base: float) -> None:
        """
        Loads the already computed values and updates the 'in_process' and 'processed' attributes.
        Updates 'just_processed' and 'processed'.
        """

        for (rheology, fixed_parameter), x_values_in_process in deepcopy(self.in_process).items():

            model_results_path = intermediate_result_subpaths[self.function_name].joinpath(
                self.model_ids[(rheology, fixed_parameter)]
            )

            i_t_tab = 0

            for x in x_values_in_process:

                name = str(exponentiation_base**x)
                file_path = model_results_path.joinpath(name + ".json")

                # Verifies is the process ended.
                if file_path.exists():

                    # Updates 'in_process'.
                    del self.in_process[(rheology, fixed_parameter)][i_t_tab]
                    if len(self.in_process[(rheology, fixed_parameter)]) == 0:
                        del self.in_process[(rheology, fixed_parameter)]

                    # Updates 'just_processed'.
                    self.just_processed.add((rheology, fixed_parameter))

                    # Updates 'processed'.
                    self.processed[rheology][fixed_parameter] = add_sorted(
                        result_dict=self.processed[rheology][fixed_parameter],
                        x=x,
                        values=load_base_model(name=name, path=model_results_path),
                    )

                else:

                    i_t_tab += 1

    def refine_discretization(self, maximum_tolerance: float, exponentiation_base: float) -> None:
        """
        Refines the discretization on the variable parameter for a rheology and a fixed_parameter.
        Stops when the result is approximated everywhere by its linear interpolation.
        """

        rheology, fixed_parameter = self.just_processed.pop()

        # Don't test for the stop criterion if processes are still running for the same model.
        if (rheology, fixed_parameter) not in self.in_process:

            # Verifies the stop criterion on the whole sequence.
            f = self.processed[rheology][fixed_parameter]["values"]  # shape (T, *S)
            x = self.processed[rheology][fixed_parameter]["x"]  # shape (T,)

            # Reshapes time arrays to broadcast with f.
            x_0 = x[:-2].reshape(-1, *([1] * (f.ndim - 1)))  # shape (T-2, 1, ..., 1)
            x_1 = x[1:-1].reshape(-1, *([1] * (f.ndim - 1)))
            x_2 = x[2:].reshape(-1, *([1] * (f.ndim - 1)))
            x_span = x_2 - x_0

            # Linear interpolation estimate at midpoints.
            mask = numpy.any(
                numpy.abs(((x_1 - x_0) * f[2:] + (x_2 - x_1) * f[:-2]) / x_span - f[1:-1])
                > maximum_tolerance * (numpy.abs(f[2:] - f[:-2]) + numpy.abs(f[-1] - f[0])),
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
                    self.to_process.add(((rheology, fixed_parameter), x))

            else:

                # Saves results for the whole model.
                save_base_model(
                    obj={"rheology": rheology, "fixed_parameter": fixed_parameter}
                    | {
                        "x": exponentiation_base ** self.processed[rheology][fixed_parameter]["x"],
                        "values": self.processed[rheology][fixed_parameter]["values"],
                    },
                    name=self.model_ids[(rheology, fixed_parameter)],
                    path=intermediate_result_subpaths[self.function_name],
                )

                # Updates 'processed'.
                del self.processed[rheology][fixed_parameter]
                if len(self.processed[rheology]) == 0:
                    del self.processed[rheology]


def adaptative_step_parallel_computing_loop(
    rheologies: list[dict],
    model: Type[MODEL],
    function_name: str,
    fixed_parameter_list: Optional[list[float]] = None,
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> None:
    """
    For every rheologies, uses an adaptative step.
    """

    # Manages defaults.
    if not fixed_parameter_list:
        fixed_parameter_list = [0.0]

    # Generates the loop's initial discretization.
    initial_variable_parameter_list = numpy.linspace(
        start=math.log(
            parameters.discretization[function_name].x_min,
            parameters.discretization[function_name].exponentiation_base,
        ),
        stop=math.log(
            parameters.discretization[function_name].x_max,
            parameters.discretization[function_name].exponentiation_base,
        ),
        num=parameters.discretization[function_name].n_0,
    )

    # Initializes data structures.
    i_job_array = 0
    global_semaphore = threading.Semaphore(value=2.0 * multiprocessing.cpu_count())
    process_catalog = ProcessCatalog(
        fixed_parameter_list=fixed_parameter_list,
        rheologies=rheologies,
        initial_variable_parameter_list=initial_variable_parameter_list,
        function_name=function_name,
    )

    # Generates rheology names and save numerical models.
    process_catalog.generate_rheologies(
        model=model,
        fixed_parameter_list=fixed_parameter_list,
        solid_earth_parameters=parameters.solid_earth,
    )

    # Loops until the stop criterion is verified for every rheologies and fixed_parameter.
    while process_catalog.processed:

        if process_catalog.to_process:

            # Launches jobs if needed.
            i_job_array = process_catalog.schedule_jobs(
                i_job_array=i_job_array,
                job_array_max_file_size=parameters.parallel_computing.job_array_max_file_size,
                exponentiation_base=parameters.discretization[function_name].exponentiation_base,
                global_semaphore=global_semaphore,
            )

        # Gets results if they are finished.
        process_catalog.get_results(
            exponentiation_base=parameters.discretization[function_name].exponentiation_base
        )

        while process_catalog.just_processed:

            # Tests the stop cirterion on newly available results.
            process_catalog.refine_discretization(
                maximum_tolerance=parameters.discretization[function_name].maximum_tolerance,
                exponentiation_base=parameters.discretization[function_name].exponentiation_base,
            )
