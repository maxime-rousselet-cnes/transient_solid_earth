"""
Describes the loop using adaptatives step for all considered rheologies.
"""

import math
from copy import deepcopy
from itertools import product
from typing import Type

import numpy
from pydantic import BaseModel

from .database import load_base_model, save_base_model
from .jobs import run_job_array
from .model import MODEL, Model
from .parameters import DEFAULT_PARAMETERS, Parameters
from .paths import intermediate_result_subpaths, worker_information_subpaths


class WorkerInformation(BaseModel):
    """
    Describes the informations a worker needs to process its task.
    """

    model_id: str
    fixed_parameter: float
    variable_parameter: float


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


def adaptative_step_parallel_computing_loop(
    rheologies: list[dict],
    model: Type[MODEL],
    function_name: str,
    fixed_parameter_list: list[float] = [0.0],
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> None:
    """
    For every rheologies, uses an adaptative step.
    """

    # Generates rheologies.
    models: dict[tuple[tuple, float], Model] = {}
    for rheology, fixed_parameter in product(rheologies, fixed_parameter_list):
        models[(tuple(rheology.values()), fixed_parameter)] = model(
            solid_earth_parameters=parameters.solid_earth, rheology=rheology
        )

    # Initializes data structures.
    i_job_array = 0
    in_process: dict[tuple[tuple, float], list[float]] = {}
    just_processed: set[tuple[tuple, float]] = set()
    processed: dict[
        tuple,  # Rheology as a tuple of parameters (model part names).
        dict[float, dict[str, numpy.ndarray]],
    ] = {
        tuple(rheology.values()): {
            fixed_parameter: {"x": [], "values": []} for fixed_parameter in fixed_parameter_list
        }
        for rheology in rheologies
    }

    # Generates the loop's input.
    initial_variable_parameter_list = numpy.linspace(
        start=math.log(
            parameters.discretization.x_min,
            parameters.discretization.exponentiation_base,
        ),
        stop=math.log(
            parameters.discretization.x_max,
            parameters.discretization.exponentiation_base,
        ),
        num=parameters.discretization.n_0,
    )
    to_process: set[tuple[tuple[tuple, float], float]] = set()
    for rheology in rheologies:
        for fixed_parameter in fixed_parameter_list:
            for x in initial_variable_parameter_list:
                to_process.add(((tuple(rheology.values()), fixed_parameter), x))

    # Loops until the stop criterion is verified for every rheologies and fixed_parameter.
    while processed:

        # Launches jobs if needed.
        if to_process:

            i_worker_informations_file, i_job, n_jobs = 0, 0, len(to_process)
            worker_informations: list[WorkerInformation] = []

            # Create files for workers informations.
            while to_process:

                (rheology, fixed_parameter), x = to_process.pop()

                worker_informations += [
                    WorkerInformation(
                        model_id=models[(rheology, fixed_parameter)].model_id,
                        fixed_parameter=fixed_parameter,
                        variable_parameter=parameters.discretization.exponentiation_base**x,
                    )
                ]
                i_job += 1

                # Create a file for 'job_array_max_file_size' worker informations
                if (i_job % parameters.parallel_computing.job_array_max_file_size == 0) or (
                    i_job == n_jobs
                ):

                    save_base_model(
                        obj=worker_informations,
                        name=str(i_worker_informations_file),
                        path=worker_information_subpaths[function_name].joinpath(str(i_job_array)),
                    )
                    worker_informations = []
                    i_worker_informations_file += 1

                # Updates 'in_process'.
                if (rheology, fixed_parameter) in in_process:
                    in_process[(rheology, fixed_parameter)] += [x]
                else:
                    in_process[(rheology, fixed_parameter)] = [x]

            # Submits the job array.
            run_job_array(
                n_jobs=n_jobs,
                function_name=function_name,
                job_array_name=str(i_job_array),
                job_array_max_file_size=parameters.parallel_computing.job_array_max_file_size,
            )

            # Updates the job array ID.
            i_job_array += 1

        # Gets results.
        for (rheology, fixed_parameter), x_values_in_process in deepcopy(in_process).items():

            model_results_path = intermediate_result_subpaths[function_name].joinpath(
                models[(rheology, fixed_parameter)].model_id
            )

            i_t_tab = 0

            for x in x_values_in_process:

                name = str(parameters.discretization.exponentiation_base**x)
                file_path = model_results_path.joinpath(name + ".json")

                # Verifies is the process ended.
                if file_path.exists():

                    # Updates 'in_process'.
                    del in_process[(rheology, fixed_parameter)][i_t_tab]
                    if len(in_process[(rheology, fixed_parameter)]) == 0:
                        del in_process[(rheology, fixed_parameter)]

                    # Updates 'just_processed'.
                    just_processed.add((rheology, fixed_parameter))

                    # Updates 'processed'.
                    processed[rheology][fixed_parameter] = add_sorted(
                        result_dict=processed[rheology][fixed_parameter],
                        x=x,
                        values=load_base_model(name=name, path=model_results_path),
                    )

                else:

                    i_t_tab += 1

        # Tests the stop cirterion on newly available results.
        while just_processed:

            rheology, fixed_parameter = just_processed.pop()

            # Don't test for the stop criterion if processes are still running for the same model.
            if (rheology, fixed_parameter) not in in_process:

                # Verifies the stop criterion on the whole sequence.
                f = processed[rheology][fixed_parameter]["values"]  # shape (T, *S)
                x = processed[rheology][fixed_parameter]["x"]  # shape (T,)

                # Reshapes time arrays to broadcast with f.
                x_0 = x[:-2].reshape(-1, *([1] * (f.ndim - 1)))  # shape (T-2, 1, ..., 1)
                x_1 = x[1:-1].reshape(-1, *([1] * (f.ndim - 1)))
                x_2 = x[2:].reshape(-1, *([1] * (f.ndim - 1)))
                x_span = x_2 - x_0

                # Linear interpolation estimate at midpoints.
                mask = numpy.any(
                    numpy.abs(((x_1 - x_0) * f[2:] + (x_2 - x_1) * f[:-2]) / x_span - f[1:-1])
                    > parameters.discretization.maximum_tolerance
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
                        to_process.add(((rheology, fixed_parameter), x))

                else:

                    # Saves results for the whole model.
                    save_base_model(
                        obj={"rheology": rheology, "fixed_parameter": fixed_parameter}
                        | {
                            "x": parameters.discretization.exponentiation_base
                            ** processed[rheology][fixed_parameter]["x"],
                            "values": processed[rheology][fixed_parameter]["values"],
                        },
                        name=models[(rheology, fixed_parameter)].model_id,
                        path=intermediate_result_subpaths[function_name],
                    )

                    # Updates 'processed'.
                    del processed[rheology][fixed_parameter]
                    if len(processed[rheology]) == 0:
                        del processed[rheology]
