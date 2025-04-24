"""
Describes the loop using frequency adaptatives step for all considered rheologies and degrees.
"""

import shutil
from copy import deepcopy
from itertools import product

import numpy
from pydantic import BaseModel

from .database import load_base_model, save_base_model
from .jobs import run_job_array
from .test_model import Model, test_models_path


class WorkerInformation(BaseModel):
    """
    Describes the informations a worker needs to process its task.
    """

    name: str
    x_0: float
    t: float


def add_sorted(result_dict: dict[str, list], t: float, values: list[float]) -> dict[str, list]:
    """
    Insert a 't' and 'values' in result_dict["t"] and result_dict["values"] respectively according
    to the order of result_dict["t"] elements.
    """

    position = 0
    while (position < len(result_dict["t"])) and (result_dict["t"][position] < t):
        position += 1
    return {
        "t": result_dict["t"][:position] + [t] + result_dict["t"][position:],
        "values": result_dict["values"][:position] + [values] + result_dict["values"][position:],
    }


def love_numbers_computing_loop(
    job_array_max_file_size: int = 10,
    x_0_list: list[float] = [0.25, 0.5],
    alpha_list: list[float] = [-1, 1],
    beta_list: list[float] = [-1, 1],
    gamma_list: list[float] = [-1, 1],
    initial_t_min: float = -5,
    initial_t_max: float = 2,
    initial_n_t: float = 10,
    normalized_curvature_threshold: float = 0.5,
):
    """
    For every rheologies and degrees, uses an adaptative step on frequency.
    """

    # Clears.
    if test_models_path.exists():
        shutil.rmtree(test_models_path)

    # Generates rheologies.
    models: dict[tuple[tuple[float, float, float], float], Model] = {}
    rheologies = list(product(alpha_list, beta_list, gamma_list))
    for x_0, (alpha, beta, gamma) in product(x_0_list, rheologies):
        model = Model(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            x_0=x_0,
            name="_".join([str(alpha), str(beta), str(gamma), str(x_0)]),
        )
        save_base_model(obj=model, name=model.name, path=test_models_path.joinpath("models"))
        models[((alpha, beta, gamma), x_0)] = model

    # Initializes data structures.
    i_job_array = 0
    in_process: dict[tuple[tuple[float, float, float], float], list[float]] = {}
    just_processed: set[tuple[tuple[float, float, float], float]] = set()
    processed: dict[
        tuple[float, float, float],
        dict[float, dict[str, list]],  # {"t": list[float], "values": list[list[float]]}
    ] = {rheology: {x_0: {"t": [], "values": []} for x_0 in x_0_list} for rheology in rheologies}

    # Generates the loop's input.
    initial_t_list = numpy.linspace(start=initial_t_min, stop=initial_t_max, num=initial_n_t)
    to_process: set[tuple[tuple[tuple[float, float, float], float], float]] = set()
    for rheology in rheologies:
        for x_0 in x_0_list:
            for t in initial_t_list:
                to_process.add(((rheology, x_0), t))

    # Loops until the stop criterion is verified for every rheologies and x_0.
    while processed:

        # Launches jobs if needed.
        if to_process:

            i_worker_informations_file, i_job, n_jobs = 0, 0, len(to_process)
            worker_informations: list[WorkerInformation] = []

            # Create files for workers informations.
            while to_process:

                (rheology, x_0), t = to_process.pop()

                worker_informations += [
                    WorkerInformation(name=models[(rheology, x_0)].name, x_0=x_0, t=t)
                ]
                i_job += 1

                # Create a file for 'job_array_max_file_size' worker informations
                if (i_job % job_array_max_file_size == 0) or (i_job == n_jobs):

                    save_base_model(
                        obj=worker_informations,
                        name=str(i_worker_informations_file),
                        path=test_models_path.joinpath("worker_informations").joinpath(
                            str(i_job_array)
                        ),
                    )
                    worker_informations = []
                    i_worker_informations_file += 1

                # Updates 'in_process'.
                if (rheology, x_0) in in_process:
                    in_process[(rheology, x_0)] += [t]
                else:
                    in_process[(rheology, x_0)] = [t]

            # Submits the job array.
            run_job_array(
                n_jobs=n_jobs,
                function_name="love_numbers",
                job_array_name=str(i_job_array),
                job_array_max_file_size=job_array_max_file_size,
            )

            # Updates the job array ID.
            i_job_array += 1

        # Gets results.
        for (rheology, x_0), t_values_in_process in deepcopy(in_process).items():

            model_results_path = test_models_path.joinpath("intermediate_results").joinpath(
                models[(rheology, x_0)].name
            )

            i_t_tab = 0

            for t in t_values_in_process:

                file_path = model_results_path.joinpath(str(t) + ".json")

                # Verifies is the process ended.
                if file_path.exists():

                    # Updates 'in_process'.
                    del in_process[(rheology, x_0)][i_t_tab]
                    if len(in_process[(rheology, x_0)]) == 0:
                        del in_process[(rheology, x_0)]

                    # Updates 'just_processed'.
                    just_processed.add((rheology, x_0))

                    # Updates 'processed'.
                    processed[rheology][x_0] = add_sorted(
                        result_dict=processed[rheology][x_0],
                        t=t,
                        values=load_base_model(name=str(t), path=model_results_path),
                    )

                else:

                    i_t_tab += 1

        # Tests the stop cirterion on newly available results.
        while just_processed:

            rheology, x_0 = just_processed.pop()

            # Don't test for the stop criterion if processes are still running for the same model.
            if (rheology, x_0) not in in_process:

                # Verifies the stop criterion.
                f = numpy.array(object=processed[rheology][x_0]["values"])
                dt = numpy.diff(a=processed[rheology][x_0]["t"])
                dt_tab = numpy.expand_dims(a=dt, axis=1)
                df = numpy.diff(a=f, axis=0)
                d2f = numpy.diff(a=df, axis=0)
                normalized_curvature = abs(
                    (d2f * numpy.expand_dims(a=f[0] - f[-1], axis=0))
                    / (
                        dt_tab[:-1]  # Matches the number of elements in d2f.
                        * numpy.max(a=(df / dt_tab) ** 2.0)
                    )
                )
                indices = numpy.unique(
                    numpy.where(normalized_curvature > normalized_curvature_threshold)[0],
                    axis=0,
                )
                print(numpy.max(normalized_curvature, axis=0))
                new_t_values = numpy.array(
                    object=numpy.array(object=processed[rheology][x_0]["t"])[indices]
                    + dt[indices] / 2.0
                ).tolist()

                if new_t_values:

                    # Updates 'to_process'.
                    for t in new_t_values:
                        to_process.add(((rheology, x_0), t))

                else:

                    # Saves results for the whole model.
                    save_base_model(
                        obj={"rheology": rheology, "x_0": x_0} | processed[rheology][x_0],
                        name=models[(rheology, x_0)].name,
                        path=test_models_path.joinpath("results"),
                    )

                    # Updates 'processed'.
                    del processed[rheology][x_0]
                    if len(processed[rheology]) == 0:
                        del processed[rheology]
