"""
Describes the loop using adaptatives step for all considered rheologies.
"""

import math
from copy import deepcopy
from typing import Optional

import numpy

from .database import load_base_model, save_base_model
from .file_creation_observer import FileCreationObserver
from .functions import add_sorted, round_value
from .model_type_names import MODEL_TYPE_NAMES
from .parameters import DEFAULT_PARAMETERS, DiscretizationParameters, Parameters
from .paths import intermediate_result_subpaths
from .process_catalog import ProcessCatalog


class AdaptativeStepProcessCatalog(ProcessCatalog):
    """
    Recap of all processes to be launched, in process, just processed and whose results have been
    stored. Manages the whole adaptative step parallel computing loop.
    """

    # To memorize.
    discretization_parameters: DiscretizationParameters

    # Actual process logs.
    file_creation_observer: FileCreationObserver

    just_processed: set[tuple[str, float, float]] = set()
    processed: dict[tuple[str, float], dict[str, numpy.ndarray]] = {}  # "x" tab in log scale.

    def __init__(
        self,
        fixed_parameter_list: list[float],
        rheologies: list[dict],
        function_name: str,
        parameters: Parameters,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        super().__init__(
            function_name=function_name, parallel_computing_parameters=parameters.parallel_computing
        )

        self.discretization_parameters = parameters.discretization[function_name]

        self.file_creation_observer = FileCreationObserver(
            base_path=intermediate_result_subpaths[self.function_name]
        )

        # Generates the loop's initial discretization.
        initial_variable_parameter_list = round_value(
            t=self.discretization_parameters.exponentiation_base
            ** numpy.linspace(
                start=math.log(
                    self.discretization_parameters.value_min,
                    self.discretization_parameters.exponentiation_base,
                ),
                stop=math.log(
                    self.discretization_parameters.value_max,
                    self.discretization_parameters.exponentiation_base,
                ),
                num=self.discretization_parameters.n_0,
            ),
            rounding=self.discretization_parameters.rounding,
        )

        for rheology in rheologies:
            model_id = MODEL_TYPE_NAMES[function_name](
                solid_earth_parameters=deepcopy(parameters.solid_earth), rheology=rheology
            ).model_id
            for fixed_parameter in fixed_parameter_list:
                self.processed[(model_id, fixed_parameter)] = {
                    "x": numpy.array(object=[]),
                    "values": numpy.array(object=[]),
                }
                for variable_parameter in initial_variable_parameter_list:
                    self.to_process.add((model_id, fixed_parameter, variable_parameter))

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
            new_variable_values = (
                self.discretization_parameters.exponentiation_base
                ** numpy.setdiff1d(
                    numpy.unique(
                        round_value(
                            t=numpy.concatenate(
                                [
                                    [
                                        (x_left[mask] + x_right[mask]) / 2.0
                                        for x_left, x_right in zip(x_left_tab, x_right_tab)
                                        if x_right - x_left
                                        > math.log(
                                            self.discretization_parameters.min_step,
                                            self.discretization_parameters.exponentiation_base,
                                        )
                                    ]
                                    for x_left_tab, x_right_tab in zip(
                                        [x[:-2], x[1:-1]], [x[1:-1], x[2:]]
                                    )
                                ]
                            ),
                            rounding=self.discretization_parameters.rounding,
                        )
                    ),
                    x,
                )
            )

            if len(new_variable_values) != 0:

                # Updates 'to_process'.
                for variable_value in new_variable_values:
                    self.to_process.add((model_id, fixed_parameter, variable_value))

            else:

                # Saves results for the whole model.
                variable_parameter_tab = round_value(
                    t=self.discretization_parameters.exponentiation_base
                    ** self.processed[model_id, fixed_parameter]["x"],
                    rounding=self.discretization_parameters.rounding,
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
    function_name: str,
    fixed_parameter_list: Optional[list[float]] = None,
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> None:
    """
    For every rheologies, uses an adaptative step.
    """

    # Manages default.
    if not fixed_parameter_list:
        fixed_parameter_list = [0.0]

    # Initializes data structures.
    process_catalog = AdaptativeStepProcessCatalog(
        fixed_parameter_list=fixed_parameter_list,
        rheologies=rheologies,
        function_name=function_name,
        parameters=parameters,
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
