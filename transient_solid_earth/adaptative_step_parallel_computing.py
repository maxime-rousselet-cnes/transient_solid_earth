"""
Describes the loop using adaptatives step for all considered rheologies.
"""

import math
from copy import deepcopy
from typing import Optional, Type

import numpy

from .database import load_complex_array, save_complex_array
from .file_creation_observer import FileCreationObserver
from .functions import add_sorted, round_value
from .generic_rheology_model import MODEL, GenericRheologyModel
from .parameters import DEFAULT_PARAMETERS, DiscretizationParameters, Parameters
from .paths import intermediate_result_subpaths
from .process_catalog import ProcessCatalog
from .separators import is_elastic
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel

MODEL_TYPE_NAMES: dict[str, Type[MODEL]] = {
    "love_numbers": SolidEarthFullNumericalModel,
    "interpolate_love_numbers": GenericRheologyModel,
}


class AdaptativeStepProcessCatalog(ProcessCatalog):
    """
    Recap of all processes to be launched, in process, just processed and whose results have been
    stored. Manages the whole adaptative step parallel computing loop on periods.
    """

    # To memorize.
    discretization_parameters: DiscretizationParameters

    # Actual process logs.
    file_creation_observer: FileCreationObserver

    just_processed: set[tuple[str, float, float]] = set()
    processed: dict[tuple[str, float], dict[str, numpy.ndarray]] = {}  # "x" tab in log scale.
    model_ids: list[str] = []
    elastic_model_id: str

    def __init__(
        self,
        degree_list: list[float],
        rheologies: list[dict],
        parameters: Parameters,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        super().__init__(
            function_name="love_numbers",
            parallel_computing_parameters=parameters.parallel_computing,
        )

        self.discretization_parameters = parameters.discretization["love_numbers"]

        self.file_creation_observer = FileCreationObserver(
            base_path=intermediate_result_subpaths[self.function_name]
        )

        # Generates the loop's initial discretization.
        initial_period_list = round_value(
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

            model_id = MODEL_TYPE_NAMES["love_numbers"](
                solid_earth_parameters=deepcopy(parameters.solid_earth), rheology=rheology
            ).model_id

            for degree in degree_list:
                self.processed[(model_id, degree)] = {
                    "x": numpy.array(object=[]),
                    "values": numpy.array(object=[]),
                }
                for period in initial_period_list:
                    self.to_process.add((model_id, degree, period))

            self.model_ids += [model_id]

            if is_elastic(model_id=model_id):

                self.elastic_model_id = model_id

    def get_results(self) -> None:
        """
        Loads the already computed values and updates the 'in_process' and 'processed' attributes.
        Updates 'just_processed' and 'processed'.
        """

        for path in self.file_creation_observer.get_created_file_paths():

            period = float(path.name)
            degree = float(path.parent.name)
            model_id = path.parent.parent.name

            if (model_id, degree) in self.in_process:

                if period in self.in_process[(model_id, degree)] or period == float("inf"):

                    # Updates 'in_process'.
                    if period in self.in_process[(model_id, degree)]:
                        self.in_process[(model_id, degree)].remove(period)
                    if len(self.in_process[(model_id, degree)]) == 0 or period == float("inf"):
                        del self.in_process[(model_id, degree)]

                    # Updates 'just_processed'.
                    self.just_processed.add((model_id, degree, period))

                    # Updates 'processed'.
                    self.processed[(model_id, degree)] = add_sorted(
                        result_dict=self.processed[model_id, degree],
                        x=math.log(period, self.discretization_parameters.exponentiation_base),
                        values=load_complex_array(path=path),
                    )

    def refine_discretization(self) -> None:
        """
        Refines the discretization on the variable parameter for a rheology and a degree.
        Stops when the result is approximated everywhere by its linear interpolation.
        """

        to_postprocess: set[tuple[str, float]] = set()
        while self.just_processed:
            model_id, degree, _ = self.just_processed.pop()
            if (model_id, degree) not in self.in_process:
                to_postprocess.add((model_id, degree))

        # Don't test for the stop criterion if processes are still running for the same model.
        for model_id, degree in to_postprocess:

            # Verifies the stop criterion on the whole sequence.
            f = self.processed[(model_id, degree)]["values"]  # shape (T, *S)
            x = self.processed[(model_id, degree)]["x"]  # shape (T,)

            # Reshapes time arrays to broadcast with f.
            x_0 = x[:-2].reshape(-1, *([1] * (f.ndim - 1)))  # shape (T-2, 1, ..., 1)
            x_1 = x[1:-1].reshape(-1, *([1] * (f.ndim - 1)))
            x_2 = x[2:].reshape(-1, *([1] * (f.ndim - 1)))
            x_span = x_2 - x_0

            # Computes initial mask based on deviation from linear interpolation.
            mask = numpy.any(
                numpy.abs(((x_1 - x_0) * f[2:] + (x_2 - x_1) * f[:-2]) / x_span - f[1:-1])
                > self.discretization_parameters.maximum_tolerance
                * (
                    numpy.abs(f[2:] - f[:-2])  # Local variation
                    + numpy.abs(numpy.max(f, axis=0) - numpy.min(f, axis=0))  # Global scale
                ),
                axis=tuple(range(1, f.ndim)),  # Apply across all non-x dimensions
            )

            # Suppresses refinement if the function varies too little between adjacent points.
            mask &= ~numpy.all(
                numpy.abs(f[2:] - f[1:-1])
                < self.discretization_parameters.precision * numpy.abs(f[1:-1]),
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
                                        (x_left + x_right) / 2.0
                                        for x_left, x_right in zip(
                                            x_left_tab[mask], x_right_tab[mask]
                                        )
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
                    self.to_process.add((model_id, degree, variable_value))

            else:

                # Saves results for the whole model.
                path = (
                    intermediate_result_subpaths[self.function_name]
                    .joinpath(model_id)
                    .joinpath(str(degree))
                )
                save_complex_array(obj=self.processed[model_id, degree]["values"], path=path)


def adaptative_step_parallel_computing_loop(
    rheologies: list[dict],
    degree_list: Optional[list[float]] = None,
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> tuple[list[str], str]:
    """
    For every rheologies, uses an adaptative step on frequencies.
    """

    # Manages default.
    if not degree_list:
        degree_list = [0.0]

    # Initializes data structures.
    process_catalog = AdaptativeStepProcessCatalog(
        degree_list=degree_list,
        rheologies=rheologies,
        parameters=parameters,
    )

    try:

        # Loops until the stop criterion is verified for every rheologies and degree.
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

    return process_catalog.model_ids, process_catalog.elastic_model_id
