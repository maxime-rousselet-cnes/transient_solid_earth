"""
Describes the loop for interpolation on the wanted axes for all considered rheologies.
"""

from copy import deepcopy
from typing import Optional

import numpy

from .database import save_base_model
from .model_type_names import MODEL_TYPE_NAMES
from .parameters import DEFAULT_PARAMETERS, Parameters
from .paths import intermediate_result_subpaths
from .process_catalog import ProcessCatalog


class InterpolateProcessCatalog(ProcessCatalog):
    """
    Memorizes all interpolation processes to runand schedule them.
    """

    def __init__(
        self,
        rheologies: list[dict],
        parameters: Parameters,
        period_new_values_per_id: dict[str, list[float] | numpy.ndarray],
        degree_new_values: list[float] | numpy.ndarray,
    ) -> None:
        """
        Memorizes the IDs of the models to interpolate. Saves the peroid/degree grid.
        """

        super().__init__(
            function_name="interpolate_love_numbers",
            parallel_computing_parameters=parameters.parallel_computing,
        )

        save_base_model(
            obj=degree_new_values,
            name="degrees",
            path=intermediate_result_subpaths["interpolate_love_numbers"],
        )

        for interpolation_basis_id, period_new_values in period_new_values_per_id.items():

            save_base_model(
                obj=period_new_values,
                name="periods",
                path=intermediate_result_subpaths["interpolate_love_numbers"].joinpath(
                    interpolation_basis_id
                ),
            )

            for rheology in rheologies:
                self.to_process.add(
                    (
                        MODEL_TYPE_NAMES["interpolate_love_numbers"](
                            solid_earth_parameters=deepcopy(parameters.solid_earth),
                            rheology=rheology,
                        ).model_id,
                        # To interpolate on periods in log scale.
                        parameters.discretization["love_numbers"].exponentiation_base,
                        interpolation_basis_id,
                    )
                )


def interpolate_parallel_computing_loop(
    rheologies: list[dict],
    period_new_values_per_id: dict[str, list[float] | numpy.ndarray],
    degree_new_values: list[float] | numpy.ndarray,
    parameters: Parameters = DEFAULT_PARAMETERS,
    timeout: Optional[float] = None,
) -> None:
    """
    For every rheologies, interpolates Love numbers.
    """

    # Initializes data structures.
    process_catalog = InterpolateProcessCatalog(
        rheologies=rheologies,
        parameters=parameters,
        period_new_values_per_id=period_new_values_per_id,
        degree_new_values=degree_new_values,
    )

    # Runs a job per rheology and per fixed parameter.
    process_catalog.schedule_jobs()

    # Waits for the jobs to finish.
    process_catalog.wait_for_jobs(timeout=timeout)
