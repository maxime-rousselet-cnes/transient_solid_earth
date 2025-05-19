"""
Describes the loop for interpolation on the wanted axes for all considered rheologies.
"""

from copy import deepcopy
from typing import Optional

import numpy

from .database import save_base_model
from .model_type_names import MODEL_TYPE_NAMES
from .parameters import DEFAULT_PARAMETERS, Parameters
from .paths import INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME, intermediate_result_subpaths
from .process_catalog import ProcessCatalog


class InterpolateProcessCatalog(ProcessCatalog):
    """
    Memorizes all interpolation processes to runand schedule them.
    """

    def __init__(
        self,
        function_name: str,
        rheologies: list[dict],
        parameters: Parameters,
        fixed_parameter_new_values: Optional[list[float] | numpy.ndarray] = None,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        interpolate_function_name = "interpolate_" + function_name
        super().__init__(
            function_name=interpolate_function_name,
            parallel_computing_parameters=parameters.parallel_computing,
        )

        for rheology in rheologies:
            self.to_process.add(
                (
                    MODEL_TYPE_NAMES[interpolate_function_name](
                        solid_earth_parameters=deepcopy(parameters.solid_earth),
                        rheology=rheology,
                    ).model_id,
                    parameters.discretization[function_name].exponentiation_base,
                    1.0 if fixed_parameter_new_values else 0.0,
                )
            )

        if fixed_parameter_new_values:
            save_base_model(
                obj=fixed_parameter_new_values,
                name="fixed_parameter_new_values",
                path=intermediate_result_subpaths[interpolate_function_name],
            )


def interpolate_parallel_computing_loop(
    function_name: str,
    rheologies: list[dict],
    parameters: Parameters = DEFAULT_PARAMETERS,
    fixed_parameter_new_values: Optional[list[float] | numpy.ndarray] = None,
    timeout: Optional[float] = None,
) -> None:
    """
    For every rheologies, interpolates on the chosen axis.
    Interpolates on the fixed parameter axis and creates a grid instance if
    'fixed_parameter_new_values' parameter is specified.
    Interpolates on the variable parameter axis on shared values if 'fixed_parameter_new_values'
    is not specified.
    """

    # Initializes data structures.
    process_catalog = InterpolateProcessCatalog(
        function_name=function_name,
        rheologies=rheologies,
        parameters=parameters,
        fixed_parameter_new_values=fixed_parameter_new_values,
    )

    # Runs a job per rheology and per fixed parameter.
    process_catalog.schedule_jobs()

    # Waits for the jobs to finish.
    process_catalog.wait_for_jobs(
        subpath_name=(
            INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME if fixed_parameter_new_values else None
        ),
        timeout=timeout,
    )
