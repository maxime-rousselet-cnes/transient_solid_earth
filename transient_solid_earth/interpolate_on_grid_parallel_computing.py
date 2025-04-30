"""
Describes the loop using adaptatives step for all considered rheologies.
"""

from copy import deepcopy

from .model_type_names import MODEL_TYPE_NAMES
from .parameters import DEFAULT_PARAMETERS, Parameters
from .process_catalog import ProcessCatalog


class InterpolateProcessCatalog(ProcessCatalog):
    """
    Memorizes all interpolation processes to run.
    """

    def __init__(
        self,
        function_name: str,
        rheologies: list[dict],
        parameters: Parameters,
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
                    0.0,  # Useless parameters for interpolation.
                )
            )


def interpolate_on_grid_parallel_computing_loop(
    function_name: str,
    rheologies: list[dict],
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> None:
    """
    For every rheologies, uses an adaptative step.
    """

    # Initializes data structures.
    process_catalog = InterpolateProcessCatalog(
        function_name=function_name,
        rheologies=rheologies,
        parameters=parameters,
    )

    # Runs a job per rheology and per fixed parameter.
    process_catalog.schedule_jobs()

    # Releases all the slots.
    process_catalog.wait()
