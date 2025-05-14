"""
Describes the loop computing Love number asymptotic values for all considered rheologies.
"""

from copy import deepcopy

from .model_type_names import MODEL_TYPE_NAMES
from .parameters import DEFAULT_PARAMETERS, Parameters
from .process_catalog import ProcessCatalog


class AsymptoticLoveNumbersProcessCatalog(ProcessCatalog):
    """
    Prepares Love numbers for Green function computations.
    """

    def __init__(
        self,
        rheologies: list[dict],
        parameters: Parameters,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        super().__init__(
            function_name="asymptotic_love_numbers",
            parallel_computing_parameters=parameters.parallel_computing,
        )

        integration_parameters = parameters.solid_earth.numerical_parameters.integration_parameters

        for rheology in rheologies:
            self.to_process.add(
                (
                    MODEL_TYPE_NAMES["asymptotic_love_numbers"](
                        solid_earth_parameters=deepcopy(parameters.solid_earth),
                        rheology=rheology,
                    ).model_id,
                    integration_parameters.n_min_for_asymptotic_behavior,
                    0.0,
                )
            )


def asymptotic_love_numbers_computing_loop(
    rheologies: list[dict],
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> None:
    """
    Prepares Love numbers for Green function computations.
    """

    # Initializes data structures.
    process_catalog = AsymptoticLoveNumbersProcessCatalog(
        rheologies=rheologies,
        parameters=parameters,
    )

    # Runs a job per rheology and per fixed parameter.
    process_catalog.schedule_jobs()
    process_catalog.wait_for_jobs()
