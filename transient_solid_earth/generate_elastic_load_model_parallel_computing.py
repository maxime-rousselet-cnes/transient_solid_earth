"""
Describes the loop for interpolation on the wanted axes for all considered rheologies.
"""

from typing import Optional

from .database import add_result_to_table, extract_terminal_attributes, save_base_model
from .model_list_generation import generate_load_model_variations
from .parameters import DEFAULT_PARAMETERS, LoadModelParameters, Parameters
from .paths import elastic_load_model_parameters_subpath
from .process_catalog import ProcessCatalog


class GenerateElasticLoadModelProcessCatalog(ProcessCatalog):
    """
    Memorizes all interpolation processes to runand schedule them.
    """

    def __init__(
        self,
        load_model_variations: list[LoadModelParameters],
        parameters: Parameters,
    ) -> None:
        """
        Generates the rheologies and saves the numerical models. Memorizes the IDs.
        """

        super().__init__(
            function_name="generate_elastic_load_models",
            parallel_computing_parameters=parameters.parallel_computing,
        )

        for load_model in load_model_variations:

            load_model_line = extract_terminal_attributes(obj=load_model)
            load_model_line["ID"] = load_model.model_id()
            add_result_to_table(table_name="elastic_load_models", dictionary=load_model_line)
            save_base_model(
                obj=load_model,
                name=load_model_line["ID"],
                path=elastic_load_model_parameters_subpath,
            )
            self.to_process.add(
                (
                    load_model_line["ID"],
                    # Only needs the model ID for this parallelization.
                    0.0,
                    0.0,
                )
            )


def generate_elastic_load_models_parallel_loop(
    parameters: Parameters = DEFAULT_PARAMETERS,
    timeout: Optional[float] = None,
) -> None:
    """
    Generates the elastic load models.
    """

    # Generates load model parameter variations.
    load_model_variations = generate_load_model_variations(
        load_model_parameters=parameters.load.model,
        load_model_variabilities=parameters.load_model_variabilities,
    )

    # Initializes data structures.
    process_catalog = GenerateElasticLoadModelProcessCatalog(
        load_model_variations=load_model_variations, parameters=parameters
    )

    # Runs a job per rheology and per fixed parameter.
    process_catalog.schedule_jobs()

    # Waits for the jobs to finish.
    process_catalog.wait_for_jobs(timeout=timeout)
