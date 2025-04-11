from copy import deepcopy
from itertools import product
from typing import Any, Optional

from .paths import SolidEarthModelPart
from .solid_earth_model_description import SolidEarthModelDescription


def create_model_variation(
    model_part: SolidEarthModelPart,
    base_model_description: SolidEarthModelDescription,
    base_model_name: Optional[str],
    parameter_values: list[tuple[str, str, list[float]]],
    create: bool = True,
) -> str:
    """
    Gets an initial model file and creates a new version of it by modifying the specified
    parameters with specified polynomials per layer. Only creates the file if needed. Returns the name.
    """

    # Builds the new model's name.
    name = LAYERS_SEPARATOR.join(
        [
            VALUES_SEPARATOR.join(
                [parameter_name, layer_name] + [str(value) for value in polynomial]
            )
            for parameter_name, layer_name, polynomial in parameter_values
        ]
    )

    # Eventually modifies its values and save it to a file.
    if create:
        for parameter_name, layer_name_part, polynomial in parameter_values:
            for layer_name in base_model.layer_names:
                if layer_name_part in layer_name:
                    base_model.polynomials[parameter_name][
                        base_model.layer_names.index(layer_name)
                    ] = polynomial
        base_model.save(
            name=name,
            path=models_path[model_part].joinpath(base_model_name),
        )

    return name


def sum_lists(lists: list[list]) -> list:
    """
    Concatenates lists. Needed to iterate on parameter variations.
    """
    concatenated_list = []
    for elt in lists:
        for sub_elt in elt:
            concatenated_list += sub_elt
    return concatenated_list


def create_all_model_variations(
    elasticity_model_names: list[Optional[str]] = [None],
    long_term_anelasticity_model_names: list[Optional[str]] = [None],
    short_term_anelasticity_model_names: list[Optional[str]] = [None],
    parameters: dict[SolidEarthModelPart, dict[str, dict[str, list[list[float]]]]] = {
        model_part: {} for model_part in SolidEarthModelPart
    },
    create: bool = True,
) -> dict[SolidEarthModelPart, list[str]]:
    """
    Creates all possible variations of parameters for the wanted models and creates the corresponding files accordingly.
    Returns all their IDs.
    """
    model_filenames: dict[SolidEarthModelPart, list[str]] = {
        model_part: [] for model_part in SolidEarthModelPart
    }
    for model_part, model_names in zip(
        SolidEarthModelPart,
        [
            elasticity_model_names,
            long_term_anelasticity_model_names,
            short_term_anelasticity_model_names,
        ],
    ):
        if (model_part in parameters.keys()) and (parameters[model_part] != {}):
            for model_name in model_names:
                model = Model(name=model_name, model_part=model_part)
                # Adds all possible combinations.
                model_filenames[model_part] += list(
                    set(
                        [
                            model_name
                            + "/"
                            + create_model_variation(
                                model_part=model_part,
                                base_model=model,
                                base_model_name=model_name,
                                parameter_values=sum_lists(lists=parameter_values),
                                create=create,
                            )
                            for parameter_values in product(  # Iterates on all combinations possibilities.
                                *(
                                    product(
                                        *(
                                            product(
                                                (
                                                    (
                                                        parameter_name,
                                                        layer_name_part,
                                                        parameter_values_possibility,
                                                    )
                                                    for parameter_values_possibility in [
                                                        model.polynomials[parameter_name][
                                                            model.layer_names.index(layer_name)
                                                        ]
                                                        for layer_name in model.layer_names
                                                        if layer_name_part in layer_name
                                                    ]
                                                    + parameter_values_per_possibility  # Adds default values in the list of values to iterate on.
                                                )
                                            )
                                            for layer_name_part, parameter_values_per_possibility in parameter_values_per_layer.items()
                                        )
                                    )
                                    for parameter_name, parameter_values_per_layer in parameters[
                                        model_part
                                    ].items()
                                )
                            )
                        ]
                    )
                )
        else:
            model_filenames[model_part] = model_names

    return model_filenames


def create_load_signal_hyper_parameter_variation(
    base_load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    parameter_values=list[list[tuple[str, Any]]],
) -> LoadSignalHyperParameters:
    """
    Creates a variation of load signal hyper parameters.
    """
    load_signal_hyper_parameters = deepcopy(base_load_signal_hyper_parameters)
    for parameter_tuple in parameter_values:
        setattr(load_signal_hyper_parameters, parameter_tuple[0][0], parameter_tuple[0][1])

    if "DDK" not in load_signal_hyper_parameters.load_spatial_behaviour_file:
        load_signal_hyper_parameters.load_spatial_behaviour_file = (
            "DDK"
            + str(load_signal_hyper_parameters.ddk_filter_level)
            + "/"
            + load_signal_hyper_parameters.load_spatial_behaviour_file
        )

    return load_signal_hyper_parameters


def create_all_load_signal_hyper_parameters_variations(
    base_load_signal_hyper_parameters: LoadSignalHyperParameters = load_load_signal_hyper_parameters(),
    load_signal_parameters: dict[str, list] = {},
) -> list[LoadSignalHyperParameters]:
    """
    Creates all possible variations of load signal hyper parameters.
    """
    return [
        create_load_signal_hyper_parameter_variation(
            base_load_signal_hyper_parameters=base_load_signal_hyper_parameters,
            parameter_values=parameter_values,
        )
        for parameter_values in product(  # Iterates on all combinations possibilities.
            *(
                product(
                    (parameter_name, parameter_value_possibility)
                    for parameter_value_possibility in parameter_values_per_possibility
                )
                for parameter_name, parameter_values_per_possibility in load_signal_parameters.items()
            )
        )
    ]


def find_minimal_computing_options(
    options: list[RunHyperParameters],
    long_term_anelasticity_model_name: str,
    short_term_anelasticity_model_name: str,
    reference_model_filenames: dict[SolidEarthModelPart, str],
) -> list[RunHyperParameters]:
    """
    Tells whether it is necessary to compute elastic case, long_term_only case and short_term_only case or to point to
    equivalent model's results.
    """

    has_reference_long_term_anelasticity = (
        long_term_anelasticity_model_name
        == reference_model_filenames[SolidEarthModelPart.long_term_anelasticity]
    )
    has_reference_short_term_anelasticity = (
        short_term_anelasticity_model_name
        == reference_model_filenames[SolidEarthModelPart.short_term_anelasticity]
    )

    # Returns the list of options that are needed for computation.
    return [
        run_hyper_parameters
        for run_hyper_parameters in [ELASTIC_RUN_HYPER_PARAMETERS] + options
        if (
            (has_reference_long_term_anelasticity and has_reference_short_term_anelasticity)
            or (
                run_hyper_parameters.use_short_term_anelasticity
                and has_reference_long_term_anelasticity
            )
            or (
                run_hyper_parameters.use_long_term_anelasticity
                and has_reference_short_term_anelasticity
            )
            or (
                run_hyper_parameters.use_long_term_anelasticity
                and run_hyper_parameters.use_short_term_anelasticity
            )
        )
    ]
