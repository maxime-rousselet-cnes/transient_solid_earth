"""
This module is responsible for preparing the loop on parameters.
"""

from copy import deepcopy
from itertools import product
from typing import Any, Optional

from .parameters import (
    SOLID_EARTH_MODEL_ALL_OPTION_PARAMETERS,
    LoadNumericalModelParameters,
    SolidEarthModelOptionParameters,
    SolidEarthVariableParameters,
)
from .paths import SolidEarthModelPart, solid_earth_model_descriptions_path
from .separators import LAYERS_SEPARATOR, UNUSED_MODEL_PART_DEFAULT_NAME, VALUES_SEPARATOR
from .solid_earth_model_description import SolidEarthModelDescription


def create_solid_earth_model_description_variation(
    model_part: SolidEarthModelPart,
    solid_earth_model_description: SolidEarthModelDescription,
    model_description_name: Optional[str],
    new_parameter_values: list[tuple[str, str, list[float]]],
) -> str:
    """
    Gets an initial model description and creates a new version of it by modifying the specified
    parameters with specified polynomials per layer.
    Only creates the file if needed. Returns the name.
    """

    # Builds the new model's name.
    new_model_description_name = LAYERS_SEPARATOR.join(
        [
            VALUES_SEPARATOR.join(
                [parameter_name, layer_name] + [str(value) for value in polynomial]
            )
            for parameter_name, layer_name, polynomial in new_parameter_values
        ]
    )

    # Modifies its values and save it to a file.
    for parameter_name, layer_name_part, polynomial in new_parameter_values:
        for layer_name in solid_earth_model_description.layer_names:
            if layer_name_part in layer_name:
                solid_earth_model_description.polynomials[parameter_name][
                    solid_earth_model_description.layer_names.index(layer_name)
                ] = polynomial
    solid_earth_model_description.save(
        name=new_model_description_name,
        path=solid_earth_model_descriptions_path[model_part].joinpath(model_description_name),
    )

    return new_model_description_name


def sum_lists(lists: list[list]) -> list:
    """
    Concatenates lists. Needed to iterate on parameter variations.
    """

    concatenated_list = []

    for elt in lists:
        for sub_elt in elt:
            concatenated_list += sub_elt

    return concatenated_list


def generate_parameter_value_possibilities(
    model_description: SolidEarthModelDescription,
    model_part_rheological_parameters: dict[str, dict[str, list[list[float]]]],
) -> list:
    """
    Generates an iterable of all possible list of parameter values for a given model description.
    """

    return product(
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
                                model_description.polynomials[parameter_name][
                                    model_description.layer_names.index(layer_name)
                                ]
                                for layer_name in model_description.layer_names
                                if layer_name_part in layer_name
                            ]
                            # Adds default values in the list to iterate on.
                            + parameter_values_per_possibility
                        )
                    )
                    for (
                        layer_name_part,
                        parameter_values_per_possibility,
                    ) in parameter_values_per_layer.items()
                )
            )
            for (
                parameter_name,
                parameter_values_per_layer,
            ) in model_part_rheological_parameters.items()
        )
    )


def create_all_model_variations(
    variable_parameters: SolidEarthVariableParameters,
    solid_earth_model_option_list: Optional[list[SolidEarthModelOptionParameters]] = None,
) -> list[tuple[dict[SolidEarthModelPart, str], list[dict[SolidEarthModelPart, str]]]]:
    """
    Creates all possible variations of parameters for the wanted models and creates the
    corresponding files accordingly. Returns a dictionary that describes their rheology parts,
    for all of them.
    Choses deterministically a base numerical model for elastic reference for every elastic model
    description. Thus, each tuple of the output list represents a couple of:
        - an elastic model.
        - all the corresponding anelastic models without redundancy.
    """

    if not solid_earth_model_option_list:
        solid_earth_model_option_list = SOLID_EARTH_MODEL_ALL_OPTION_PARAMETERS

    # Generates a structure to contain all possible model descriptions.
    model_description_filenames: dict[SolidEarthModelPart, list[str]] = {
        model_part: [] for model_part in SolidEarthModelPart
    }
    for model_part, part_model_names in variable_parameters.model_names.items():

        if (model_part in variable_parameters.rheological_parameters.keys()) and (
            variable_parameters.rheological_parameters[model_part] != {}
        ):

            for model_description_name in part_model_names:
                model_description = SolidEarthModelDescription(
                    name=model_description_name, solid_earth_model_part=model_part
                )
                model_part_rheological_parameters = variable_parameters.rheological_parameters[
                    model_part
                ]
                # Adds all possible combinations.
                model_description_filenames[model_part] += list(
                    {
                        model_description_name
                        + "/"
                        + create_solid_earth_model_description_variation(
                            model_part=model_part,
                            solid_earth_model_description=model_description,
                            model_description_name=model_description_name,
                            new_parameter_values=sum_lists(lists=parameter_values),
                        )
                        # Iterates on all combination possibilities.
                        for parameter_values in generate_parameter_value_possibilities(
                            model_description=model_description,
                            model_part_rheological_parameters=model_part_rheological_parameters,
                        )
                    }
                )

        else:

            model_description_filenames[model_part] = part_model_names

        model_description_filenames[model_part].sort()

    # Merges for non-redundancy.
    all_model_variations = []
    for elastic_model_name in model_description_filenames[SolidEarthModelPart.ELASTICITY]:
        all_anelastic_model_variations = {}
        for long_term_anelasticity_model_name, short_term_anelasticity_model_name in product(
            model_description_filenames[SolidEarthModelPart.LONG_TERM_ANELASTICITY],
            model_description_filenames[SolidEarthModelPart.SHORT_TERM_ANELASTICITY],
        ):
            for options in solid_earth_model_option_list:
                # TODO.
                all_anelastic_model_variations[
                    (
                        (
                            long_term_anelasticity_model_name
                            if options.use_long_term_anelasticity
                            else UNUSED_MODEL_PART_DEFAULT_NAME
                        ),
                        (
                            short_term_anelasticity_model_name
                            if options.use_short_term_anelasticity
                            else UNUSED_MODEL_PART_DEFAULT_NAME
                        ),
                    )
                ] = {
                    SolidEarthModelPart.ELASTICITY: elastic_model_name,
                    SolidEarthModelPart.LONG_TERM_ANELASTICITY: (
                        long_term_anelasticity_model_name
                        if options.use_long_term_anelasticity
                        else UNUSED_MODEL_PART_DEFAULT_NAME
                    ),
                    SolidEarthModelPart.SHORT_TERM_ANELASTICITY: (
                        short_term_anelasticity_model_name
                        if options.use_short_term_anelasticity
                        else UNUSED_MODEL_PART_DEFAULT_NAME
                    ),
                }
        all_model_variations += [
            (
                {
                    SolidEarthModelPart.ELASTICITY: elastic_model_name,
                    SolidEarthModelPart.LONG_TERM_ANELASTICITY: UNUSED_MODEL_PART_DEFAULT_NAME,
                    SolidEarthModelPart.SHORT_TERM_ANELASTICITY: UNUSED_MODEL_PART_DEFAULT_NAME,
                },
                list(all_anelastic_model_variations.values()),
            )
        ]

    return all_model_variations


def extract_parameter_paths(
    param_dict: dict[str, Any], prefix: tuple = ()
) -> list[tuple[tuple[str, ...], list]]:
    """
    Recursively extracts attribute paths and their list of values
    from a nested dictionary.
    Returns a list of (path_tuple, values).
    """
    paths = []
    for key, value in param_dict.items():
        current_path = prefix + (key,)
        if isinstance(value, dict):
            paths.extend(extract_parameter_paths(value, current_path))
        else:
            paths.append((current_path, value))
    return paths


def set_attr_by_path(obj: Any, path: tuple[str, Any], value: Any):
    """
    Sets nested attribute value in obj given a path tuple.
    """
    for attr in path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, path[-1], value)


def generate_object_variants_from_nested_parameters_grid(
    load_model_parameters: LoadNumericalModelParameters, load_model_variabilities: dict[str, Any]
) -> list[LoadNumericalModelParameters]:
    """
    Creates all possible variations of load signal model parameters.
    """

    paths_and_values = extract_parameter_paths(load_model_variabilities)
    paths = [p for p, _ in paths_and_values]
    value_lists = [v for _, v in paths_and_values]

    combinations = product(*value_lists)
    results = []

    for combo in combinations:
        obj_copy = deepcopy(load_model_parameters)
        for path, value in zip(paths, combo):
            set_attr_by_path(obj_copy, path, value)
        results.append(obj_copy)

    return results
