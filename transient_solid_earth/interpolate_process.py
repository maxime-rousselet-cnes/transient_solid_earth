"""
Worker to interpolate either Love numbers onthe same periods, test models on the same variable
parameters or Green functions on the same angles.
"""

from pathlib import Path

import numpy
from scipy import interpolate

from .database import load_base_model, save_base_model, save_complex_array
from .functions import generate_n_factor
from .paths import INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME, intermediate_result_subpaths
from .worker_parser import WorkerInformation

PARTS = ["real", "imag"]


def worker_interpolate(worker_information: WorkerInformation, function_name: str) -> None:
    """
    Interpolates the output of an adaptative step algorithm for multiple rheologies.
    """

    interpolate_function_name = "interpolate_" + function_name
    save_path = intermediate_result_subpaths[interpolate_function_name].joinpath(
        worker_information.model_id
    )

    # Check whether the task has already been computed.
    if save_path.joinpath(
        (
            INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME
            if worker_information.variable_parameter == 1.0
            else "real"
        )
        + ".json"
    ).exists():

        return

    if worker_information.variable_parameter == 1.0:

        interpolate_on_fixed_parameter(function_name=function_name, save_path=save_path)
        return

    load_path = intermediate_result_subpaths[function_name].joinpath(worker_information.model_id)
    inputs = {part: {} for part in PARTS}
    variable_parameters = set()
    fixed_parameter_list = []

    for fixed_parameter_sub_path in load_path.iterdir():

        for part in PARTS:

            if fixed_parameter_sub_path.joinpath(part + ".json").exists():

                input_part = load_base_model(name=part, path=fixed_parameter_sub_path)
                fixed_parameter = float(fixed_parameter_sub_path.name)
                inputs[part][fixed_parameter] = input_part

                if part == "real":

                    fixed_parameter_list.append(fixed_parameter)

                    for variable_parameter in input_part["variable_parameter"]:

                        variable_parameters.add(variable_parameter)

    variable_parameter_list = list(variable_parameters)
    variable_parameter_list.sort()
    fixed_parameter_list.sort()

    interpolate_on_variable_parameter(
        fixed_parameter_list=fixed_parameter_list,
        inputs=inputs,
        save_path=save_path,
        variable_parameter_list=variable_parameter_list,
        exponentiation_scale=numpy.log(worker_information.fixed_parameter),
    )


def interpolate_on_fixed_parameter(function_name: str, save_path: Path) -> None:
    """
    Interpolates on the fixed parameter axis supposing data already shares the variable parameter
    axis.
    """

    fixed_parameter_new_values = load_base_model(
        name="fixed_parameter_new_values", path=save_path.parent
    )
    result = {}

    for part in PARTS:

        input_data = load_base_model(name=part, path=save_path)
        n_factor_pre_interpolation = (
            1
            if "love_numbers" not in function_name
            else generate_n_factor(fixed_parameter_values=input_data["fixed_parameter"])
        )
        n_factor_post_interpolation = (
            1
            if "love_numbers" not in function_name
            else generate_n_factor(fixed_parameter_values=fixed_parameter_new_values)
        )
        result[part] = (
            interpolate.interp1d(
                x=input_data["fixed_parameter"],
                y=numpy.array(object=input_data["values"]) * n_factor_pre_interpolation,
                axis=0,
            )(x=fixed_parameter_new_values)
            / n_factor_post_interpolation
        )

    save_complex_array(
        obj=result, name=INTERPOLATED_ON_FIXED_PARAMETER_SUBPATH_NAME, path=save_path
    )
    save_base_model(
        obj=input_data["variable_parameter"], name="variable_parameter_values", path=save_path
    )


def interpolate_on_variable_parameter(
    fixed_parameter_list: list[float],
    inputs: dict[str, dict[float, dict[str, list]]],
    save_path: Path,
    variable_parameter_list: list[float],
    exponentiation_scale: float,
) -> None:
    """
    Interpolates on the variable parameter axis for all fixed parameter values.
    """

    for part in PARTS:

        output = []

        for fixed_parameter in fixed_parameter_list:

            variable_parameter = numpy.array(
                object=inputs[part][fixed_parameter]["variable_parameter"]
            )

            if numpy.inf in variable_parameter:
                # Handles the elastic case.
                output.append(inputs[part][fixed_parameter]["values"])  # Length 1 along axis 1.
            else:
                output.append(
                    interpolate.interp1d(
                        x=numpy.log(variable_parameter) / exponentiation_scale,
                        y=inputs[part][fixed_parameter]["values"],
                        axis=0,
                    )(x=numpy.log(variable_parameter_list) / exponentiation_scale)
                )

        save_base_model(
            obj={
                "fixed_parameter": fixed_parameter_list,
                "variable_parameter": variable_parameter_list,
                "values": output,
            },
            name=part,
            path=save_path,
        )
