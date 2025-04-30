"""
Worker to interpolate either Love numbers onthe same periods, test models on the same variable
parameters or Green functions on the same angles.
"""

from pathlib import Path

import numpy
from scipy import interpolate

from .database import load_base_model, save_base_model
from .paths import intermediate_result_subpaths
from .worker_parser import WorkerInformation, parse_worker_information

PARTS = ["real", "imag"]


def worker_interpolate(function_name: str) -> None:
    """
    Interpolates the output of an adaptative step algorithm for multiple rheologies.
    """

    interpolate_function_name = "interpolate_" + function_name
    worker_information: WorkerInformation = parse_worker_information(
        function_name=interpolate_function_name
    )
    exponentiation_scale = numpy.log(worker_information.fixed_parameter)
    save_path = intermediate_result_subpaths[interpolate_function_name].joinpath(
        worker_information.model_id
    )

    # Check whether the task has already been computed.
    if not save_path.exists():

        load_path = intermediate_result_subpaths[function_name].joinpath(
            worker_information.model_id
        )

        inputs = {part: {} for part in PARTS}
        variable_parameters = set()
        fixed_parameter_list = []

        for fixed_parameter_sub_path in load_path.iterdir():

            for part in PARTS:

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
            exponentiation_scale=exponentiation_scale,
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

            output.append(
                interpolate.interp1d(
                    x=numpy.log(
                        numpy.array(object=inputs[part][fixed_parameter]["variable_parameter"])
                    )
                    / exponentiation_scale,
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
