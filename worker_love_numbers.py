"""
Defines a process/job for Love numbers computing.
"""

from os import utime

from transient_solid_earth import (
    SolidEarthFullNumericalModel,
    SolidEarthTimeDependentNumericalModel,
    WorkerInformation,
    intermediate_result_subpaths,
    is_elastic,
    parse_worker_information,
)

if __name__ == "__main__":
    worker_information: WorkerInformation = parse_worker_information(function_name="love_numbers")
    path = (
        intermediate_result_subpaths["love_numbers"]
        .joinpath(worker_information.model_id)
        .joinpath(str(worker_information.fixed_parameter))
    ).joinpath(
        "inf"
        if is_elastic(model_id=worker_information.model_id)
        else str(worker_information.variable_parameter)
    )

    # Checks whether the task has already been computed.
    if path.exists():

        utime(path=path.joinpath("imag.json"))

    else:

        # Loads the full numerical model, processes and saves.
        SolidEarthTimeDependentNumericalModel(
            solid_earth_full_numerical_model=SolidEarthFullNumericalModel(
                model_id=worker_information.model_id, load_numerical_model=True
            ),
            period=worker_information.variable_parameter,
            n=worker_information.fixed_parameter,
        ).integrate_y_i_systems()
