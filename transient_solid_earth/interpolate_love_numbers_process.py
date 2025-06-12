"""
Worker to interpolate Love numbers on the same periods.
"""

import warnings

import numpy
from scipy import interpolate

from .database import load_base_model, load_complex_array, save_complex_array
from .functions import generate_n_factor
from .paths import intermediate_result_subpaths, interpolated_love_numbers_path
from .worker_parser import WorkerInformation


def worker_interpolate_love_numbers(worker_information: WorkerInformation) -> None:
    """
    Interpolates the output of an adaptative step algorithm for multiple rheologies.
    Interpolates Love numbers on both the period axis and the degree axis.
    """

    save_path = interpolated_love_numbers_path(
        periods_id=worker_information.variable_parameter,
        rheological_model_id=worker_information.model_id,
    )

    # Check whether the task has already been computed.
    if save_path.exists():

        return

    # Gathers data.
    exponentiation_scale = worker_information.fixed_parameter
    degree_new_values = load_base_model(
        name="degrees", path=intermediate_result_subpaths["interpolate_love_numbers"]
    )
    period_new_values = load_base_model(
        name="periods",
        path=intermediate_result_subpaths["interpolate_love_numbers"].joinpath(
            worker_information.variable_parameter
        ),
    )
    load_path = intermediate_result_subpaths["love_numbers"].joinpath(worker_information.model_id)
    inputs: dict[int, numpy.ndarray] = {}
    degrees = []
    periods: dict[int, list[float]] = {}

    for degree_sub_path in load_path.iterdir():

        degree = int(float(degree_sub_path.name))
        degrees += [degree]
        periods[degree] = [
            float(period_sub_path.name)
            for period_sub_path in degree_sub_path.iterdir()
            if period_sub_path.is_dir()
        ]
        periods[degree].sort()
        inputs[degree] = load_complex_array(path=degree_sub_path)

    degrees.sort()
    interpolated_on_periods = []

    # Interpolates on periods.
    for degree in degrees:

        if [numpy.inf] == periods[degree]:
            # Handles the elastic case.
            interpolated_on_periods.append(inputs[degree])  # Length 1 along axis 1.
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                interpolated_on_periods.append(
                    interpolate.interp1d(
                        x=numpy.log(periods[degree]) / exponentiation_scale,
                        y=inputs[degree].real,
                        axis=0,
                        bounds_error=False,
                        fill_value=0.0,
                    )(x=numpy.log(numpy.abs(period_new_values)) / exponentiation_scale)
                    # To get hermitian.
                    + 1.0j
                    * numpy.sign(period_new_values)[:, numpy.newaxis, numpy.newaxis]
                    * interpolate.interp1d(
                        x=numpy.log(periods[degree]) / exponentiation_scale,
                        y=inputs[degree].imag,
                        axis=0,
                        bounds_error=False,
                        fill_value=0.0,
                    )(x=numpy.log(numpy.abs(period_new_values)) / exponentiation_scale)
                )

    # Interpolates on degrees.
    save_complex_array(
        obj=interpolate.interp1d(
            x=degrees,
            y=numpy.array(object=numpy.real(interpolated_on_periods))
            * generate_n_factor(degrees=degrees),
            axis=0,
        )(x=degree_new_values)
        / generate_n_factor(degrees=degree_new_values)
        + 1.0j
        * interpolate.interp1d(
            x=degrees,
            y=numpy.array(object=numpy.imag(interpolated_on_periods))
            * generate_n_factor(degrees=degrees),
            axis=0,
        )(x=degree_new_values)
        / generate_n_factor(degrees=degree_new_values),
        path=save_path,
    )
