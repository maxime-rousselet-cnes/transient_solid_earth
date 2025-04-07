"""
Defines a convenient class to manipulate Love numbers.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy

from .database import complex_array_to_dict, complex_dict_to_array, load_base_model, save_base_model
from .parameters import SolidEarthParameters
from .paths import love_numbers_path


class Direction(Enum):
    """
    Love number directions.
    """

    RADIAL = 0
    TANGENTIAL = 1
    POTENTIAL = 2


class BoundaryCondition(Enum):
    """
    Love number boundary conditions.
    """

    LOAD = 0
    SHEAR = 1
    POETENTIAL = 2


class Values(dict[Direction, dict[BoundaryCondition, numpy.ndarray[complex]]]):
    """
    Love numbers are characterized by their direction and their boundary conditions.
    """


class LoveNumbers:
    """
    Describes Love numbers and how to manipulate them.
    """

    solid_earth_parameters: SolidEarthParameters
    values: Values
    axe_names: list[str]  # in the order of 'values' axes.
    axes: dict[str, numpy.ndarray[float]]

    def __init__(
        self,
        values: Optional[Values] = None,
        solid_earth_parameters: Optional[SolidEarthParameters] = None,
        axes: Optional[dict[str, numpy.ndarray[complex]]] = None,
        result_array: Optional[numpy.ndarray] = None,
    ) -> None:
        """
        Loves numbers are identified by the solid Earth model parameters that were used to processs
        them.
        """

        self.solid_earth_parameters = solid_earth_parameters
        self.values = values
        if not axes is None:
            self.axe_names = list(axes.keys())
        self.axes = axes
        if not result_array is None:
            self.update_values_from_array(result_array=result_array)

    def update_values_from_array(
        self,
        result_array: numpy.ndarray,
    ) -> None:
        """
        Converts p-dimmensionnal array to 'values' field, p >=3.
        The axis in position -1 should count 9 components corresponding to every combination of
        Direction and BoundaryCondition.
        """

        result_shape = result_array.shape[:-1]
        # Saves h_n, n*l_n and n*k_n.
        non_radial_factor = (
            numpy.array(object=self.axes["degrees"], dtype=int)
            if len(result_shape) == 1
            else numpy.expand_dims(a=self.axes["degrees"], axis=1)
        )
        radial_factor = numpy.ones(shape=result_shape)

        self.values = {
            direction: {
                boundary_condition: (
                    result_array[:, i_direction + 3 * i_boundary_condition]
                    if len(result_shape) == 1
                    else result_array[:, :, i_direction + 3 * i_boundary_condition]
                )
                * (radial_factor if direction == Direction.RADIAL else non_radial_factor)
                for i_boundary_condition, boundary_condition in enumerate(BoundaryCondition)
            }
            for i_direction, direction in enumerate(Direction)
        }

    def save(self, name: str, path: Path = love_numbers_path):
        """
        Saves the results in a (.JSON) file. Handles Enum classes. Converts complex values arrays
        to dictionary.
        """

        save_base_model(
            obj={
                "solid_earth_parameters": self.solid_earth_parameters,
                "values": {
                    key.value: {
                        sub_key.value: complex_array_to_dict(array=sub_values)
                        for sub_key, sub_values in values.items()
                    }
                    for key, values in self.values.items()
                },
                "axe_names": self.axe_names,
                "axes": {
                    axe_name: ({"real": ["inf"] if axe_values[0] == numpy.inf else axe_values})
                    for axe_name, axe_values in self.axes.items()
                },
            },
            name=name,
            path=path,
        )

    def load(self, name: str, path: Path = love_numbers_path) -> None:
        """
        Loads a Result structure from (.JSON) file.
        """

        loaded_content = load_base_model(
            name=name,
            path=path,
        )

        self.solid_earth_parameters = SolidEarthParameters(
            **loaded_content["solid_earth_parameters"]
        )
        _values: dict[str, dict[str, dict[str, list[float]]]] = loaded_content["values"]
        _axes: dict[str, dict[str, list[float]]] = loaded_content["axes"]
        self.axe_names = list(_axes.keys())

        self.values = Values(
            {
                Direction(int(direction)): {
                    (BoundaryCondition(int(boundary_condition))): complex_dict_to_array(
                        dictionary=sub_values
                    )
                    for boundary_condition, sub_values in values.items()
                }
                for direction, values in _values.items()
            }
        )

        self.axes = {
            axe_name: numpy.array(
                object=[numpy.inf] if "inf" in axe_values["real"] else axe_values["real"]
            )
            for axe_name, axe_values in _axes.items()
        }
