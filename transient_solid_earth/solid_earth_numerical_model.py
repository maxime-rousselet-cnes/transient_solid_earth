"""
Class for Solid Earth numerical model for preprocessing.
"""

from pathlib import Path
from typing import Optional

from numpy import array, inf, ndarray

from .constants import DEFAULT_MODELS
from .database import load_base_model, save_base_model
from .description_layer import DescriptionLayer
from .parameters import SolidEarthParameters
from .paths import (
    SolidEarthModelPart,
    solid_earth_full_numerical_models_path,
    solid_earth_numerical_models_path,
)
from .solid_earth_model_description import SolidEarthModelDescription


class SolidEarthNumericalModel:
    """
    Defines preprocessed model polynomial splines, layer by layer.
    """

    # Proper attributes.
    model_id: Optional[str]
    model_filename: Optional[str]
    model_part: Optional[SolidEarthModelPart]

    # Attribute to memorize parameters.
    solid_earth_parameters: SolidEarthParameters

    # Actual model's description.
    description_layers: list[DescriptionLayer]

    def __init__(
        self,
        solid_earth_parameters: SolidEarthParameters,
        model_id: Optional[str] = None,
        model_filename: Optional[str] = None,
        solid_earth_model_part: Optional[SolidEarthModelPart] = None,
    ) -> None:
        """
        Initializes either with None attributes or already built ones.
        """

        # Initializes IDs.
        self.model_filename = (
            DEFAULT_MODELS[solid_earth_model_part] if model_filename is None else model_filename
        )
        self.model_part = solid_earth_model_part
        self.model_id = model_id if not (model_id is None) else self.model_filename

        # Updates fields.
        self.solid_earth_parameters = solid_earth_parameters

        # Initializes description layers as empty.
        self.description_layers = []

    def build(
        self,
        solid_earth_model_part: SolidEarthModelPart,
        overwrite_description: bool = False,
        save: bool = True,
    ):
        """
        Builds description layers from model file parameters.
        """

        # Loads (.JSON) model's file.
        model = SolidEarthModelDescription(
            name=self.model_filename, solid_earth_model_part=solid_earth_model_part
        )

        # Gets layers descriptions.
        self.description_layers = model.build_description_layers_list(
            radius_unit=self.solid_earth_parameters.model.radius_unit,
            spline_number=self.solid_earth_parameters.numerical_parameters.spline_number,
            real_crust=self.solid_earth_parameters.model.real_crust,
        )

        # Eventually saves.
        if save:
            self.save(overwrite_description=overwrite_description)

    def load(self) -> None:
        """
        Loads a Description instance with correctly formatted fields.
        """

        # Gets raw description.
        description_dict: dict = load_base_model(name=self.model_id, path=self.get_path())
        self.model_part = (
            solid_earth_full_numerical_models_path
            if description_dict["model_part"] is None
            else SolidEarthModelPart(description_dict["model_part"])
        )

        # Formats attributes.
        for key, value in description_dict.items():
            setattr(self, key, value)

        # Formats layers.
        for i_layer, layer in enumerate(description_dict["description_layers"]):
            self.description_layers[i_layer] = DescriptionLayer(**layer)
            splines: dict[str, tuple] = layer["splines"]

            for variable_name, spline in splines.items():

                # Handles infinite values, as strings in files but as inf float for computing.
                if not isinstance(spline[0], list) and spline[0] == "inf":
                    self.description_layers[i_layer].splines[variable_name] = (
                        inf,
                        inf,
                        0,
                    )
                else:
                    spline = (
                        array(object=self.description_layers[i_layer].splines[variable_name][0]),
                        array(object=self.description_layers[i_layer].splines[variable_name][1]),
                        self.description_layers[i_layer].splines[variable_name][2],
                    )
                    # Formats every polynomial spline as a scipy polynomial spline.
                    self.description_layers[i_layer].splines[variable_name] = spline

    def save(self, overwrite_description: bool = True) -> None:
        """
        Saves the Description instance in a (.JSON) file.
        """

        path = self.get_path()

        if not (path.joinpath(self.model_id + ".json").is_file() and not overwrite_description):
            self_dict = self.__dict__
            self_dict["model_part"] = None if self.model_part is None else self.model_part.value
            layer: DescriptionLayer

            # Converts Infinite values to strings.
            for i_layer, layer in enumerate(self_dict["description_layers"]):
                splines: dict[str, tuple] = layer.splines
                for variable_name, spline in splines.items():
                    if not isinstance(spline[0], ndarray) and spline[0] == inf:
                        description_layer: DescriptionLayer = self_dict["description_layers"][
                            i_layer
                        ]
                        description_layer.splines[variable_name] = ("inf", "inf", 0)
                        self_dict["description_layers"][i_layer] = description_layer

            # Saves as basic type.
            save_base_model(
                obj=self_dict,
                name=self.model_id,
                path=path,
            )

            # Convert back if needed.
            for i_layer, layer in enumerate(self.description_layers):
                splines: dict[str, tuple] = layer.splines
                for variable_name, spline in splines.items():
                    if not isinstance(spline[0], ndarray) and spline[0] == "inf":
                        self.description_layers[i_layer].splines[variable_name] = (
                            inf,
                            inf,
                            0,
                        )

    def get_path(self) -> Path:
        """
        Returns directory path to save the description.
        """

        return (
            solid_earth_full_numerical_models_path
            if self.model_part is None
            else solid_earth_numerical_models_path[self.model_part]
        )
