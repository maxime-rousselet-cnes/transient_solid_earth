"""
Performs preprocessing steps for solid Earth numerical models: computes the g integral.
"""

from typing import Optional

from numpy import pi, sqrt
from scipy import interpolate

from .constants import G
from .parameters import SolidEarthParameters
from .paths import SolidEarthModelPart
from .rheological_formulas import g_0_computing, lambda_0_computing, mu_0_computing
from .solid_earth_numerical_model import SolidEarthNumericalModel


class SolidEarthElasticNumericalModel(SolidEarthNumericalModel):
    """
    Includes the integration constants and all elasticity part description layers (unitless).
    """

    # Proper fields.
    x_cmb: float
    period_unit: float
    density_unit: float
    speed_unit: float
    pi_times_g: float

    # To memorize parameters.
    solid_earth_parameters: SolidEarthParameters

    def __init__(
        self,
        solid_earth_parameters: SolidEarthParameters,
        # Inherited parameters.
        model_id: Optional[str] = None,
        model_filename: Optional[str] = None,
    ) -> None:

        # Updates inherited fields.
        super().__init__(
            solid_earth_parameters=solid_earth_parameters,
            model_id=model_id,
            model_filename=model_filename,
            solid_earth_model_part=SolidEarthModelPart.ELASTICITY,
        )

        # Updates proper fields.
        self.solid_earth_model_parameters = solid_earth_parameters

    def find_fluid_layers(self) -> tuple[int, int]:
        """
        Counts the number of layers describing the Inner-Core and the Outer-Core.
        All Outer-Core layers should include "FLUID" in their name.
        """

        below_icb_layers, below_cmb_layers = 0, 0

        # Iterates on layer names from Geocenter.
        for layer_name in [description_layer.name for description_layer in self.description_layers]:
            if "FLUID" in layer_name:
                below_cmb_layers += 1
            elif below_icb_layers == below_cmb_layers:
                below_cmb_layers += 1
                below_icb_layers += 1
            else:
                return below_icb_layers, below_cmb_layers

        return below_icb_layers, below_cmb_layers

    def build(
        self,
        solid_earth_model_part: SolidEarthModelPart = SolidEarthModelPart.ELASTICITY,
        overwrite_description: bool = True,
        save: bool = True,
    ):
        """
        Builds description layers from model file parameters and preprocesses elasticity variables.
        """

        # Initializes description layers from model.
        super().build(solid_earth_model_part=solid_earth_model_part, save=False)

        # Updates basic fields.
        if (
            self.solid_earth_model_parameters.model.below_icb_layers is None
            or self.solid_earth_model_parameters.model.below_cmb_layers is None
        ):
            below_icb_layers, below_cmb_layers = self.find_fluid_layers()
            if self.solid_earth_model_parameters.model.below_cmb_layers is None:
                self.solid_earth_model_parameters.model.below_cmb_layers = below_cmb_layers
            if self.solid_earth_model_parameters.model.below_icb_layers is None:
                self.solid_earth_model_parameters.model.below_icb_layers = below_icb_layers

        self.x_cmb = self.description_layers[below_cmb_layers].x_inf

        # Defines units.
        self.density_unit = self.description_layers[below_cmb_layers].evaluate(
            x=self.x_cmb, variable="rho_0"
        )  # := rho_0(CMB+) (kg.m^-3).
        self.period_unit = 1.0 / sqrt(self.density_unit * pi * G)  # (s).
        self.speed_unit = self.solid_earth_parameters.model.radius_unit / self.period_unit
        self.pi_times_g = 1.0  # By definition of this unit system.

        # Preprocesses unitless variables, including g_0, mu_0 and lambda_0.
        g_0_inf = 0.0  # g_0 at the bottom of the layer (unitless).
        for i_layer, description_layer in enumerate(self.description_layers):

            # Gets unitless variables.
            for variable_name, variable_unit in [
                ("v_s", self.speed_unit),
                ("v_p", self.speed_unit),
                ("rho_0", self.density_unit),
            ]:
                self.description_layers[i_layer].splines[variable_name] = (
                    description_layer.splines[variable_name][0],
                    description_layer.splines[variable_name][1] / variable_unit,
                    description_layer.splines[variable_name][2],
                )

            # Computes g_0, mu_0, lambda_0.
            x = description_layer.x_profile(
                spline_number=self.solid_earth_parameters.numerical_parameters.spline_number
            )
            rho_0 = description_layer.evaluate(x=x, variable="rho_0")
            mu_0 = mu_0_computing(rho_0=rho_0, v_s=description_layer.evaluate(x=x, variable="v_s"))
            g_0 = g_0_computing(
                x=x,
                pi_times_g=self.pi_times_g,
                rho_0=rho_0,
                g_0_inf=g_0_inf,
                spline_number=self.solid_earth_model_parameters.numerical_parameters.spline_number,
            )

            # Updates unitless variables.
            g_0_inf = g_0[-1]
            self.description_layers[i_layer].splines["g_0"] = interpolate.splrep(
                x=x, y=g_0, k=self.solid_earth_model_parameters.numerical_parameters.spline_degree
            )
            self.description_layers[i_layer].splines["mu_0"] = interpolate.splrep(
                x=x, y=mu_0, k=self.solid_earth_model_parameters.numerical_parameters.spline_degree
            )
            self.description_layers[i_layer].splines["lambda_0"] = interpolate.splrep(
                x=x,
                y=lambda_0_computing(
                    rho_0=rho_0, v_p=description_layer.evaluate(x=x, variable="v_p"), mu_0=mu_0
                ),
                k=self.solid_earth_model_parameters.numerical_parameters.spline_degree,
            )

        # Eventually saves in (.JSON) file.
        if save:
            self.save(overwrite_description=overwrite_description)
