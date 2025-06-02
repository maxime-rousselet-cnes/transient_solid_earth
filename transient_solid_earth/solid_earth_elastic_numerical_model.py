"""
Performs preprocessing steps for solid Earth numerical models: computes the g integral.
"""

from typing import Optional

from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel, ylim
from numpy import pi, sqrt
from scipy import interpolate

from .constants import G
from .parameters import SolidEarthParameters
from .paths import SolidEarthModelPart
from .rheological_formulas import g_0_computing, lambda_0_computing, mu_0_computing, p_0_computing
from .solid_earth_numerical_model import SolidEarthNumericalModel


class SolidEarthElasticNumericalModel(SolidEarthNumericalModel):
    """
    Includes the integration constants and all elasticity part description layers (unitless).
    """

    # Proper fields.
    density_unit: float
    speed_unit: float

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

    def find_fluid_layers(self) -> tuple[int, int]:
        """
        Counts the number of layers describing the Inner-Core and the outer core.
        All outer core layers should include "FLUID" in their name.
        """

        below_icb_layers, below_cmb_layers = 0, 0

        # Iterates on layer names from Geocenter.
        for layer_name in [model_layer.name for model_layer in self.model_layers]:
            if "FLUID" in layer_name:
                below_cmb_layers += 1
            elif below_icb_layers == below_cmb_layers:
                below_cmb_layers += 1
                below_icb_layers += 1
            else:
                return below_icb_layers, below_cmb_layers

        return below_icb_layers, below_cmb_layers

    def build_layer(self, g_0_inf: float, p_0_inf: float, i_layer: int) -> tuple[float, float]:
        """
        Builds a single layer. To be called from core to surface.
        """

        # Gets unitless variables.
        for variable_name, variable_unit in [
            ("v_s", self.speed_unit),
            ("v_p", self.speed_unit),
            ("rho_0", self.density_unit),
        ]:
            self.model_layers[i_layer].splines[variable_name] = (
                self.model_layers[i_layer].splines[variable_name][0],
                self.model_layers[i_layer].splines[variable_name][1] / variable_unit,
                self.model_layers[i_layer].splines[variable_name][2],
            )

        # Computes g_0, mu_0, lambda_0.
        x = self.model_layers[i_layer].x_profile(
            spline_number=self.solid_earth_parameters.numerical_parameters.spline_number
        )
        rho_0 = self.model_layers[i_layer].evaluate(x=x, variable="rho_0")
        mu_0 = mu_0_computing(
            rho_0=rho_0, v_s=self.model_layers[i_layer].evaluate(x=x, variable="v_s")
        )
        g_0 = g_0_computing(
            x=x,
            rho_0=rho_0,
            g_0_inf=g_0_inf,
            spline_number=self.solid_earth_parameters.numerical_parameters.spline_number,
        )
        p_0 = p_0_computing(
            x=x,
            rho_0=rho_0,
            g_0=g_0,
            p_0_inf=p_0_inf,
            spline_number=self.solid_earth_parameters.numerical_parameters.spline_number,
        )

        # Updates unitless variables.
        g_0_inf = g_0[-1]
        p_0_inf = p_0[-1]
        self.model_layers[i_layer].splines["g_0"] = interpolate.splrep(
            x=x, y=g_0, k=self.solid_earth_parameters.numerical_parameters.spline_degree
        )
        self.model_layers[i_layer].splines["mu_0"] = interpolate.splrep(
            x=x, y=mu_0, k=self.solid_earth_parameters.numerical_parameters.spline_degree
        )
        self.model_layers[i_layer].splines["drho_0_dx"] = interpolate.splder(
            tck=self.model_layers[i_layer].splines["rho_0"], n=1
        )
        self.model_layers[i_layer].splines["p_0"] = interpolate.splrep(
            x=x, y=p_0, k=self.solid_earth_parameters.numerical_parameters.spline_degree
        )
        self.model_layers[i_layer].splines["dp_0_dx"] = interpolate.splder(
            tck=self.model_layers[i_layer].splines["p_0"], n=1
        )
        self.model_layers[i_layer].splines["kappa_asymptotic"] = interpolate.splrep(
            x=x,
            y=self.model_layers[i_layer].evaluate(x=x, variable="drho_0_dx")
            / self.model_layers[i_layer].evaluate(x=x, variable="dp_0_dx"),
            k=self.solid_earth_parameters.numerical_parameters.spline_degree,
        )

        # Beware that asymptotic compressibility is not used in the core or where the density
        # gradient is negative with depth.
        if (
            self.solid_earth_parameters.model.structure_parameters.asymptotic_compressibility
            and i_layer >= self.solid_earth_parameters.model.structure_parameters.below_cmb_layers
            and (
                self.model_layers[i_layer].evaluate(x=x, variable="drho_0_dx")
                < -self.solid_earth_parameters.model.structure_parameters.drho_dx_epsilon
            ).all()
        ):

            # TODO: no plot.
            # Units for plot.
            radius_unit = self.solid_earth_parameters.model.radius_unit  # (m).
            elasticity_unit = (
                self.density_unit
                * self.solid_earth_parameters.model.radius_unit**2
                / self.period_unit**2
            )  # (Pa).

            # Uses 1 / kappa = lambda + 2/3 mu.
            self.model_layers[i_layer].splines["lambda_0"] = interpolate.splrep(
                x=x,
                y=self.model_layers[i_layer].evaluate(x=x, variable="kappa_asymptotic") ** -1
                - 2.0 / 3.0 * self.model_layers[i_layer].evaluate(x=x, variable="mu_0"),
                k=self.solid_earth_parameters.numerical_parameters.spline_degree,
            )

            plot(
                elasticity_unit
                * self.model_layers[i_layer].evaluate(x=x, variable="kappa_asymptotic") ** -1,
                radius_unit / 1000 * (1.0 - x),
                color="red",
                label="with transitions" if i_layer == 2 else "",
            )
            plot(
                elasticity_unit
                * (
                    lambda_0_computing(
                        rho_0=rho_0,
                        v_p=self.model_layers[i_layer].evaluate(x=x, variable="v_p"),
                        mu_0=mu_0,
                    )
                    + 2.0 / 3.0 * self.model_layers[i_layer].evaluate(x=x, variable="mu_0")
                ),
                radius_unit / 1000 * (1.0 - x),
                color="green",
                label="without transitions" if i_layer == 2 else "",
            )

        else:

            self.model_layers[i_layer].splines["lambda_0"] = interpolate.splrep(
                x=x,
                y=lambda_0_computing(
                    rho_0=rho_0,
                    v_p=self.model_layers[i_layer].evaluate(x=x, variable="v_p"),
                    mu_0=mu_0,
                ),
                k=self.solid_earth_parameters.numerical_parameters.spline_degree,
            )

    def build(
        self,
        solid_earth_model_part: SolidEarthModelPart,
        overwrite_model: bool = True,
        save: bool = True,
    ):
        """
        Builds description layers from model file parameters and preprocesses elasticity variables.
        """

        # Initializes description layers from model.
        super().build(
            solid_earth_model_part=solid_earth_model_part,
            save=False,
        )

        # Updates basic fields.
        if (
            self.solid_earth_parameters.model.structure_parameters.below_icb_layers is None
            or self.solid_earth_parameters.model.structure_parameters.below_cmb_layers is None
        ):
            below_icb_layers, below_cmb_layers = self.find_fluid_layers()
            if self.solid_earth_parameters.model.structure_parameters.below_cmb_layers is None:
                self.solid_earth_parameters.model.structure_parameters.below_cmb_layers = (
                    below_cmb_layers
                )
            if self.solid_earth_parameters.model.structure_parameters.below_icb_layers is None:
                self.solid_earth_parameters.model.structure_parameters.below_icb_layers = (
                    below_icb_layers
                )
        else:
            below_cmb_layers = (
                self.solid_earth_parameters.model.structure_parameters.below_cmb_layers
            )

        self.x_cmb = self.model_layers[below_cmb_layers].x_inf

        # Defines units.
        self.density_unit = self.model_layers[below_cmb_layers].evaluate(
            x=self.x_cmb, variable="rho_0"
        )  # := rho_0(CMB+) (kg.m^-3).
        self.period_unit = 1.0 / sqrt(self.density_unit * pi * G)  # (s).
        self.speed_unit = self.solid_earth_parameters.model.radius_unit / self.period_unit

        if self.solid_earth_parameters.model.structure_parameters.asymptotic_compressibility:
            figure(figsize=(8, 8))

        # Preprocesses unitless variables, including g_0, mu_0 and lambda_0.
        g_0_inf = 0.0  # g_0 at the bottom of the layer (unitless).
        p_0_inf = 0.0

        for i_layer, _ in enumerate(self.model_layers):

            self.build_layer(g_0_inf=g_0_inf, p_0_inf=p_0_inf, i_layer=i_layer)

        if self.solid_earth_parameters.model.structure_parameters.asymptotic_compressibility:
            ylim(700, 370)
            legend()
            title("Bulk Modulus")
            xlabel("(Pa)")
            ylabel("Depth (km)")
            show()

        # Eventually saves in (.JSON) file.
        if save:
            self.save(overwrite_model=overwrite_model)
