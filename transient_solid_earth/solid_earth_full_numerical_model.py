"""
Contins both the elastic and the anelastic models informations, preprocessed.
The variables of this preprocessed class are still independent on frequency.
"""

from typing import Optional

import numpy
from scipy import interpolate

from .constants import ASYMPTOTIC_MU_RATIO_DECIMALS, LAYER_DECIMALS, SECONDS_PER_YEAR
from .description_layer import DescriptionLayer
from .parameters import SolidEarthParameters, load_parameters
from .rheological_formulas import find_tau_m, mu_k_computing
from .separators import (
    LAYER_NAMES_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
)
from .solid_earth_elastic_numerical_model import SolidEarthElasticNumericalModel
from .solid_earth_model_description import SolidEarthModelPart
from .solid_earth_numerical_model import SolidEarthNumericalModel


def anelasticity_description_id_from_part_names(
    elasticity_name: str, long_term_anelasticity_name: str, short_term_anelasticity_name: str
) -> str:
    """
    Builds an id for an anelasticity description given the names of its components.
    """
    return SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(
        (
            part_name.replace("/", SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR)
            for part_name in (
                elasticity_name,
                long_term_anelasticity_name,
                short_term_anelasticity_name,
            )
        )
    )


class SolidEarthFullNumericalModel(SolidEarthNumericalModel):
    """
    Describes the integration constants and all layer model descriptions, including anelastic
    parameters.
    """

    # Proper fields.
    elasticity_unit: float
    viscosity_unit: float
    variable_values_per_layer: list[dict[str, numpy.ndarray]]

    #  Fields also present in the elasticity description, but may differ if it is loaded.
    x_cmb: float
    period_unit: float
    pi_times_g: float

    # Different parts descriptions.
    elasticity_model_name: str
    long_term_anelasticity_model_name: str
    short_term_anelasticity_model_name: str
    elasticity_description: str  # Unitless.
    anelasticicty_description: str  # With units.
    attenuation_description: str  # With units.

    def load(self):
        """
        Loads an Anelasticity Description instance with correctly formatted fields.
        """
        super().load()

        # Formats variable array values.
        layer_values_list: list[dict[str, list[float]]] = self.variable_values_per_layer
        self.variable_values_per_layer: list[dict[str, numpy.ndarray]] = [
            {
                variable_name: numpy.array(object=values, dtype=float)
                for variable_name, values in layer_values.items()
            }
            for layer_values in layer_values_list
        ]

    def save(self, overwrite_description: bool = True) -> None:
        """
        Replace carrefully infinite values by strings in proper fields for convenient (.JSON)
        writing, then save and replace
        back by infinite values.
        """

        # Replace infinite values by strings.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if numpy.inf in values:
                    self.variable_values_per_layer[i_layer][variable_name] = numpy.array(
                        object=["inf"] * len(values)
                    )

        # Saves to (.JSON) file.
        super().save(overwrite_description=overwrite_description)

        # Replace back strings by infinite values.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if "inf" in values:
                    self.variable_values_per_layer[i_layer][variable_name] = numpy.array(
                        object=[numpy.inf] * len(values)
                    )

    def __init__(
        self,
        solid_earth_parameters: SolidEarthParameters = load_parameters().solid_earth,
        elasticity_name: Optional[str] = None,
        long_term_anelasticity_name: Optional[str] = None,
        short_term_anelasticity_name: Optional[str] = None,
    ) -> None:

        # Updates inherited fields.
        super().__init__(
            solid_earth_parameters=solid_earth_parameters,
            model_id=(
                solid_earth_parameters.options.model_id
                if not (solid_earth_parameters.options.model_id is None)
                else anelasticity_description_id_from_part_names(
                    elasticity_name=elasticity_name,
                    long_term_anelasticity_name=long_term_anelasticity_name,
                    short_term_anelasticity_name=short_term_anelasticity_name,
                )
            ),
        )

        # Eventually loads already preprocessed anelasticity description...
        if (
            solid_earth_parameters.options.load_description
            and self.get_path().joinpath(self.model_id + ".json").is_file()
        ):

            self.load()

        # ... or builds the description.
        else:

            # Initializes all model description parts.
            description_parts: dict[
                SolidEarthModelPart, SolidEarthNumericalModel | SolidEarthElasticNumericalModel
            ] = {}
            part_names: dict[SolidEarthModelPart, str] = {
                SolidEarthModelPart.ELASTICITY: elasticity_name,
                SolidEarthModelPart.LONG_TERM_ANELASTICITY: long_term_anelasticity_name,
                SolidEarthModelPart.SHORT_TERM_ANELASTICITY: short_term_anelasticity_name,
            }

            for solid_earth_model_part, (_, part_name) in zip(
                SolidEarthModelPart, part_names.items()
            ):

                # Initializes.
                if solid_earth_model_part == SolidEarthModelPart.ELASTICITY:
                    description_parts[solid_earth_model_part] = SolidEarthElasticNumericalModel(
                        solid_earth_parameters=solid_earth_parameters,
                        model_id=elasticity_name,
                        model_filename=elasticity_name,
                    )
                else:
                    description_parts[solid_earth_model_part] = SolidEarthNumericalModel(
                        solid_earth_parameters=solid_earth_parameters,
                        model_id=part_name,
                        model_filename=part_name,
                        solid_earth_model_part=solid_earth_model_part,
                    )

                # Eventually loads the model description part ...
                if (
                    not solid_earth_parameters.options.overwrite_descriptions
                ) and description_parts[solid_earth_model_part].get_path().joinpath(
                    description_parts[solid_earth_model_part].model_id
                ).is_file():

                    description_parts[solid_earth_model_part].load()

                # ... or builds it.
                else:
                    description_parts[solid_earth_model_part].build(
                        model_part=solid_earth_model_part,
                        overwrite_description=True,
                        save=solid_earth_parameters.options.save,
                    )

            # Updates fields from elasticity description.
            self.x_cmb = description_parts[SolidEarthModelPart.ELASTICITY].x_cmb
            self.period_unit = description_parts[SolidEarthModelPart.ELASTICITY].period_unit
            self.pi_times_g = description_parts[SolidEarthModelPart.ELASTICITY].pi_times_g

            # Updates new fields.
            self.viscosity_unit = (
                description_parts[SolidEarthModelPart.ELASTICITY].density_unit
                * self.solid_earth_parameters.model.radius_unit**2
                / self.period_unit
            )
            self.elasticity_unit = self.viscosity_unit / self.period_unit

            # Builds common description layers.
            self.merge_descriptions(description_parts=description_parts)

            # Computes explicit variable values for incoming lambda and mu complex computings.
            self.variable_values_per_layer = self.compute_variable_values()

            # Saves resulting anelasticity description in a (.JSON) file.
            if solid_earth_parameters.options.save:
                self.save(
                    overwrite_description=solid_earth_parameters.options.overwrite_descriptions
                )

    def merge_descriptions(
        self,
        description_parts: dict[
            SolidEarthModelPart, SolidEarthNumericalModel | SolidEarthElasticNumericalModel
        ],
    ):
        """
        Merges all model description parts with unitless variables only.
        """

        # Initializes with Core elastic and liquid layers.
        self.description_layers = description_parts[
            SolidEarthModelPart.ELASTICITY
        ].description_layers[: self.solid_earth_parameters.model.below_cmb_layers]

        # Initializes accumulators.
        x_inf: float = self.x_cmb
        layer_indices_per_part: dict[SolidEarthModelPart, int] = {
            model_part: 0 for model_part in SolidEarthModelPart
        }
        layer_indices_per_part[SolidEarthModelPart.ELASTICITY] = (
            self.solid_earth_parameters.model.below_cmb_layers
        )

        # Checks all layers from CMB to surface and merges their descrptions.
        while x_inf < 1.0:

            # Checks which layer ends first.
            x_sup_per_part: dict[SolidEarthModelPart, float] = {
                model_part: numpy.round(
                    a=description_parts[model_part]
                    .description_layers[layer_indices_per_part[model_part]]
                    .x_sup,
                    decimals=LAYER_DECIMALS,
                )
                for model_part in SolidEarthModelPart
            }
            x_sup: float = numpy.min([value for _, value in x_sup_per_part.items()])

            # Updates.
            self.description_layers += [
                self.merge_layers(
                    x_inf=x_inf,
                    x_sup=x_sup,
                    layers_per_part={
                        model_part: description_parts[model_part].description_layers[
                            layer_indices_per_part[model_part]
                        ]
                        for model_part in SolidEarthModelPart
                    },
                )
            ]

            x_inf = x_sup
            for model_part in SolidEarthModelPart:
                if x_sup == x_sup_per_part[model_part]:
                    layer_indices_per_part[model_part] += 1

    def merge_layers(
        self,
        x_inf: float,
        x_sup: float,
        layers_per_part: dict[SolidEarthModelPart, DescriptionLayer],
    ) -> DescriptionLayer:
        """
        Merges elasticity, anelasticity, and attenuation description layers with unitless variables
        only.
        """

        # Creates corresponding minimal length layer with elasticity variables.
        description_layer = DescriptionLayer(
            name=LAYER_NAMES_SEPARATOR.join((layer.name for _, layer in layers_per_part.items())),
            x_inf=x_inf,
            x_sup=x_sup,
            splines=layers_per_part[SolidEarthModelPart.ELASTICITY].splines.copy(),
        )

        # Adds other unitless variables.
        description_layer.splines["c"] = layers_per_part[
            SolidEarthModelPart.LONG_TERM_ANELASTICITY
        ].splines["c"]
        description_layer.splines["alpha"] = layers_per_part[
            SolidEarthModelPart.SHORT_TERM_ANELASTICITY
        ].splines["alpha"]
        description_layer.splines["asymptotic_mu_ratio"] = layers_per_part[
            SolidEarthModelPart.SHORT_TERM_ANELASTICITY
        ].splines["asymptotic_mu_ratio"]
        description_layer.splines["q_mu"] = layers_per_part[
            SolidEarthModelPart.SHORT_TERM_ANELASTICITY
        ].splines["q_mu"]

        # Builds other unitless variables from variables with units.
        for variable_name, unit, splines in [
            (
                "eta_m",
                self.viscosity_unit,
                layers_per_part[SolidEarthModelPart.LONG_TERM_ANELASTICITY].splines,
            ),
            (
                "eta_k",
                self.viscosity_unit,
                layers_per_part[SolidEarthModelPart.LONG_TERM_ANELASTICITY].splines,
            ),
            (
                "mu_k1",
                self.elasticity_unit,
                layers_per_part[SolidEarthModelPart.LONG_TERM_ANELASTICITY].splines,
            ),
            (
                "omega_m",
                1.0 / self.period_unit,
                layers_per_part[SolidEarthModelPart.SHORT_TERM_ANELASTICITY].splines,
            ),
            (
                "tau_m",
                self.period_unit / SECONDS_PER_YEAR,
                layers_per_part[SolidEarthModelPart.SHORT_TERM_ANELASTICITY].splines,
            ),
        ]:
            description_layer.splines[variable_name] = (
                splines[variable_name][0],
                splines[variable_name][1] / unit,  # Gets unitless variable.
                splines[variable_name][2],
            )

        return description_layer

    def compute_variable_values(
        self,
    ) -> list[dict[str, numpy.ndarray]]:
        """
        Computes explicit variable values for all layers.
        """

        variable_values_per_layer = []
        for i_layer, layer in enumerate(self.description_layers):
            variable_values_per_layer += [
                self.compute_variable_values_per_layer(i_layer=i_layer, layer=layer)
            ]
        return variable_values_per_layer

    def compute_variable_values_per_layer(
        self, i_layer: int, layer: DescriptionLayer
    ) -> dict[str, numpy.ndarray]:
        """
        Computes the needed explicit variable values for a single layer.
        """

        x = layer.x_profile(
            spline_number=self.solid_earth_parameters.numerical_parameters.spline_number
        )

        # Variables needed for all layers.
        variable_values: dict[str, numpy.ndarray] = {
            "x": x,
            "mu_0": layer.evaluate(x=x, variable="mu_0"),
            "lambda_0": layer.evaluate(x=x, variable="lambda_0"),
        }

        if i_layer >= self.solid_earth_parameters.model.below_cmb_layers:

            # Variables needed above the Core-Mantle Boundary.
            variable_values.update(
                {
                    "eta_m": layer.evaluate(x=x, variable="eta_m"),
                    "mu_k": mu_k_computing(
                        mu_k1=layer.evaluate(x=x, variable="mu_k1"),
                        c=layer.evaluate(x=x, variable="c"),
                        mu_0=layer.evaluate(x=x, variable="mu_0"),
                    ),
                    "eta_k": layer.evaluate(x=x, variable="eta_k"),
                    "q_mu": layer.evaluate(x=x, variable="q_mu"),
                    "alpha": layer.evaluate(x=x, variable="alpha"),
                    "omega_m": layer.evaluate(x=x, variable="omega_m"),
                    "tau_m": layer.evaluate(x=x, variable="tau_m"),
                    "asymptotic_mu_ratio": layer.evaluate(x=x, variable="asymptotic_mu_ratio"),
                }
            )

            # Eventually finds tau_m profile that constrains
            # mu(omega -> inf) = asymptotic_ratio * mu_0:
            if numpy.round(
                a=1.0 - variable_values["asymptotic_mu_ratio"],
                decimals=ASYMPTOTIC_MU_RATIO_DECIMALS,
            ).any():
                for i_x, (omega_m, alpha, asymptotic_mu_ratio, q_mu) in enumerate(
                    zip(
                        variable_values["omega_m"],
                        variable_values["alpha"],
                        variable_values["asymptotic_mu_ratio"],
                        variable_values["q_mu"],
                    )
                ):
                    # Updates explicit variable.
                    variable_values["tau_m"][i_x] = find_tau_m(
                        omega_m=omega_m,
                        alpha=alpha,
                        asymptotic_mu_ratio=asymptotic_mu_ratio,
                        q_mu=q_mu,
                    )

                # Updates spline.
                self.description_layers[i_layer].splines.update(
                    {
                        "tau_m": interpolate.splrep(
                            x=variable_values["x"], y=variable_values["tau_m"]
                        ),
                    }
                )
        return variable_values
