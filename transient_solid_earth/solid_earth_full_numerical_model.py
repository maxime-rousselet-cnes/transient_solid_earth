"""
Contains both the elastic and the anelastic models informations, preprocessed.
The variables of this preprocessed class are still independent on frequency.
"""

from typing import Optional

import numpy
from scipy import interpolate

from .constants import (
    ASYMPTOTIC_MU_RATIO_DECIMALS,
    DEFAULT_MODELS,
    LAYER_DECIMALS,
    SECONDS_PER_YEAR,
)
from .generic_rheology_model import solid_earth_full_numerical_model_id_from_part_names
from .model_layer import ModelLayer
from .parameters import DEFAULT_SOLID_EARTH_PARAMETERS, SolidEarthParameters
from .paths import SolidEarthModelPart
from .rheological_formulas import find_tau_m, mu_k_computing
from .separators import LAYER_NAMES_SEPARATOR, UNUSED_MODEL_PART_DEFAULT_NAME
from .solid_earth_elastic_numerical_model import SolidEarthElasticNumericalModel
from .solid_earth_numerical_model import SolidEarthNumericalModel


def initialize_model_parts(
    rheology: dict[SolidEarthModelPart, Optional[str]], solid_earth_parameters: SolidEarthParameters
) -> dict[SolidEarthModelPart, SolidEarthNumericalModel | SolidEarthElasticNumericalModel]:
    """
    Initializes all model parts with their corresponding parameters.
    """

    model_parts: dict[
        SolidEarthModelPart, SolidEarthNumericalModel | SolidEarthElasticNumericalModel
    ] = {}

    for solid_earth_model_part, (_, part_name) in zip(SolidEarthModelPart, rheology.items()):

        # Initializes.
        if solid_earth_model_part == SolidEarthModelPart.ELASTICITY:
            model_parts[solid_earth_model_part] = SolidEarthElasticNumericalModel(
                solid_earth_parameters=solid_earth_parameters,
                model_id=rheology[SolidEarthModelPart.ELASTICITY],
                model_filename=rheology[SolidEarthModelPart.ELASTICITY],
            )
        else:
            model_parts[solid_earth_model_part] = SolidEarthNumericalModel(
                solid_earth_parameters=solid_earth_parameters,
                model_id=part_name.replace(
                    UNUSED_MODEL_PART_DEFAULT_NAME, DEFAULT_MODELS[solid_earth_model_part]
                ),
                model_filename=part_name.replace(
                    UNUSED_MODEL_PART_DEFAULT_NAME, DEFAULT_MODELS[solid_earth_model_part]
                ),
                solid_earth_model_part=solid_earth_model_part,
            )

        # Eventually loads the model part ...
        if (not solid_earth_parameters.options.overwrite_model) and model_parts[
            solid_earth_model_part
        ].get_path().joinpath(model_parts[solid_earth_model_part].model_id).is_file():

            model_parts[solid_earth_model_part].load()

        # ... or builds it.
        else:
            model_parts[solid_earth_model_part].build(
                solid_earth_model_part=solid_earth_model_part,
                overwrite_model=True,
                save=solid_earth_parameters.options.save,
            )

    return model_parts


class SolidEarthFullNumericalModel(SolidEarthNumericalModel):
    """
    Describes the integration constants and all layer models, including anelastic parameters.
    """

    # Proper fields.
    elasticity_unit: float
    viscosity_unit: float
    variable_values_per_layer: list[dict[str, numpy.ndarray]]

    elasticity_model: str  # Unitless.
    long_term_anelasticity_model: str  # With units.
    short_term_anelasticity_model: str  # With units.

    def load(self):
        """
        Loads a SolidEarthFullNumericalModel instance with correctly formatted fields.
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

    def save(self, overwrite_model: bool = True) -> None:
        """
        Replace carrefully infinite values by strings in proper fields for convenient (.JSON)
        writing, then save and replace back by infinite values.
        """

        # Replace infinite values by strings.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if numpy.inf in values:
                    self.variable_values_per_layer[i_layer][variable_name] = numpy.array(
                        object=["inf"] * len(values)
                    )

        # Saves to (.JSON) file.
        super().save(overwrite_model=overwrite_model)

        # Replace back strings by infinite values.
        for i_layer, variable_values in enumerate(self.variable_values_per_layer):
            for variable_name, values in variable_values.items():
                if "inf" in values:
                    self.variable_values_per_layer[i_layer][variable_name] = numpy.array(
                        object=[numpy.inf] * len(values)
                    )

    def __init__(
        self,
        solid_earth_parameters: SolidEarthParameters = DEFAULT_SOLID_EARTH_PARAMETERS,
        rheology: Optional[dict[SolidEarthModelPart, Optional[str]]] = None,
        model_id: Optional[str] = None,
        load_numerical_model: bool = False,
    ) -> None:

        # In case the model has to be loaded.
        if model_id:

            solid_earth_parameters.options.model_id = model_id

        else:

            # Manages defaults.
            rheology = (
                rheology
                if rheology
                else {
                    SolidEarthModelPart.ELASTICITY: None,
                    SolidEarthModelPart.LONG_TERM_ANELASTICITY: None,
                    SolidEarthModelPart.SHORT_TERM_ANELASTICITY: None,
                }
            )

            rheology[SolidEarthModelPart.ELASTICITY] = (
                rheology[SolidEarthModelPart.ELASTICITY]
                if rheology[SolidEarthModelPart.ELASTICITY]
                else DEFAULT_MODELS[SolidEarthModelPart.ELASTICITY]
            )
            rheology[SolidEarthModelPart.LONG_TERM_ANELASTICITY] = (
                rheology[SolidEarthModelPart.LONG_TERM_ANELASTICITY]
                if rheology[SolidEarthModelPart.LONG_TERM_ANELASTICITY]
                else DEFAULT_MODELS[SolidEarthModelPart.LONG_TERM_ANELASTICITY]
            )
            rheology[SolidEarthModelPart.SHORT_TERM_ANELASTICITY] = (
                rheology[SolidEarthModelPart.SHORT_TERM_ANELASTICITY]
                if rheology[SolidEarthModelPart.SHORT_TERM_ANELASTICITY]
                else DEFAULT_MODELS[SolidEarthModelPart.SHORT_TERM_ANELASTICITY]
            )

            solid_earth_parameters.model.options.use_long_term_anelasticity = (
                solid_earth_parameters.model.options.use_long_term_anelasticity
                and (
                    rheology[SolidEarthModelPart.LONG_TERM_ANELASTICITY]
                    != UNUSED_MODEL_PART_DEFAULT_NAME
                )
            )
            solid_earth_parameters.model.options.use_short_term_anelasticity = (
                solid_earth_parameters.model.options.use_short_term_anelasticity
                and (
                    rheology[SolidEarthModelPart.SHORT_TERM_ANELASTICITY]
                    != UNUSED_MODEL_PART_DEFAULT_NAME
                )
            )

            # Manages the option cases, including the elastic case.
            if (not solid_earth_parameters.model.options.use_long_term_anelasticity) and (
                not solid_earth_parameters.model.options.use_short_term_anelasticity
            ):
                solid_earth_parameters.options.model_id = (
                    solid_earth_full_numerical_model_id_from_part_names(
                        elasticity_name=rheology[SolidEarthModelPart.ELASTICITY],
                        long_term_anelasticity_name=UNUSED_MODEL_PART_DEFAULT_NAME,
                        short_term_anelasticity_name=UNUSED_MODEL_PART_DEFAULT_NAME,
                    )
                )
            elif not solid_earth_parameters.model.options.use_long_term_anelasticity:
                solid_earth_parameters.options.model_id = (
                    solid_earth_full_numerical_model_id_from_part_names(
                        elasticity_name=rheology[SolidEarthModelPart.ELASTICITY],
                        long_term_anelasticity_name=UNUSED_MODEL_PART_DEFAULT_NAME,
                        short_term_anelasticity_name=rheology[
                            SolidEarthModelPart.SHORT_TERM_ANELASTICITY
                        ],
                    )
                )
            elif not solid_earth_parameters.model.options.use_short_term_anelasticity:
                solid_earth_parameters.options.model_id = (
                    solid_earth_full_numerical_model_id_from_part_names(
                        elasticity_name=rheology[SolidEarthModelPart.ELASTICITY],
                        long_term_anelasticity_name=rheology[
                            SolidEarthModelPart.LONG_TERM_ANELASTICITY
                        ],
                        short_term_anelasticity_name=UNUSED_MODEL_PART_DEFAULT_NAME,
                    )
                )

        # Updates inherited fields.
        super().__init__(
            solid_earth_parameters=solid_earth_parameters,
            model_id=(
                solid_earth_parameters.options.model_id
                if not (solid_earth_parameters.options.model_id is None)
                else solid_earth_full_numerical_model_id_from_part_names(
                    elasticity_name=rheology[SolidEarthModelPart.ELASTICITY],
                    long_term_anelasticity_name=rheology[
                        SolidEarthModelPart.LONG_TERM_ANELASTICITY
                    ],
                    short_term_anelasticity_name=rheology[
                        SolidEarthModelPart.SHORT_TERM_ANELASTICITY
                    ],
                )
            ),
        )

        # Eventually loads already preprocessed full model...
        if load_numerical_model and self.get_path().joinpath(self.model_id + ".json").is_file():

            self.load()

        # ... or builds the model.
        else:

            # Initializes all 3 numerical model parts.
            model_parts = initialize_model_parts(
                rheology=rheology, solid_earth_parameters=solid_earth_parameters
            )

            # Updates fields from elasticity model.
            self.x_cmb = model_parts[SolidEarthModelPart.ELASTICITY].x_cmb

            # Updates new fields.
            self.period_unit = model_parts[SolidEarthModelPart.ELASTICITY].period_unit
            self.viscosity_unit = (
                model_parts[SolidEarthModelPart.ELASTICITY].density_unit
                * self.solid_earth_parameters.model.radius_unit**2
                / self.period_unit
            )
            self.elasticity_unit = self.viscosity_unit / self.period_unit

            # Builds common model layers.
            self.merge_models(model_parts=model_parts)

            # Computes explicit variable values for incoming lambda and mu complex computings.
            self.variable_values_per_layer = self.compute_variable_values()

            # Saves resulting full model in a (.JSON) file.
            if solid_earth_parameters.options.save:
                self.save(overwrite_model=solid_earth_parameters.options.overwrite_model)

    def merge_models(
        self,
        model_parts: dict[
            SolidEarthModelPart, SolidEarthNumericalModel | SolidEarthElasticNumericalModel
        ],
    ):
        """
        Merges all model parts with unitless variables only.
        """

        # Initializes with Core elastic and liquid layers.
        self.model_layers = model_parts[SolidEarthModelPart.ELASTICITY].model_layers[
            : self.solid_earth_parameters.model.structure_parameters.below_cmb_layers
        ]

        # Initializes accumulators.
        x_inf: float = self.x_cmb
        layer_indices_per_part: dict[SolidEarthModelPart, int] = {
            model_part: 0 for model_part in SolidEarthModelPart
        }
        layer_indices_per_part[SolidEarthModelPart.ELASTICITY] = (
            self.solid_earth_parameters.model.structure_parameters.below_cmb_layers
        )

        # Checks all layers from CMB to surface and merges their descrptions.
        while x_inf < 1.0:

            # Checks which layer ends first.
            x_sup_per_part: dict[SolidEarthModelPart, float] = {
                model_part: numpy.round(
                    a=model_parts[model_part]
                    .model_layers[layer_indices_per_part[model_part]]
                    .x_sup,
                    decimals=LAYER_DECIMALS,
                )
                for model_part in SolidEarthModelPart
            }
            x_sup: float = numpy.min([value for _, value in x_sup_per_part.items()])

            # Updates.
            self.model_layers += [
                self.merge_layers(
                    x_inf=x_inf,
                    x_sup=x_sup,
                    layers_per_part={
                        model_part: model_parts[model_part].model_layers[
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
        layers_per_part: dict[SolidEarthModelPart, ModelLayer],
    ) -> ModelLayer:
        """
        Merges elasticity, short-term anelasticity, and long-term anelasticity model layers with
        unitless variables only.
        """

        # Creates corresponding minimal length layer with elasticity variables.
        model_layer = ModelLayer(
            name=LAYER_NAMES_SEPARATOR.join((layer.name for _, layer in layers_per_part.items())),
            x_inf=x_inf,
            x_sup=x_sup,
            splines=layers_per_part[SolidEarthModelPart.ELASTICITY].splines.copy(),
        )

        # Adds other unitless variables.
        model_layer.splines["c"] = layers_per_part[
            SolidEarthModelPart.LONG_TERM_ANELASTICITY
        ].splines["c"]
        model_layer.splines["alpha"] = layers_per_part[
            SolidEarthModelPart.SHORT_TERM_ANELASTICITY
        ].splines["alpha"]
        model_layer.splines["asymptotic_mu_ratio"] = layers_per_part[
            SolidEarthModelPart.SHORT_TERM_ANELASTICITY
        ].splines["asymptotic_mu_ratio"]
        model_layer.splines["q_mu"] = layers_per_part[
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
            model_layer.splines[variable_name] = (
                splines[variable_name][0],
                splines[variable_name][1] / unit,  # Gets unitless variable.
                splines[variable_name][2],
            )

        return model_layer

    def compute_variable_values(
        self,
    ) -> list[dict[str, numpy.ndarray]]:
        """
        Computes explicit variable values for all layers.
        """

        variable_values_per_layer = []
        for i_layer, layer in enumerate(self.model_layers):
            variable_values_per_layer += [
                self.compute_variable_values_per_layer(i_layer=i_layer, layer=layer)
            ]
        return variable_values_per_layer

    def compute_variable_values_per_layer(
        self, i_layer: int, layer: ModelLayer
    ) -> dict[str, numpy.ndarray]:
        """
        Computes the needed explicit variable values for a single layer.
        """

        x = layer.x_profile(
            spline_number=self.solid_earth_parameters.numerical_parameters.spline_number,
        )

        # Variables needed for all layers.
        variable_values: dict[str, numpy.ndarray] = {
            "x": x,
            "mu_0": layer.evaluate(x=x, variable="mu_0"),
            "lambda_0": layer.evaluate(x=x, variable="lambda_0"),
        }

        if i_layer >= self.solid_earth_parameters.model.structure_parameters.below_cmb_layers:

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
                self.model_layers[i_layer].splines.update(
                    {
                        "tau_m": interpolate.splrep(
                            x=variable_values["x"], y=variable_values["tau_m"]
                        ),
                    }
                )
        return variable_values
