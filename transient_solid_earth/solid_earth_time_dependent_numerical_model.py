""" "
Class that manages the solid Earth Y system integration from core to surface and produces the Love
numbers. An instance of this class represents the full solid Earth model at one given frequency.
"""

import numpy
from scipy import interpolate

from .constants import INITIAL_Y_VECTOR
from .model_layer import ModelLayer
from .rheological_formulas import (
    b_computing,
    build_cutting_omegas,
    delta_mu_computing,
    f_attenuation_computing,
    m_prime_computing,
    mu_computing,
)
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel
from .solid_earth_numerical_model import SolidEarthNumericalModel


class SolidEarthTimeDependentNumericalModel(SolidEarthNumericalModel):
    """
    "Applies" full solid Earth model to some given frequency and some given degree.
    Describes the integration constants and all complex model layers at a given frequency.
    Handles elastic case for frequency = numpy.inf.
    Model layers variables include mu and lambda real and imaginary parts and the A matrix for the
    given degree.
    """

    n: int  # degree.

    def __init__(
        self,
        # Proper field parameters.
        solid_earth_full_numerical_model: SolidEarthFullNumericalModel,
        log_frequency: float,  # Base 10 logarithm of the unitless frequency.
        n: int,
    ) -> None:
        """
        Creates an Integration instance.
        """

        super().__init__(
            solid_earth_parameters=solid_earth_full_numerical_model.solid_earth_parameters,
        )

        # Updates proper attributes.
        self.n = n
        frequency = numpy.inf if log_frequency == numpy.inf else 10.0**log_frequency
        omega = numpy.inf if frequency == numpy.inf else 2 * numpy.pi * frequency
        omega_j = numpy.inf if omega == numpy.inf else omega * 1.0j

        # Updates attributes from the full numerical model.
        self.x_cmb = solid_earth_full_numerical_model.x_cmb
        options = self.solid_earth_parameters.model.options

        # Initializes the needed model layers.
        for i_layer, (variables, layer) in enumerate(
            zip(
                solid_earth_full_numerical_model.variable_values_per_layer,
                solid_earth_full_numerical_model.model_layers,
            )
        ):

            # First gets the needed real variable splines.
            model_layer = ModelLayer(
                name=layer.name,
                x_inf=layer.x_inf,
                x_sup=layer.x_sup,
                splines={
                    variable_name: layer.splines[variable_name]
                    for variable_name in ["g_0", "rho_0", "mu_0", "lambda_0", "Vs", "Vp"]
                    + (
                        ["eta_m"]
                        if i_layer >= self.solid_earth_parameters.model.below_cmb_layers
                        else []
                    )
                },
            )

            if (
                self.solid_earth_parameters.model.below_cmb_layers
                > i_layer
                >= self.solid_earth_parameters.model.below_icb_layers
            ):

                # Updates fluid matrix splines.
                model_layer.update_fluid_system_matrix(
                    n=n,
                    spline_number=self.solid_earth_parameters.numerical_parameters.spline_number,
                    x=variables["x"],
                )

            # Computes complex mu and lambda.
            else:

                # Default.
                variables["lambda"] = numpy.array(object=variables["lambda_0"], dtype=complex)
                variables["mu"] = numpy.array(object=variables["mu_0"], dtype=complex)

                if i_layer >= self.solid_earth_parameters.model.below_cmb_layers:

                    # Attenuation.
                    if options.use_short_term_anelasticity:

                        # Updates with attenuation functions f_r and f_i.
                        f = f_attenuation_computing(
                            variables=variables,
                            omega=omega,
                            frequency=frequency,
                            frequency_unit=1.0 / solid_earth_full_numerical_model.period_unit,
                            use_bounded_attenuation_functions=(
                                options.use_bounded_attenuation_functions
                            ),
                        )
                        model_layer.splines.update(
                            {
                                "f_r": interpolate.splrep(x=variables["x"], y=f.real),
                                "f_i": interpolate.splrep(x=variables["x"], y=f.imag),
                            }
                        )

                        # Adds delta mu, computed using f_r and f_i.
                        variables["mu"] = variables["mu_0"] + delta_mu_computing(
                            mu_0=variables["mu_0"],
                            q_mu=variables["q_mu"],
                            f=f,
                        )

                    # Long-term anelasticity.
                    if options.use_long_term_anelasticity:

                        # Complex cut frequency variables.
                        variables.update(build_cutting_omegas(variables=variables))
                        # Frequency filtering functions.
                        m_prime = m_prime_computing(
                            omega_cut_m=variables["omega_cut_m"], omega_j=omega_j
                        )
                        b = b_computing(
                            omega_cut_m=variables["omega_cut_m"],
                            omega_cut_k=variables["omega_cut_k"],
                            omega_cut_b=variables["omega_cut_b"],
                            omega_j=omega_j,
                        )
                        variables["mu"] = mu_computing(
                            mu_complex=variables["mu"], m_prime=m_prime, b=b
                        )

                    variables["lambda"] = variables["lambda_0"] - 2.0 / 3.0 * (
                        variables["mu"] - variables["mu_0"]
                    )

                # Updates solid matrix splines.
                model_layer.update_solid_system_matrix(
                    n=n,
                    omega=(
                        omega
                        if self.solid_earth_parameters.model.dynamic_term and omega != numpy.inf
                        else 0.0
                    ),
                    spline_number=self.solid_earth_parameters.numerical_parameters.spline_number,
                    variables=variables,
                )

            # Updates.
            self.model_layers += [model_layer]

    def integrate_y_i_systems(
        self,
    ) -> numpy.ndarray[complex]:
        """
        Integrates the unitless gravito-(an)elastic system from the Geocenter to the surface for the
        given degree, frequency and rheology.
        """

        numerical_parameters = self.solid_earth_parameters.numerical_parameters

        # The basis of solution to integrate.
        y_1 = INITIAL_Y_VECTOR[0, :].flatten()
        y_2 = INITIAL_Y_VECTOR[1, :].flatten()
        y_3 = INITIAL_Y_VECTOR[2, :].flatten()

        # I - Integrate from Geocenter to CMB.
        if self.n <= numerical_parameters.integration_parameters.n_max_for_sub_cmb_integration:

            # Integrate from Geocenter to CMB for low degrees only.
            self.model_layers[0].x_inf = (
                numerical_parameters.integration_parameters.minimal_radius
                / self.solid_earth_parameters.model.radius_unit
            )

            # Integrates in the Inner-Core.
            for n_layer in range(self.solid_earth_parameters.model.below_icb_layers):
                y_1, _ = self.model_layers[n_layer].integrate_y_i_system(
                    y_i=y_1, numerical_parameters=numerical_parameters
                )
                y_2, _ = self.model_layers[n_layer].integrate_y_i_system(
                    y_i=y_2, numerical_parameters=numerical_parameters
                )
                y_3, _ = self.model_layers[n_layer].integrate_y_i_system(
                    y_i=y_3, numerical_parameters=numerical_parameters
                )

            # ICB Boundary conditions.
            y = self.model_layers[
                int(self.solid_earth_parameters.model.below_icb_layers)
            ].solid_to_fluid(y_1=y_1, y_2=y_2, y_3=y_3)

            # Integrates in the Outer-Core.
            for n_layer in range(
                self.solid_earth_parameters.model.below_icb_layers,
                self.solid_earth_parameters.model.below_cmb_layers,
            ):
                y = self.model_layers[n_layer].integrate_y_i_system(
                    Y_i=y, numerical_parameters=numerical_parameters
                )

            # CMB Boundary conditions.
            y_1, y_2, y_3 = self.model_layers[
                self.solid_earth_parameters.model.below_cmb_layers - 1
            ].fluid_to_solid(yf_1=y)

        # Integrates from the CMB to the surface.
        for n_layer in range(
            self.solid_earth_parameters.model.below_cmb_layers, len(self.model_layers)
        ):
            y_1, _ = self.model_layers[n_layer].integrate_y_i_system(
                y_i=y_1, numerical_parameters=numerical_parameters
            )
            y_2, _ = self.model_layers[n_layer].integrate_y_i_system(
                y_i=y_2, numerical_parameters=numerical_parameters
            )
            y_3, _ = self.model_layers[n_layer].integrate_y_i_system(
                y_i=y_3, numerical_parameters=numerical_parameters
            )

        # h_load, l_load, k_load, h_shear, l_shear, k_shear, h_potential, l_potential, k_potential.
        return self.model_layers[-1].surface_solution(n=self.n, y_1_s=y_1, y_2_s=y_2, y_3_s=y_3)
