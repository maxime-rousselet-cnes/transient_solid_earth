"""
Defines all parameter classes.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .constants import DEFAULT_SPLINE_NUMBER, EARTH_RADIUS
from .database import load_base_model
from .paths import data_path


class SolidEarthModelOptionParameters(BaseModel):
    """
    Options for solid Earth model parameterization.
    """

    use_long_term_anelasticity: bool  # Whether to use long term anelasticity model or not.
    use_short_term_anelasticity: bool  # Whether to use short term anelasticity model or not.
    use_bounded_attenuation_functions: (
        bool  # Whether to use the bounded version of attenuation functions or not.
    )

    def string(self):
        """
        For figures.
        """

        return (
            "long-term and\nshort-term\nanelasticities"
            if self.use_long_term_anelasticity and self.use_short_term_anelasticity
            else (
                "long-term\nanelasticity"
                if self.use_long_term_anelasticity
                else (
                    "short-term\nanelasticity"
                    if self.use_short_term_anelasticity
                    else "pure elastic"
                )
            )
        )


class SolidEarthModelParameters(BaseModel):
    """
    Parameterizes the solid Earth model.
    """

    options: SolidEarthModelOptionParameters
    dynamic_term: bool  # Whether to use omega^2 terms or not.
    real_crust: (
        Optional[
            bool
        ]  # Whether to use 'real_crust' values or not. Usefull to easily switch from ocenanic
        # to continental crust parameters.
    )
    radius_unit: Optional[float]  # Length unit (m).

    # Number of layers under boundaries. If they are None: Automatic detection using elasticity
    # model layer names.
    # Number of layers under the Inner-Core Boundary.
    below_icb_layers: Optional[int]  # Should be >= 0.
    # Number of total layers under the Mantle-Core Boundary.
    below_cmb_layers: Optional[int]  # Should be >= below_ICB_layers.


class SolidEarthFrequencyDiscretizationParameters(BaseModel):
    """
    Describes the initial solid Earth model discretization on the frequency axis and its
    convergence criteria.
    """

    period_min_year: float  # High frequency limit (yr).
    period_max_year: float  # Low frequency limit (yr).
    n_frequency_0: int  # Minimal number of computed frequencies per degree.
    max_tol: float  # Maximal curvature criteria between orders 1 and 2.
    decimals: int  # Precision in log10(frequency / frequency_unit).


class SolidEarthDegreeDiscretizationParameters(BaseModel):
    """
    Describes the initial solid Earth model discretization on the degree axis.
    """

    steps: list[int]
    thresholds: list[int]


class SolidEarthNumericalParameters(BaseModel):
    """
    Describes the solid Earth model discretization and algorithm on the radial axis.
    """

    spline_number: int  # Should be >= max(2, 1 + polynomials degree).
    spline_degree: int  # Should be >= 0.
    n_max_for_sub_cmb_integration: (
        int  # Maximal degree for integration under the Core-Mantle Boundary.
    )
    minimal_radius: float  # r ~= 0 km exact definition (m).
    method: str  # Solver's numerical integration method.
    atol: float  # The solver keeps the local error estimates under atol + rtol * abs(yr).
    rtol: float  # See atol parameter description.
    t_eval: Optional[float]


class SolidEarthOptionParameters(BaseModel):
    """
    Parameters for optional computations
    """

    compute_Green: bool
    load_numerical_model: bool
    model_id: Optional[str]
    save: bool
    overwrite_model: bool


class SolidEarthParameters(BaseModel):
    """
    Defines all solid Earth algorithm parameters.
    """

    model: SolidEarthModelParameters
    frequency_discretization: SolidEarthFrequencyDiscretizationParameters
    degree_discretization: SolidEarthDegreeDiscretizationParameters
    numerical_parameters: SolidEarthNumericalParameters
    options: SolidEarthOptionParameters


class LoadSaveOptionParameters(BaseModel):
    """
    Defines load processing step save options for a given data structure.
    """

    all: bool  # Overwrites all other parameters with 'True' if set to 'True'.
    step_1: bool  # Initial model signal.
    step_2: bool  # Anelastic correcting polatr tide coefficients.
    step_3: (
        bool  # Anelastic load signal computed after frequencial filtering by Love number fractions.
    )
    step_4: bool  # Anelastic load signal computed after degree one inversion.
    step_5: bool  # Anelastic load signal computed after leakage corretion.
    # Initial.
    inversion_components: (
        bool  # Three remaining components of degree one inversion equation:
        # geoid height, radial displacement and residuals.
    )


class LoadModelNumericalParameters(BaseModel):
    """
    Defines the load algorithm parameters.
    """

    leakage_correction_iterations: int
    renormalize_recent_trend: (
        bool  # Wether to rescale recent period trends on GRACE ocean mean trend.
    )
    initial_past_trend_factor: float
    anti_Gibbs_effect_factor: int  # Integer, minimum equal to 1 (unitless).
    spline_time_years: int  # Time for the anti-symmetrization spline process in years.
    initial_plateau_time_years: (
        int  # Time of the zero-value plateau before the signal history (yr).
    )
    signal_threshold: float  # (mm/yr).
    signal_threshold_past: float  # (mm/yr).
    mean_signal_threshold: Optional[float]  # (mm/yr).
    mean_signal_threshold_past: Optional[float]  # (mm/yr).
    ddk_filter_level: int
    ocean_mask: str
    continents: str
    buffer_distance: int  # Buffer to coast (km).
    first_year_for_trend: int
    last_year_for_trend: int
    past_trend_error: float  # Maximal admitted error for past trend matching to data (mm/yr).


class LoadModelPoleParameters(BaseModel):
    """
    Defines all parameters for the pole time series.
    """

    use: bool  # Whether to performs Wahr (2015) recommended polar tide correction.
    file: str  # (.csv) file path relative to data/pole_data.
    mean_pole_convention: str  # IERS_2010, IERS_2018_update, etc...
    case: str  # Whether "lower", "mean" or "upper".
    pole_secular_term_trend_start_date: int
    pole_secular_term_trend_end_date: int
    ramp: bool
    filter_wobble: bool  # Whether to filter low-pass at the annual frequency.
    phi_constant: bool
    remove_pole_secular_trend: bool
    remove_mean_pole: bool
    wobble_filtering_kernel_length: int


class LoadModelHistoryParameters(BaseModel):
    """
    Defines the temporal evolution of the load model.
    """

    file: str  # (.csv) file path relative to data/GMSL_data.
    start_date: int  # Usually 1900 for Frederikse GMSL data.
    case: str  # Whether "lower", "mean" or "upper".


class LoadModelLIAParameters(BaseModel):
    """
    Defines the simplified LIA (little ice age) model.
    """

    use: bool  # Whethter to take LIA into account or not.
    end_date: int  # Usualy ~ 1400 (yr).
    time_years: int  # Usually ~ 100 (yr).
    amplitude_effect: float  # Usually ~ 0.25 (unitless).


class LoadModelSpatialSignatureParameters(BaseModel):
    """
    Defines the load spatial signature parameters.
    """

    opposite_load_on_continents: bool
    n_max: int
    file: str  # (.csv) file path relative to data.


class LoadModelOptionParameters(BaseModel):
    """
    Defines optional computations for the load algorithm.
    """

    compute_residuals: bool
    invert_for_J2: bool
    compute_displacements: bool


class LoadNumericalModelParameters(BaseModel):
    """
    Defines the load model and algorithm parameters.
    """

    numerical_parameters: LoadModelNumericalParameters
    pole: LoadModelPoleParameters
    history: LoadModelHistoryParameters
    lia: LoadModelLIAParameters
    signature: LoadModelSpatialSignatureParameters
    options: LoadModelOptionParameters


class LoadParameters(BaseModel):
    """
    Load model and algorithm parameters, including save options.
    """

    save_options: LoadSaveOptionParameters
    model: LoadNumericalModelParameters


class Parameters(BaseModel):
    """
    includes all transient solid Earth and load re-estimation algorithm parameters.
    """

    solid_earth: SolidEarthParameters
    load: LoadParameters


def load_parameters(
    name: str = "parameters", path: Path = data_path, fix_ddk: bool = False
) -> Parameters:
    """
    Routine that gets parameters from (.JSON) file.
    """

    parameters: Parameters = load_base_model(name=name, path=path, base_model_type=Parameters)
    if fix_ddk and ("MSSA" in parameters.load.model.signature.file):
        parameters.load.model.numerical_parameters.ddk_filter_level = 7
    if not "DDK" in parameters.load.model.signature.file:
        parameters.load.model.signature.file = (
            "DDK"
            + str(parameters.load.model.numerical_parameters.ddk_filter_level)
            + "/"
            + parameters.load.model.signature.file
        )
    parameters.solid_earth.model.radius_unit = (
        EARTH_RADIUS
        if parameters.solid_earth.model.radius_unit is None
        else parameters.solid_earth.model.radius_unit
    )
    parameters.solid_earth.model.real_crust = (
        False
        if parameters.solid_earth.model.real_crust is None
        else parameters.solid_earth.model.real_crust
    )
    parameters.solid_earth.numerical_parameters.spline_number = (
        DEFAULT_SPLINE_NUMBER
        if parameters.solid_earth.numerical_parameters.spline_number is None
        else parameters.solid_earth.numerical_parameters.spline_number
    )

    return parameters


# List of all possible boolean triplets.
SOLID_EARTH_MODEL_ALL_OPTION_PARAMETERS: list[SolidEarthModelOptionParameters] = [
    SolidEarthModelOptionParameters(
        use_long_term_anelasticity=True,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=True,
    ),
    SolidEarthModelOptionParameters(
        use_long_term_anelasticity=False,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=True,
    ),
    SolidEarthModelOptionParameters(
        use_long_term_anelasticity=True,
        use_short_term_anelasticity=False,
        use_bounded_attenuation_functions=False,
    ),
    SolidEarthModelOptionParameters(
        use_long_term_anelasticity=False,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=False,
    ),
    SolidEarthModelOptionParameters(
        use_long_term_anelasticity=True,
        use_short_term_anelasticity=True,
        use_bounded_attenuation_functions=False,
    ),
    SolidEarthModelOptionParameters(
        use_long_term_anelasticity=False,
        use_short_term_anelasticity=False,
        use_bounded_attenuation_functions=False,
    ),
]

# Elastic.
ELASTIC_SOLID_EARTH_MODEL_OPTION_PARAMETERS = SolidEarthModelOptionParameters(
    use_long_term_anelasticity=False,
    use_short_term_anelasticity=False,
    use_bounded_attenuation_functions=False,
)
