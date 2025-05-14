"""
Defines all parameter classes.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from .constants import DEFAULT_SPLINE_NUMBER, EARTH_RADIUS
from .database import load_base_model
from .paths import SolidEarthModelPart, data_path


class SolidEarthModelOptionParameters(BaseModel):
    """
    Options for solid Earth model parameterization.
    """

    use_long_term_anelasticity: bool = True  # Whether to use long term anelasticity model or not.
    use_short_term_anelasticity: bool = True  # Whether to use short term anelasticity model or not.
    use_bounded_attenuation_functions: (
        bool  # Whether to use the bounded version of attenuation functions or not.
    ) = True

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


DEFAULT_SOLID_EARTH_MODEL_OPTION_PARAMETERS = SolidEarthModelOptionParameters()


class SolidEarthModelStructureParameters(BaseModel):
    """
    Defines the solid Earth model parameters usefull for Y_i system integration.
    """

    dynamic_term: bool = True  # Whether to use omega^2 terms or not.
    # Number of layers under boundaries. If they are None: Automatic detection using elasticity
    # model layer names.
    # Number of layers under the Inner-Core Boundary.
    below_icb_layers: Optional[int] = None  # Should be >= 0.
    # Number of total layers under the Mantle-Core Boundary.
    below_cmb_layers: Optional[int] = None  # Should be >= below_ICB_layers.


DEFAULT_SOLID_EARTH_MODEL_STRUCTURE_PARAMETERS = SolidEarthModelStructureParameters()


class SolidEarthModelParameters(BaseModel):
    """
    Parameterizes the solid Earth model.
    """

    options: SolidEarthModelOptionParameters = DEFAULT_SOLID_EARTH_MODEL_OPTION_PARAMETERS
    real_crust: (
        Optional[
            bool
        ]  # Whether to use 'real_crust' values or not. Usefull to easily switch from ocenanic
        # to continental crust parameters.
    ) = None
    radius_unit: Optional[float] = None  # Length unit (m).
    structure_parameters: SolidEarthModelStructureParameters = (
        DEFAULT_SOLID_EARTH_MODEL_STRUCTURE_PARAMETERS
    )

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __init__(
        self,
        options: SolidEarthModelOptionParameters = DEFAULT_SOLID_EARTH_MODEL_OPTION_PARAMETERS,
        real_crust: Optional[bool] = None,
        radius_unit: Optional[float] = None,
        structure_parameters: SolidEarthModelStructureParameters = (
            DEFAULT_SOLID_EARTH_MODEL_STRUCTURE_PARAMETERS
        ),
    ):

        super().__init__()

        self.options = (
            options
            if isinstance(options, SolidEarthModelOptionParameters)
            else SolidEarthModelOptionParameters(**options)
        )
        self.real_crust = False if real_crust is None else real_crust
        self.radius_unit = EARTH_RADIUS if radius_unit is None else radius_unit
        self.structure_parameters = (
            structure_parameters
            if isinstance(structure_parameters, SolidEarthModelStructureParameters)
            else SolidEarthModelStructureParameters(**structure_parameters)
        )


DEFAULT_SOLID_EARTH_MODEL_PARAMETERS = SolidEarthModelParameters()


class DiscretizationParameters(BaseModel):
    """
    Describes and initial discretization and its stop criterion.
    """

    value_min: float = 1.0e-2
    value_max: float = 1.0e5
    n_0: int = 3  # Minimal number of evaluations. Should be >=3.
    maximum_tolerance: float = 5.0e-3  # Curvature criterion.
    exponentiation_base: float = 10.0  # Because the discretization algorithm considers a log axis.
    rounding: int = 10
    min_step: float = 1.2


DEFAULT_LOVE_NUMBERS_DISCRETIZATION_PARAMETERS = DiscretizationParameters()
DEFAULT_TEST_MODELS_DISCRETIZATION_PARAMETERS = DiscretizationParameters(
    value_min=1.0e-2,
    value_max=1.0e2,
    n_0=3,
    maximum_tolerance=5e-3,
    exponentiation_base=10.0,
    rounding=10,
    min_step=1.2,
)
DEFAULT_GREEN_FUNCTIONS_DISCRETIZATION_PARAMETERS = DiscretizationParameters(
    value_min=1.0e-10,
    value_max=180.0,
    n_0=3,
    maximum_tolerance=5e-3,
    exponentiation_base=10.0,
    round=10,
    min_step=1.2,
)


class SolidEarthDegreeDiscretizationParameters(BaseModel):
    """
    Describes the initial solid Earth model discretization on the degree axis.
    """

    steps: list[int] = [1, 2, 5, 10, 50, 100, 200, 500, 1000, 10000]
    thresholds: list[int] = [1, 20, 40, 100, 200, 400, 1000, 2000, 6000, 10000, 100000]


DEFAULT_SOLID_EARTH_DEGREE_DISCRETIZATION_PARAMETERS = SolidEarthDegreeDiscretizationParameters()


class SolidEarthIntegrationNumericalParameters(BaseModel):
    """
    Describes the parameters necessary for the numerical integration of the Y_i system.
    """

    high_degrees_radius_sensibility: (
        float  # Integrates starting whenever x**n > high_degrees_radius_sensibility.
    ) = 1.0e-4
    minimal_radius: float = 1.0e3  # r ~= 0 km exact definition (m).
    atol: float = 1.0e-14  # The solver keeps the local error estimates under atol + rtol * abs(yr).
    rtol: float = 1.0e-7  # See atol parameter description.
    n_min_for_asymptotic_behavior: int = 5000


DEFAULT_SOLID_EARTH_INTEGRATION_NUMERICAL_PARAMETERS = SolidEarthIntegrationNumericalParameters()


class SolidEarthNumericalParameters(BaseModel):
    """
    Describes the solid Earth model discretization and algorithm on the radial axis.
    """

    spline_number: int = 100  # Should be >= max(2, 1 + polynomials degree).
    spline_degree: int = 1  # Should be >= 0.
    integration_parameters: SolidEarthIntegrationNumericalParameters = (
        DEFAULT_SOLID_EARTH_INTEGRATION_NUMERICAL_PARAMETERS
    )
    n_max_green: int = 10000

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __init__(
        self,
        spline_number: int = 100,
        spline_degree: int = 1,
        integration_parameters: SolidEarthIntegrationNumericalParameters = (
            DEFAULT_SOLID_EARTH_INTEGRATION_NUMERICAL_PARAMETERS
        ),
        n_max_green: int = 10000,
    ) -> None:

        super().__init__()

        self.spline_number = DEFAULT_SPLINE_NUMBER if spline_number is None else spline_number
        self.spline_degree = spline_degree
        self.integration_parameters = (
            integration_parameters
            if isinstance(integration_parameters, SolidEarthIntegrationNumericalParameters)
            else SolidEarthIntegrationNumericalParameters(**integration_parameters)
        )
        self.n_max_green = n_max_green


DEFAULT_SOLID_EARTH_NUMERICAL_PARAMETERS = SolidEarthNumericalParameters()


class SolidEarthOptionParameters(BaseModel):
    """
    Parameters for optional computations
    """

    compute_green: bool = True
    model_id: Optional[str] = None
    save: bool = True
    overwrite_model: bool = False


DEFAULT_SOLID_EARTH_OPTION_PARAMETERS = SolidEarthOptionParameters()


class ParallelComputingParameters(BaseModel):
    """
    Needed parameters to configure parallel computing tasks.
    """

    job_array_max_file_size: int = 1000
    max_concurrent_threads_factor: int = 4
    max_concurrent_processes_factor: int = 4


DEFAULT_PARALLEL_COMPUTING_PARAMETERS = ParallelComputingParameters()


class SolidEarthParameters(BaseModel):
    """
    Defines all solid Earth algorithm parameters.
    """

    model: SolidEarthModelParameters = DEFAULT_SOLID_EARTH_MODEL_PARAMETERS
    degree_discretization: SolidEarthDegreeDiscretizationParameters = (
        DEFAULT_SOLID_EARTH_DEGREE_DISCRETIZATION_PARAMETERS
    )
    numerical_parameters: SolidEarthNumericalParameters = DEFAULT_SOLID_EARTH_NUMERICAL_PARAMETERS
    options: SolidEarthOptionParameters = DEFAULT_SOLID_EARTH_OPTION_PARAMETERS


DEFAULT_SOLID_EARTH_PARAMETERS = SolidEarthParameters()


class LoadSaveOptionParameters(BaseModel):
    """
    Defines load processing step save options for a given data structure.
    """

    all: bool = True  # Overwrites all other parameters with 'True' if set to 'True'.
    step_1: bool = True  # Initial model signal.
    step_2: bool = True  # Anelastic correcting polatr tide coefficients.
    step_3: (
        bool  # Anelastic load signal computed after frequencial filtering by Love number fractions.
    ) = True
    step_4: bool = True  # Anelastic load signal computed after degree one inversion.
    step_5: bool = True  # Anelastic load signal computed after leakage corretion.
    # Initial.
    inversion_components: (
        bool  # Three remaining components of degree one inversion equation:
        # geoid height, radial displacement and residuals.
    ) = True


DEFAULT_LOAD_SAVE_OPTION_PARAMETERS = LoadSaveOptionParameters()


class LoadModelNumericalParameters(BaseModel):
    """
    Defines the load algorithm parameters.
    """

    leakage_correction_iterations: int = 1
    renormalize_recent_trend: (
        bool  # Wether to rescale recent period trends on GRACE ocean mean trend.
    ) = True
    initial_past_trend_factor: float = 1.22
    anti_Gibbs_effect_factor: int = 0  # Integer, minimum equal to 1 (unitless).
    spline_time_years: int = 50  # Time for the anti-symmetrization spline process in years.
    initial_plateau_time_years: (
        int  # Time of the zero-value plateau before the signal history (yr).
    ) = 2000
    signal_threshold: float = 12.0  # (mm/yr).
    signal_threshold_past: float = 6.0  # (mm/yr).
    mean_signal_threshold: Optional[float] = None  # (mm/yr).
    mean_signal_threshold_past: Optional[float] = None  # (mm/yr).
    ddk_filter_level: int = 5
    ocean_mask: str = "IMERG_land_sea_mask.nc"
    continents: str = "geopandas-continents.zip"
    buffer_distance: float = 300.0  # Buffer to coast (km).
    first_year_for_trend: int = 2003
    last_year_for_trend: int = 2022
    past_trend_error: float = (
        1e-3  # Maximal admitted error for past trend matching to data (mm/yr).
    )


DEFAULT_LOAD_MODEL_NUMERICAL_PARAMETERS = LoadModelNumericalParameters()


class LoadModelPoleParameters(BaseModel):
    """
    Defines all parameters for the pole time series.
    """

    use: bool = True  # Whether to performs Wahr (2015) recommended polar tide correction.
    file: str = "pole"  # (.csv) file path relative to data/pole_data.
    mean_pole_convention: str = "IERS_2018_update"  # IERS_2010, IERS_2018_update, etc...
    case: str = "mean"  # Whether "lower", "mean" or "upper".
    pole_secular_term_trend_start_date: int = 1900
    pole_secular_term_trend_end_date: int = 1978
    ramp: bool = False
    filter_wobble: bool = True  # Whether to filter low-pass at the annual frequency.
    phi_constant: bool = True
    remove_pole_secular_trend: bool = False
    remove_mean_pole: bool = True
    wobble_filtering_kernel_length: int = 50


DEFAULT_LOAD_MODEL_POLE_PARAMETERS = LoadModelPoleParameters()


class LoadModelLIAParameters(BaseModel):
    """
    Defines the simplified LIA (little ice age) model.
    """

    use: bool = False  # Whethter to take LIA into account or not.
    end_date: int = 1400  # Usualy ~ 1400 (yr).
    time_years: int = 100  # Usually ~ 100 (yr).
    amplitude_effect: float = 0.25  # Usually ~ 0.25 (unitless).


DEFAULT_LOAD_MODEL_LIA_PARAMETERS = LoadModelLIAParameters()


class LoadModelHistoryParameters(BaseModel):
    """
    Defines the temporal evolution of the load model.
    """

    file: str = (
        "Frederikse/global_basin_timeseries.csv"  # (.csv) file path relative to data/GMSL_data.
    )
    start_date: int = 1900  # Usually 1900 for Frederikse GMSL data.
    case: str = "mean"  # Whether "lower", "mean" or "upper".
    pole: LoadModelPoleParameters = DEFAULT_LOAD_MODEL_POLE_PARAMETERS
    lia: LoadModelLIAParameters = DEFAULT_LOAD_MODEL_LIA_PARAMETERS


DEFAULT_LOAD_MODEL_HISTORY_PARAMETERS = LoadModelHistoryParameters()


class LoadModelSpatialSignatureParameters(BaseModel):
    """
    Defines the load spatial signature parameters.
    """

    opposite_load_on_continents: bool = False
    n_max: int = 88
    # (.csv) file path relative to data.
    file: str = "DDK7/TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv"


DEFAULT_LOAD_MODEL_SPATIAL_SIGNATURE_PARAMETERS = LoadModelSpatialSignatureParameters()


class LoadModelOptionParameters(BaseModel):
    """
    Defines optional computations for the load algorithm.
    """

    compute_residuals: bool = False
    invert_for_J2: bool = False
    compute_displacements: bool = True


DEFAULT_LOAD_MODEL_OPTION_PARAMETERS = LoadModelOptionParameters()


class LoadNumericalModelParameters(BaseModel):
    """
    Defines the load model and algorithm parameters.
    """

    numerical_parameters: LoadModelNumericalParameters = DEFAULT_LOAD_MODEL_NUMERICAL_PARAMETERS
    history: LoadModelHistoryParameters = DEFAULT_LOAD_MODEL_HISTORY_PARAMETERS
    signature: LoadModelSpatialSignatureParameters = DEFAULT_LOAD_MODEL_SPATIAL_SIGNATURE_PARAMETERS
    options: LoadModelOptionParameters = DEFAULT_LOAD_MODEL_OPTION_PARAMETERS

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __init__(
        self,
        numerical_parameters: LoadModelNumericalParameters = (
            DEFAULT_LOAD_MODEL_NUMERICAL_PARAMETERS
        ),
        history: LoadModelHistoryParameters = DEFAULT_LOAD_MODEL_HISTORY_PARAMETERS,
        signature: LoadModelSpatialSignatureParameters = (
            DEFAULT_LOAD_MODEL_SPATIAL_SIGNATURE_PARAMETERS
        ),
        options: LoadModelOptionParameters = DEFAULT_LOAD_MODEL_OPTION_PARAMETERS,
    ) -> None:

        super().__init__()

        self.numerical_parameters = (
            numerical_parameters
            if isinstance(numerical_parameters, LoadModelNumericalParameters)
            else LoadModelNumericalParameters(**numerical_parameters)
        )
        self.history = (
            history
            if isinstance(history, LoadModelHistoryParameters)
            else LoadModelHistoryParameters(**history)
        )
        self.signature = (
            signature
            if isinstance(signature, LoadModelSpatialSignatureParameters)
            else LoadModelSpatialSignatureParameters(**signature)
        )
        self.options = (
            options
            if isinstance(options, LoadModelOptionParameters)
            else LoadModelOptionParameters(**options)
        )
        if "MSSA" in self.signature.file:
            self.numerical_parameters.ddk_filter_level = 7
        if not "DDK" in self.signature.file:
            self.signature.file = (
                "DDK" + str(self.numerical_parameters.ddk_filter_level) + "/" + self.signature.file
            )


DEFAULT_LOAD_NUMERICAL_MODEL_PARAMETERS = LoadNumericalModelParameters()


class LoadParameters(BaseModel):
    """
    Load model and algorithm parameters, including save options.
    """

    save_options: LoadSaveOptionParameters = DEFAULT_LOAD_SAVE_OPTION_PARAMETERS
    model: LoadNumericalModelParameters = DEFAULT_LOAD_NUMERICAL_MODEL_PARAMETERS


DEFAULT_LOAD_PARAMETERS = LoadParameters()


class SolidEarthVariableParameters(BaseModel):
    """
    Needed fields to describes the loop on rheological models.
    """

    model_names: dict[SolidEarthModelPart, list[str]] = {
        SolidEarthModelPart.ELASTICITY: ["PREM"],
        SolidEarthModelPart.LONG_TERM_ANELASTICITY: [
            "VM7",
            "VM5a",
            "Mao_Zhong",
            "Lau",
            "Lambeck_2017",
            "Caron",
        ],
        SolidEarthModelPart.SHORT_TERM_ANELASTICITY: [
            "Benjamin_Q_Resovsky",
            "Benjamin_Q_PAR3P",
            "Benjamin_Q_PREM",
            "Benjamin_Q_QL6",
            "Benjamin_Q_QM1",
        ],
    }
    rheological_parameters: dict[SolidEarthModelPart, dict[str, dict[str, list[list[float]]]]] = {
        SolidEarthModelPart.LONG_TERM_ANELASTICITY: {"eta_m": {"ASTHENOSPHERE": [[3e18], [3e19]]}},
        SolidEarthModelPart.SHORT_TERM_ANELASTICITY: {
            "asymptotic_mu_ratio": {"MANTLE": [[0.1], [0.2]]},
            "alpha": {"MANTLE": [[0.223], [0.297]]},
        },
    }


DEFAULT_SOLID_EARTH_VARIABLE_PARAMETERS = SolidEarthVariableParameters()


class Parameters(BaseModel):
    """
    Includes all transient solid Earth and load re-estimation algorithm parameters.
    """

    solid_earth: SolidEarthParameters = DEFAULT_SOLID_EARTH_PARAMETERS
    load: LoadParameters = DEFAULT_LOAD_PARAMETERS
    solid_earth_variabilities: SolidEarthVariableParameters = (
        DEFAULT_SOLID_EARTH_VARIABLE_PARAMETERS
    )
    load_model_variabilities: dict[str, Any] = {}
    parallel_computing: ParallelComputingParameters = DEFAULT_PARALLEL_COMPUTING_PARAMETERS
    discretization: dict[str, DiscretizationParameters] = {
        "love_numbers": DEFAULT_LOVE_NUMBERS_DISCRETIZATION_PARAMETERS,
        "test_models": DEFAULT_TEST_MODELS_DISCRETIZATION_PARAMETERS,
        "green_functions": DEFAULT_GREEN_FUNCTIONS_DISCRETIZATION_PARAMETERS,
    }


DEFAULT_PARAMETERS = Parameters()


def load_parameters(name: str = "parameters", path: Path = data_path) -> Parameters:
    """
    Gets parameters from (.JSON) file.
    """

    return Parameters.model_validate(load_base_model(name=name, path=path))


# List of all possible non-elastic models.
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
]

# Elastic.
ELASTIC_SOLID_EARTH_MODEL_OPTION_PARAMETERS = SolidEarthModelOptionParameters(
    use_long_term_anelasticity=False,
    use_short_term_anelasticity=False,
    use_bounded_attenuation_functions=False,
)


def asymptotic_degree_value(parameters: Parameters) -> int:
    """
    Returns the maximum degree for the asymptotic love numbers.
    """

    return min(
        parameters.solid_earth.numerical_parameters.n_max_green,
        parameters.solid_earth.degree_discretization.thresholds[-1],
    )
