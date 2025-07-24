"""
Minimal package structure for transient_solid_earth.
"""

from .adaptative_step_parallel_computing import adaptative_step_parallel_computing_loop
from .compute_love_numbers_process import worker_compute_love_numbers
from .constants import BoundaryCondition, Direction
from .database import (
    generate_degrees_list,
    get_periods,
    load_base_model,
    load_complex_array,
    save_base_model,
)
from .elastic_load_models import ElasticLoadModel, load_elastic_load_model
from .formating import (
    load_barystatic_load_model,
    load_load_model_harmonic_component,
    make_grid,
    make_grid_from_unstacked,
    make_harmonics,
    make_unstacked_harmonics,
    mean_on_mask,
    stack_harmonics,
)
from .generate_elastic_load_model_parallel_computing import (
    generate_elastic_load_models_parallel_loop,
)
from .generate_elastic_load_models_process import worker_generate_elastic_load_models
from .interpolate_love_numbers_process import worker_interpolate_love_numbers
from .interpolate_parallel_computing import interpolate_parallel_computing_loop
from .leakage_correction import (
    _pool_apply_DDK_filter,
    collection_sh_data_from_grid,
    grid_from_collection_sh_data,
)
from .main_loop_functions import (
    anelastic_load_model_re_estimation_processing_loop,
    clear_path,
    get_period_interpolation_basis,
)
from .model_layer import ModelLayer
from .model_list_generation import create_all_model_variations
from .parameters import (
    DEFAULT_SOLID_EARTH_MODEL_OPTION_PARAMETERS,
    SolidEarthModelOptionParameters,
    SolidEarthVariableParameters,
    load_parameters,
)
from .paths import (
    SolidEarthModelPart,
    anelastic_load_models_path,
    data_path,
    elastic_load_models_path,
    harmonic_geoid_deformation_trends_path,
    harmonic_residual_trends_path,
    harmonic_vertical_displacement_trends_path,
    intermediate_result_subpaths,
    interpolated_love_numbers_path,
    loads_path,
    logs_subpaths,
    tables_path,
)
from .pole_tide import elastic_pole_tide_correction_back
from .separators import (
    LAYERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    VALUES_SEPARATOR,
)
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel
from .solid_earth_time_dependent_numerical_model import SolidEarthTimeDependentNumericalModel
from .trends import get_ocean_mean_trend
from .worker_parser import parse_worker_information

objects = [
    adaptative_step_parallel_computing_loop,
    worker_compute_love_numbers,
    BoundaryCondition,
    Direction,
    generate_degrees_list,
    get_periods,
    load_base_model,
    load_complex_array,
    save_base_model,
    ElasticLoadModel,
    load_elastic_load_model,
    load_barystatic_load_model,
    load_load_model_harmonic_component,
    make_grid,
    make_grid_from_unstacked,
    make_harmonics,
    make_unstacked_harmonics,
    mean_on_mask,
    stack_harmonics,
    generate_elastic_load_models_parallel_loop,
    worker_generate_elastic_load_models,
    worker_interpolate_love_numbers,
    interpolate_parallel_computing_loop,
    _pool_apply_DDK_filter,
    collection_sh_data_from_grid,
    grid_from_collection_sh_data,
    anelastic_load_model_re_estimation_processing_loop,
    clear_path,
    get_period_interpolation_basis,
    ModelLayer,
    create_all_model_variations,
    DEFAULT_SOLID_EARTH_MODEL_OPTION_PARAMETERS,
    SolidEarthModelOptionParameters,
    SolidEarthVariableParameters,
    load_parameters,
    SolidEarthModelPart,
    anelastic_load_models_path,
    data_path,
    elastic_load_models_path,
    harmonic_geoid_deformation_trends_path,
    harmonic_residual_trends_path,
    harmonic_vertical_displacement_trends_path,
    intermediate_result_subpaths,
    interpolated_love_numbers_path,
    loads_path,
    logs_subpaths,
    tables_path,
    elastic_pole_tide_correction_back,
    LAYERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    VALUES_SEPARATOR,
    SolidEarthFullNumericalModel,
    SolidEarthTimeDependentNumericalModel,
    get_ocean_mean_trend,
    parse_worker_information,
]
