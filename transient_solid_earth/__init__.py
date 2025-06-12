"""
Minimal package structure for transient_solid_earth.
"""

from .adaptative_step_parallel_computing import adaptative_step_parallel_computing_loop
from .compute_love_numbers_process import worker_compute_love_numbers
from .database import generate_degrees_list, load_base_model, load_complex_array
from .functions import trend
from .generate_elastic_load_model_parallel_computing import (
    generate_elastic_load_models_parallel_loop,
)
from .generate_elastic_load_models_process import worker_generate_elastic_load_models
from .interpolate_love_numbers_process import worker_interpolate_love_numbers
from .interpolate_parallel_computing import interpolate_parallel_computing_loop
from .load_signal_model import ElasticLoadModel, load_elastic_load_model
from .main_loop_functions import (
    anelastic_load_model_re_estimation_processing_loop,
    get_period_interpolation_basis,
)
from .model_list_generation import create_all_model_variations
from .parameters import LoadModelParameters, Parameters, load_parameters
from .paths import (
    SolidEarthModelPart,
    elastic_load_model_parameters_subpath,
    elastic_load_models_path,
    interpolated_love_numbers_path,
    logs_subpaths,
    tables_path,
)
from .polar_tide import elastic_polar_tide_correction_back
from .worker_parser import parse_worker_information

objects = [
    adaptative_step_parallel_computing_loop,
    worker_compute_love_numbers,
    generate_degrees_list,
    load_base_model,
    load_complex_array,
    trend,
    generate_elastic_load_models_parallel_loop,
    worker_generate_elastic_load_models,
    interpolate_parallel_computing_loop,
    ElasticLoadModel,
    load_elastic_load_model,
    anelastic_load_model_re_estimation_processing_loop,
    get_period_interpolation_basis,
    worker_interpolate_love_numbers,
    create_all_model_variations,
    LoadModelParameters,
    Parameters,
    load_parameters,
    SolidEarthModelPart,
    elastic_load_model_parameters_subpath,
    elastic_load_models_path,
    interpolated_love_numbers_path,
    logs_subpaths,
    tables_path,
    elastic_polar_tide_correction_back,
    parse_worker_information,
]
