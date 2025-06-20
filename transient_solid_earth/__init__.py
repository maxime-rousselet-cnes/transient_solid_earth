"""
Minimal package structure for transient_solid_earth.
"""

from .adaptative_step_parallel_computing import adaptative_step_parallel_computing_loop
from .compute_love_numbers_process import worker_compute_love_numbers
from .database import generate_degrees_list, load_complex_array
from .generate_elastic_load_model_parallel_computing import (
    generate_elastic_load_models_parallel_loop,
)
from .generate_elastic_load_models_process import worker_generate_elastic_load_models
from .interpolate_love_numbers_process import worker_interpolate_love_numbers
from .interpolate_parallel_computing import interpolate_parallel_computing_loop
from .main_loop_functions import (
    anelastic_load_model_re_estimation_processing_loop,
    clear_path,
    get_period_interpolation_basis,
)
from .model_list_generation import create_all_model_variations
from .parameters import load_parameters
from .paths import (
    SolidEarthModelPart,
    interpolated_love_numbers_path,
    loads_path,
    logs_subpaths,
    tables_path,
)
from .polar_tide import elastic_polar_tide_correction_back
from .worker_parser import parse_worker_information

objects = [
    SolidEarthModelPart,
    adaptative_step_parallel_computing_loop,
    anelastic_load_model_re_estimation_processing_loop,
    clear_path,
    create_all_model_variations,
    elastic_polar_tide_correction_back,
    generate_degrees_list,
    generate_elastic_load_models_parallel_loop,
    get_period_interpolation_basis,
    interpolate_parallel_computing_loop,
    interpolated_love_numbers_path,
    load_complex_array,
    load_parameters,
    loads_path,
    logs_subpaths,
    tables_path,
    worker_compute_love_numbers,
    worker_generate_elastic_load_models,
    worker_interpolate_love_numbers,
    parse_worker_information,
]
