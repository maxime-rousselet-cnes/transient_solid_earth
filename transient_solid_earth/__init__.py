"""
Minimal package structure for transient_solid_earth.
"""

from .adaptative_step_parallel_computing import adaptative_step_parallel_computing_loop
from .asymptotic_love_numbers_parallel_computing import asymptotic_love_numbers_computing_loop
from .compute_asymptotic_love_numbers_process import compute_asymptotic_love_numbers
from .compute_green_functions_process import compute_green_functions
from .compute_love_numbers_process import compute_love_numbers
from .compute_test_models_process import compute_test_models
from .database import generate_degrees_list, load_base_model, save_base_model
from .interpolate_parallel_computing import interpolate_parallel_computing_loop
from .interpolate_process import worker_interpolate
from .model_list_generation import create_all_model_variations
from .parameters import asymptotic_degree_value, load_parameters
from .paths import SolidEarthModelPart, intermediate_result_subpaths, logs_subpaths
from .separators import is_elastic
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel
from .solid_earth_time_dependent_numerical_model import SolidEarthTimeDependentNumericalModel
from .test_models import TestModel, TestModelRheology
from .worker_parser import WorkerInformation, parse_worker_information

objects = [
    adaptative_step_parallel_computing_loop,
    asymptotic_love_numbers_computing_loop,
    compute_asymptotic_love_numbers,
    compute_green_functions,
    compute_love_numbers,
    compute_test_models,
    generate_degrees_list,
    load_base_model,
    save_base_model,
    interpolate_parallel_computing_loop,
    worker_interpolate,
    create_all_model_variations,
    asymptotic_degree_value,
    load_parameters,
    SolidEarthModelPart,
    intermediate_result_subpaths,
    logs_subpaths,
    is_elastic,
    SolidEarthFullNumericalModel,
    SolidEarthTimeDependentNumericalModel,
    TestModel,
    TestModelRheology,
    WorkerInformation,
    parse_worker_information,
]
