"""
Minimal package structure for transient_solid_earth.
"""

from .adaptative_step_parallel_computing import adaptative_step_parallel_computing_loop
from .database import generate_degrees_list, load_base_model, save_base_model
from .interpolate_on_grid_parallel_computing import interpolate_on_grid_parallel_computing_loop
from .model_list_generation import create_all_model_variations
from .parameters import load_parameters
from .paths import SolidEarthModelPart, intermediate_result_subpaths, logs_subpaths
from .separators import is_elastic
from .solid_earth_full_numerical_model import SolidEarthFullNumericalModel
from .solid_earth_time_dependent_numerical_model import SolidEarthTimeDependentNumericalModel
from .test_models import TestModel, TestModelRheology
from .worker_interpolate import worker_interpolate
from .worker_parser import WorkerInformation, parse_worker_information

objects = [
    WorkerInformation,
    adaptative_step_parallel_computing_loop,
    generate_degrees_list,
    load_base_model,
    save_base_model,
    interpolate_on_grid_parallel_computing_loop,
    create_all_model_variations,
    load_parameters,
    SolidEarthModelPart,
    intermediate_result_subpaths,
    logs_subpaths,
    is_elastic,
    SolidEarthFullNumericalModel,
    SolidEarthTimeDependentNumericalModel,
    TestModel,
    worker_interpolate,
    TestModelRheology,
    parse_worker_information,
]
