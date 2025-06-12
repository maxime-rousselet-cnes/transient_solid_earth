"""
For generic parallel processing.
"""

from .compute_love_numbers_process import worker_compute_love_numbers
from .generate_elastic_load_models_process import worker_generate_elastic_load_models
from .interpolate_love_numbers_process import worker_interpolate_love_numbers

functions = {
    "love_numbers": worker_compute_love_numbers,
    "interpolate_love_numbers": worker_interpolate_love_numbers,
    "generate_elastic_load_models": worker_generate_elastic_load_models,
}
