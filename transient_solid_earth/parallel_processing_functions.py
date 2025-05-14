"""
For generic parallel processing.
"""

from .compute_asymptotic_love_numbers_process import compute_asymptotic_love_numbers
from .compute_green_functions_process import compute_green_functions
from .compute_love_numbers_process import compute_love_numbers
from .compute_test_models_process import compute_test_models
from .interpolate_process import worker_interpolate
from .paths import sub_paths

functions = {
    "love_numbers": compute_love_numbers,
    "test_models": compute_test_models,
    "green_functions": compute_green_functions,
}
for function_name in sub_paths:
    functions["interpolate_" + function_name] = (
        lambda worker_information, fn=function_name: worker_interpolate(
            worker_information=worker_information, function_name=fn
        )
    )
functions["asymptotic_love_numbers"] = compute_asymptotic_love_numbers
