"""
Main call for Love numbers computing loop on rheologies.
"""

import os
import shutil
from itertools import product
from time import time

from transient_solid_earth import (
    adaptative_step_parallel_computing_loop,
    interpolate_parallel_computing_loop,
    load_parameters,
    logs_subpaths,
)

if __name__ == "__main__":

    # Initializes.
    x_0_list = [1e-1, 0.25, 2.0]
    alpha_list = [-1.0, 10.0]
    beta_list = [-1.0, 100.0]
    gamma_list = [-1.0, 10.0]

    parameters = load_parameters()
    rheologies = [
        {"alpha": alpha, "beta": beta, "gamma": gamma}
        for alpha, beta, gamma in product(alpha_list, beta_list, gamma_list)
    ]

    # Clears.
    if logs_subpaths["test_models"].exists():
        shutil.rmtree(logs_subpaths["test_models"])

    t_0 = time()

    # Processes.
    adaptative_step_parallel_computing_loop(
        rheologies=rheologies,
        function_name="test_models",
        fixed_parameter_list=x_0_list,
        parameters=parameters,
    )

    if logs_subpaths["interpolate_test_models"].exists():
        shutil.rmtree(logs_subpaths["interpolate_test_models"])

    # Interpolates.
    interpolate_parallel_computing_loop(
        function_name="test_models", rheologies=rheologies, parameters=parameters
    )

    # Interpolates.
    interpolate_parallel_computing_loop(
        function_name="test_models",
        rheologies=rheologies,
        parameters=parameters,
        fixed_parameter_new_values=[1.0, 1.5, 2.0],
    )

    print(time() - t_0)

    os._exit(status=0)
