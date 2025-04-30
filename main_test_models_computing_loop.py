"""
Main call for Love numbers computing loop on rheologies.
"""

import shutil
from itertools import product
from time import time

from transient_solid_earth import (
    adaptative_step_parallel_computing_loop,
    interpolate_on_grid_parallel_computing_loop,
    load_parameters,
    logs_subpaths,
)

if __name__ == "__main__":

    # Clears.
    if logs_subpaths["test_models"].exists():
        shutil.rmtree(logs_subpaths["test_models"])

    x_0_list = [1.0, 2.0]
    alpha_list = [-1.0, 1.0]
    beta_list = [-1.0, 1.0]
    gamma_list = [-1.0, 1.0]

    parameters = load_parameters()
    rheologies = [
        {"alpha": alpha, "beta": beta, "gamma": gamma}
        for alpha, beta, gamma in product(alpha_list, beta_list, gamma_list)
    ]

    t_0 = time()

    adaptative_step_parallel_computing_loop(
        rheologies=rheologies,
        function_name="test_models",
        fixed_parameter_list=x_0_list,
        parameters=parameters,
    )

    t_1 = time()
    print(t_1 - t_0)

    interpolate_on_grid_parallel_computing_loop(
        function_name="test_models", rheologies=rheologies, parameters=parameters
    )

    print(time() - t_1)
