"""
Main call for Love numbers computing loop on rheologies.
"""

import shutil
from itertools import product

import numpy
from matplotlib.pyplot import scatter, semilogx, show

from transient_solid_earth import (
    TestModels,
    TestModelsRheology,
    adaptative_step_parallel_computing_loop,
    intermediate_result_subpaths,
    load_base_model,
    load_parameters,
    logs_subpaths,
)

if __name__ == "__main__":

    # Clears.
    if logs_subpaths["love_numbers"].exists():
        shutil.rmtree(logs_subpaths["test_models"])

    x_0_list = [0.25, 0.5, 2.0]
    alpha_list = [-1.0, 1.0]
    beta_list = [-1.0, 1.0]
    gamma_list = [-1.0, 1.0]

    parameters = load_parameters()
    rheologies = [
        {"alpha": alpha, "beta": beta, "gamma": gamma, "x_0": x_0}
        for alpha, beta, gamma, x_0 in product(alpha_list, beta_list, gamma_list, x_0_list)
    ]

    adaptative_step_parallel_computing_loop(
        rheologies=rheologies,
        model=TestModels,
        function_name="test_models",
        parameters=parameters,
    )

    for rheology in rheologies[:1]:
        result = load_base_model(
            name=TestModelsRheology(**rheology).model_id(),
            path=intermediate_result_subpaths["test_models"],
        )
        for f in numpy.array(object=result["values"]).reshape(len(result["x"]), -1).T:
            semilogx(result["x"], f)
            scatter(result["x"], f)
        show()
