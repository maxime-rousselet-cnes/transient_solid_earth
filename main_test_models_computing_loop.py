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
    if logs_subpaths["test_models"].exists():
        shutil.rmtree(logs_subpaths["test_models"])

    x_0_list = [0.25, 0.5, 2.0]
    alpha_list = [-1.0, 1.0]
    beta_list = [-1.0, 1.0]
    gamma_list = [-1.0, 1.0]

    parameters = load_parameters()
    rheologies = [
        {"alpha": alpha, "beta": beta, "gamma": gamma}
        for alpha, beta, gamma in product(alpha_list, beta_list, gamma_list)
    ]

    adaptative_step_parallel_computing_loop(
        rheologies=rheologies,
        model=TestModels,
        function_name="test_models",
        fixed_parameter_list=x_0_list,
        parameters=parameters,
    )

    for rheology in rheologies[:1]:
        for x_0 in x_0_list[:1]:
            result = load_base_model(
                name="real",
                path=intermediate_result_subpaths["test_models"]
                .joinpath(TestModelsRheology(**rheology).model_id())
                .joinpath(str(x_0)),
            )
            for f in (
                numpy.array(object=result["values"])
                .reshape(len(result["variable_parameter"]), -1)
                .T
            ):
                semilogx(result["variable_parameter"], f)
                scatter(result["variable_parameter"], f)
            show()
