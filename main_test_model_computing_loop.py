"""
Main call for Love numbers computing loop on rheologies.
"""

import shutil
from itertools import product

import numpy
from matplotlib.pyplot import scatter, semilogx, show

from transient_solid_earth.adaptative_step_parallel_computing import (
    adaptative_step_parallel_computing_loop,
)
from transient_solid_earth.database import load_base_model
from transient_solid_earth.parameters import DiscretizationParameters, load_parameters
from transient_solid_earth.paths import intermediate_result_subpaths, logs_subpaths
from transient_solid_earth.test_model import TestModel, TestModelRheology

if __name__ == "__main__":

    # Clears.
    if logs_subpaths["test_model"].exists():
        shutil.rmtree(logs_subpaths["test_model"])

    x_0_list = [0.25, 0.5, 2.0]
    alpha_list = [-1.0, 1.0]
    beta_list = [-1.0, 1.0]
    gamma_list = [-1.0, 1.0]

    parameters = load_parameters()
    parameters.discretization = DiscretizationParameters(
        x_min=1e-2, x_max=100.0, maximum_tolerance=5e-3
    )

    rheologies = [
        {"alpha": alpha, "beta": beta, "gamma": gamma, "x_0": x_0}
        for alpha, beta, gamma, x_0 in product(alpha_list, beta_list, gamma_list, x_0_list)
    ]

    adaptative_step_parallel_computing_loop(
        rheologies=rheologies,
        model=TestModel,
        function_name="test_model",
        fixed_parameter_list=[0.0],
        parameters=parameters,
    )

    for rheology in rheologies:
        result = load_base_model(
            name=TestModelRheology(**rheology).model_id(),
            path=intermediate_result_subpaths["test_model"],
        )
        for f in numpy.array(object=result["values"]).reshape(len(result["x"]), -1).T:
            semilogx(result["x"], f)
            scatter(result["x"], f)
        show()
