"""
Worker to interpolate all test models on the same variable parameters.
"""

from transient_solid_earth import parse_worker_information, worker_interpolate

if __name__ == "__main__":
    worker_interpolate(
        worker_information=parse_worker_information(function_name="interpolate_test_models"),
        function_name="test_models",
    )
