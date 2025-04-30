"""
Worker to interpolate all test models on the same variable parameters.
"""

from transient_solid_earth import worker_interpolate

if __name__ == "__main__":
    worker_interpolate(function_name="test_models")
