"""
Defines a process/job for test models computing.
"""

from transient_solid_earth import compute_test_models, parse_worker_information

if __name__ == "__main__":
    compute_test_models(worker_information=parse_worker_information(function_name="test_models"))
