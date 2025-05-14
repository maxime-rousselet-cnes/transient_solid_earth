"""
Defines a job for Green functions computing.
"""

from transient_solid_earth import compute_green_functions, parse_worker_information

if __name__ == "__main__":
    compute_green_functions(
        worker_information=parse_worker_information(function_name="green_functions")
    )
