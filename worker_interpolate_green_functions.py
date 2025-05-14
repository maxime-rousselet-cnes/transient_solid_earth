"""
Worker to interpolate all Green funcions on the same angular distance to source.
"""

from transient_solid_earth import parse_worker_information, worker_interpolate

if __name__ == "__main__":
    worker_interpolate(
        worker_information=parse_worker_information(function_name="interpolate_green_functions"),
        function_name="green_functions",
    )
