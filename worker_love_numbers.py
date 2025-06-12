"""
Defines a job for Love numbers computing.
"""

from transient_solid_earth import parse_worker_information, worker_compute_love_numbers

if __name__ == "__main__":
    worker_compute_love_numbers(
        worker_information=parse_worker_information(function_name="love_numbers")
    )
