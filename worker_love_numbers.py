"""
Defines a job for Love numbers computing.
"""

from transient_solid_earth import compute_love_numbers, parse_worker_information

if __name__ == "__main__":
    compute_love_numbers(worker_information=parse_worker_information(function_name="love_numbers"))
