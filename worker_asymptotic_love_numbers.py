"""
Worker to get asymptotic values of Love numbers for Green function computing.
"""

from transient_solid_earth import compute_asymptotic_love_numbers, parse_worker_information

if __name__ == "__main__":
    compute_asymptotic_love_numbers(
        worker_information=parse_worker_information(function_name="asymptotic_love_numbers"),
    )
