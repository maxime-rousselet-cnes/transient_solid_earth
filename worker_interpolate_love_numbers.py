"""
Worker to interpolate all Love numbers on the same periods.
"""

from transient_solid_earth import parse_worker_information, worker_interpolate

if __name__ == "__main__":
    worker_interpolate(
        worker_information=parse_worker_information(function_name="interpolate_love_numbers"),
        function_name="love_numbers",
    )
