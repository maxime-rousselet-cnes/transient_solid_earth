"""
Worker to interpolate all Love numbers on the same periods.
"""

from transient_solid_earth import worker_interpolate

if __name__ == "__main__":
    worker_interpolate(function_name="love_numbers")
