"""
Defines a job for elastic load model generation.
"""

from transient_solid_earth import parse_worker_information, worker_generate_elastic_load_models

if __name__ == "__main__":
    worker_generate_elastic_load_models(
        worker_information=parse_worker_information(function_name="love_numbers")
    )
