"""
Short script for refactoring older version results.
"""

import json

from pandas import DataFrame, read_csv

# Load data
harmonic = read_csv("harmonic_load_signal_trends.csv")

with open("data/parameters.json", encoding="utf-8") as f:

    parameters = json.load(f)


# Build rheological_model_id
def build_rheological_id(row):
    """
    Eventually overwrites model name parts if unused.
    """

    elastic = row["elasticity_model"].replace("/", "____")
    long_term = (
        row["long_term_anelasticity_model"].replace("/", "____") if row["long-term"] else "unused"
    )
    short_term = (
        row["short_term_anelasticity_model"].replace("/", "____") if row["short-term"] else "unused"
    )
    return f"{elastic}_____{long_term}_____{short_term}"


# Prepare output columns (order from anelastic)
anelastic_cols = [
    "numerical_parameters:leakage_correction_iterations",
    "numerical_parameters:renormalize_recent_trend",
    "numerical_parameters:initial_past_trend_factor",
    "numerical_parameters:anti_Gibbs_effect_factor",
    "numerical_parameters:spline_time_years",
    "numerical_parameters:initial_plateau_date",
    "numerical_parameters:ewh_threshold",
    "numerical_parameters:ewh_threshold_past",
    "numerical_parameters:mean_ewh_threshold",
    "numerical_parameters:mean_ewh_threshold_past",
    "numerical_parameters:ddk_filter_level",
    "numerical_parameters:ocean_mask",
    "numerical_parameters:continents",
    "numerical_parameters:buffer_distance",
    "numerical_parameters:first_year_for_recent_trend",
    "numerical_parameters:last_year_for_recent_trend",
    "numerical_parameters:first_year_for_past_trend",
    "numerical_parameters:last_year_for_past_trend",
    "numerical_parameters:past_trend_error",
    "history:file",
    "history:start_date",
    "history:case",
    "history:pole:use",
    "history:pole:file",
    "history:pole:mean_pole_convention",
    "history:pole:case",
    "history:pole:pole_secular_term_trend_start_date",
    "history:pole:pole_secular_term_trend_end_date",
    "history:pole:ramp",
    "history:pole:filter_wobble",
    "history:pole:remove_mean_pole",
    "history:pole:wobble_filtering_kernel_length",
    "history:lia:use",
    "history:lia:end_date",
    "history:lia:time_years",
    "history:lia:amplitude_effect",
    "signature:opposite_load_on_continents",
    "signature:n_max",
    "signature:file",
    "options:compute_residuals",
    "options:invert_for_j_2",
    "options:save_options:all",
    "options:save_options:inversion_components",
    "rheological_model_id",
    "ocean_mean_trend_step_1",
    "ocean_mean_trend_step_2",
    "ocean_mean_trend_step_3",
    "ocean_mean_trend_step_4",
    "ocean_mean_trend_step_5",
    "geoid_deformation_ocean_mean_trend",
    "vertical_deformation_ocean_mean_trend",
    "residuals_ocean_mean_trend",
    "ID",
]

defaults = parameters["load_model"]["numerical_parameters"]
history = parameters["load_model"]["history"]
signature = parameters["load_model"]["signature"]
options = parameters["load_model"]["options"]


def get_save_option(key):
    """
    Accesses not straightforward information.
    """

    return options.get("save_options", {}).get("json_harmonics", {}).get(key, None)


# Build the output DataFrame
out = DataFrame()
out["numerical_parameters:leakage_correction_iterations"] = [
    defaults["leakage_correction_iterations"]
] * len(harmonic)
out["numerical_parameters:renormalize_recent_trend"] = [defaults["renormalize_recent_trend"]] * len(
    harmonic
)
out["numerical_parameters:initial_past_trend_factor"] = [
    defaults["initial_past_trend_factor"]
] * len(harmonic)
out["numerical_parameters:anti_Gibbs_effect_factor"] = [defaults["anti_Gibbs_effect_factor"]] * len(
    harmonic
)
out["numerical_parameters:spline_time_years"] = [defaults["spline_time_years"]] * len(harmonic)
out["numerical_parameters:initial_plateau_date"] = [defaults["initial_plateau_date"]] * len(
    harmonic
)
out["numerical_parameters:ewh_threshold"] = [defaults["ewh_threshold"]] * len(harmonic)
out["numerical_parameters:ewh_threshold_past"] = [defaults["ewh_threshold_past"]] * len(harmonic)
out["numerical_parameters:mean_ewh_threshold"] = [defaults["mean_ewh_threshold"]] * len(harmonic)
out["numerical_parameters:mean_ewh_threshold_past"] = [defaults["mean_ewh_threshold_past"]] * len(
    harmonic
)
out["numerical_parameters:ddk_filter_level"] = harmonic["ddk_filter_level"]
out["numerical_parameters:ocean_mask"] = [defaults["ocean_mask"]] * len(harmonic)
out["numerical_parameters:continents"] = [defaults["continents"]] * len(harmonic)
out["numerical_parameters:buffer_distance"] = harmonic["buffer_distance"]
out["numerical_parameters:first_year_for_recent_trend"] = [
    defaults["first_year_for_recent_trend"]
] * len(harmonic)
out["numerical_parameters:last_year_for_recent_trend"] = [
    defaults["last_year_for_recent_trend"]
] * len(harmonic)
out["numerical_parameters:first_year_for_past_trend"] = [
    defaults["first_year_for_past_trend"]
] * len(harmonic)
out["numerical_parameters:last_year_for_past_trend"] = [defaults["last_year_for_past_trend"]] * len(
    harmonic
)
out["numerical_parameters:past_trend_error"] = [defaults["past_trend_error"]] * len(harmonic)
out["history:file"] = [history["file"]] * len(harmonic)
out["history:start_date"] = [history["start_date"]] * len(harmonic)
out["history:case"] = harmonic["case"]
out["history:pole:use"] = [history["pole"]["use"]] * len(harmonic)
out["history:pole:file"] = [history["pole"]["file"]] * len(harmonic)
out["history:pole:mean_pole_convention"] = [history["pole"]["mean_pole_convention"]] * len(harmonic)
out["history:pole:case"] = harmonic["pole_case"]
out["history:pole:pole_secular_term_trend_start_date"] = harmonic[
    "pole_secular_term_trend_start_date"
]
out["history:pole:pole_secular_term_trend_end_date"] = harmonic["pole_secular_term_trend_end_date"]
out["history:pole:ramp"] = [history["pole"]["ramp"]] * len(harmonic)
out["history:pole:filter_wobble"] = harmonic["filter_wobble"]
out["history:pole:remove_mean_pole"] = harmonic["remove_mean_pole"]
out["history:pole:wobble_filtering_kernel_length"] = [
    history["pole"]["wobble_filtering_kernel_length"]
] * len(harmonic)
out["history:lia:use"] = harmonic["LIA"]
out["history:lia:end_date"] = [history["lia"]["end_date"]] * len(harmonic)
out["history:lia:time_years"] = [history["lia"]["time_years"]] * len(harmonic)
out["history:lia:amplitude_effect"] = [history["lia"]["amplitude_effect"]] * len(harmonic)
out["signature:opposite_load_on_continents"] = harmonic["opposite_load_on_continents"]
out["signature:n_max"] = [signature["n_max"]] * len(harmonic)
out["signature:file"] = (
    harmonic["load_spatial_behaviour_file"]
    if "load_spatial_behaviour_file" in harmonic
    else harmonic["signature:file"]
)
out["options:compute_residuals"] = [options["compute_residuals"]] * len(harmonic)
out["options:invert_for_j_2"] = [options["invert_for_j_2"]] * len(harmonic)
out["options:save_options:all"] = [get_save_option("all")] * len(harmonic)
out["options:save_options:inversion_components"] = [get_save_option("inversion_components")] * len(
    harmonic
)
out["rheological_model_id"] = harmonic.apply(build_rheological_id, axis=1)
out["ocean_mean_trend_step_1"] = harmonic["ocean_mean_step_1"]
out["ocean_mean_trend_step_2"] = harmonic["ocean_mean_step_2"]
out["ocean_mean_trend_step_3"] = harmonic["ocean_mean_step_3"]
out["ocean_mean_trend_step_4"] = harmonic["ocean_mean_step_4"]
out["ocean_mean_trend_step_5"] = harmonic["ocean_mean_step_5"]
out["geoid_deformation_ocean_mean_trend"] = (
    harmonic["ocean_mean_geoid_component"] if "ocean_mean_geoid_component" in harmonic else None
)
out["vertical_deformation_ocean_mean_trend"] = (
    harmonic["ocean_mean_radial_displacement_component"]
    if "ocean_mean_radial_displacement_component" in harmonic
    else None
)
out["residuals_ocean_mean_trend"] = None
out["ID"] = None

# Save to file
out.to_csv("transient_solid_earth/data/outputs/tables/backup.csv", index=False)
