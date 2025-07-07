"""
Formats data for figrues of the main paper.
"""

from transient_solid_earth import (
    ModelLayer,
    SolidEarthFullNumericalModel,
    SolidEarthModelPart,
    SolidEarthTimeDependentNumericalModel,
    load_barystatic_load_model,
    load_elastic_load_model,
    load_parameters,
    save_base_model,
)

from .figures_formater_utils import (
    DEFAULT_FILTER_UNWANTED_VALUES,
    DEFAULT_FILTER_WANTED_VALUES,
    LONG_TERM_MAP,
    REFERENCE_ELASTIC_LOAD_MODEL_ID,
    SHORT_TERM_MAP,
    figures_path,
    preprocess_dataframe,
    preprocess_grid,
    preprocess_variabilities,
    replace_and_save_to_csv,
)


def preprocess_figure_1() -> None:
    """
    Barystatic curb and extremas, and MSSA trends.
    """

    parameters = load_parameters()
    parameters.load_model.history.case = "mean"
    dates, mean_curb = load_barystatic_load_model(
        load_model_parameters=parameters.load_model, zero_at_origin=False
    )
    parameters.load_model.history.case = "lower"
    dates, lower_bound = load_barystatic_load_model(
        load_model_parameters=parameters.load_model, zero_at_origin=False
    )
    parameters.load_model.history.case = "upper"
    dates, upper_bound = load_barystatic_load_model(
        load_model_parameters=parameters.load_model, zero_at_origin=False
    )

    lower_bound -= mean_curb[0]
    upper_bound -= mean_curb[0]
    mean_curb -= mean_curb[0]

    elastic_load_model = load_elastic_load_model(model_id=REFERENCE_ELASTIC_LOAD_MODEL_ID)  # MSSA.

    elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask = None
    latitudes, longitudes, mask, grid, _ = preprocess_grid(load_model=elastic_load_model)

    save_base_model(
        obj={
            "dates": dates,
            "lower_bound": upper_bound,
            "mean_curb": mean_curb,
            "upper_bound": upper_bound,
            "latitudes": latitudes,
            "longitudes": longitudes,
            "mask": mask,
            "grid": grid,
        },
        name="figure_1",
        path=figures_path,
    )


def preprocess_figure_2() -> None:
    """
    mu_0, Q_mu and viscosity profiles.
    """

    parameters = load_parameters()

    eta_m = {}

    for long_term_model_name in LONG_TERM_MAP:

        solid_earth_full_numerical_model = SolidEarthFullNumericalModel(
            solid_earth_parameters=parameters.solid_earth,
            rheology={
                SolidEarthModelPart.ELASTICITY: "PREM",
                SolidEarthModelPart.LONG_TERM_ANELASTICITY: long_term_model_name,
                SolidEarthModelPart.SHORT_TERM_ANELASTICITY: None,
            },
        )
        eta_m[long_term_model_name] = {"depth": [], "value": []}
        solid_earth_parameters = solid_earth_full_numerical_model.solid_earth_parameters
        model_layer: ModelLayer

        for model_layer in solid_earth_full_numerical_model.model_layers[
            solid_earth_parameters.model.structure_parameters.below_cmb_layers :
        ]:

            x = model_layer.x_profile(
                spline_number=solid_earth_parameters.numerical_parameters.spline_number,
            )

            eta_m[long_term_model_name]["value"] += (
                list(
                    model_layer.evaluate(x=x, variable="eta_m")
                    * solid_earth_full_numerical_model.viscosity_unit
                ),
            )
            eta_m[long_term_model_name]["depth"] += list(
                (1.0 - x) * solid_earth_parameters.model.radius_unit / 1e3
            )

    q_mu = {}

    for short_term_model_name in SHORT_TERM_MAP:

        solid_earth_full_numerical_model = SolidEarthFullNumericalModel(
            solid_earth_parameters=parameters.solid_earth,
            rheology={
                SolidEarthModelPart.ELASTICITY: "PREM",
                SolidEarthModelPart.LONG_TERM_ANELASTICITY: None,
                SolidEarthModelPart.SHORT_TERM_ANELASTICITY: short_term_model_name,
            },
        )
        q_mu[short_term_model_name] = {"depth": [], "value": []}
        solid_earth_parameters = solid_earth_full_numerical_model.solid_earth_parameters
        model_layer: ModelLayer

        for model_layer in solid_earth_full_numerical_model.model_layers[
            solid_earth_parameters.model.structure_parameters.below_cmb_layers :
        ]:

            x = model_layer.x_profile(
                spline_number=solid_earth_parameters.numerical_parameters.spline_number,
            )

            q_mu[short_term_model_name]["value"] += (
                list(
                    model_layer.evaluate(x=x, variable="q_mu")
                    * solid_earth_full_numerical_model.viscosity_unit
                ),
            )
            q_mu[short_term_model_name]["depth"] += list(
                (1.0 - x) * solid_earth_parameters.model.radius_unit / 1e3
            )

    mu = {"depth": [], "value": []}

    model_layer: ModelLayer
    for model_layer in solid_earth_full_numerical_model.model_layers[
        solid_earth_parameters.model.structure_parameters.below_cmb_layers :
    ]:

        x = model_layer.x_profile(
            spline_number=solid_earth_parameters.numerical_parameters.spline_number,
        )

        mu["value"] += (
            list(
                model_layer.evaluate(x=x, variable="mu_0")
                * solid_earth_full_numerical_model.viscosity_unit
            ),
        )
        mu["depth"] += list((1.0 - x) * solid_earth_parameters.model.radius_unit / 1e3)

    save_base_model(
        obj={
            "mu": mu,
            "q_mu": q_mu,
            "eta_m": eta_m,
            "asthenospheric_viscosity": 3e19,
        },
        name="figure_2",
        path=figures_path,
    )


def preprocess_figure_3() -> None:
    """
    Real and imaginary parts of mu_0/mu with respect to depth for 10, 100 and 1000 years.
    """

    parameters = load_parameters()
    solid_earth_full_numerical_model = SolidEarthFullNumericalModel(
        solid_earth_parameters=parameters.solid_earth,
        rheology={
            SolidEarthModelPart.ELASTICITY: "PREM",
            SolidEarthModelPart.LONG_TERM_ANELASTICITY: "VM7",
            SolidEarthModelPart.SHORT_TERM_ANELASTICITY: "Benjamin_Q_Resovsky",
        },
    )
    solid_earth_parameters = solid_earth_full_numerical_model.solid_earth_parameters

    ratio = {}

    for period in [10.0, 100.0, 1000.0]:

        ratio[period] = {}

        for use_long_term_anelasticity in [True, False]:

            ratio[period][use_long_term_anelasticity] = {}

            for use_short_term_anelasticity in [True, False]:

                solid_earth_parameters.model.options.use_long_term_anelasticity = (
                    use_long_term_anelasticity
                )
                solid_earth_parameters.model.options.use_short_term_anelasticity = (
                    use_short_term_anelasticity
                )
                solid_earth_full_numerical_model.solid_earth_parameters = solid_earth_parameters
                solid_earth_time_dependent_numerical_model = SolidEarthTimeDependentNumericalModel(
                    solid_earth_full_numerical_model=solid_earth_full_numerical_model,
                    period=period,
                    n=1,
                )

                ratio[period][use_long_term_anelasticity][use_short_term_anelasticity] = {
                    "depth": [],
                    "real": [],
                    "imag": [],
                }

                for model_layer in solid_earth_time_dependent_numerical_model.model_layers[
                    solid_earth_parameters.model.structure_parameters.below_cmb_layers :
                ]:

                    x = model_layer.x_profile(
                        spline_number=solid_earth_parameters.numerical_parameters.spline_number,
                    )
                    variable = model_layer.evaluate(x=x, variable="mu_0") / (
                        model_layer.evaluate(x=x, variable="mu_real")
                        + 1.0j * model_layer.evaluate(x=x, variable="mu_imag")
                    )

                    ratio[period][use_long_term_anelasticity][use_short_term_anelasticity][
                        "real"
                    ] += (list(variable.real),)
                    ratio[period][use_long_term_anelasticity][use_short_term_anelasticity][
                        "imag"
                    ] += (list(variable.imag),)
                    ratio[period][use_long_term_anelasticity][use_short_term_anelasticity][
                        "depth"
                    ] += list((1.0 - x) * solid_earth_parameters.model.radius_unit / 1e3)

    save_base_model(
        obj=ratio,
        name="figure_3",
        path=figures_path,
    )


def preprocess_figure_4() -> None:
    """
    Stores the whole needed data for panel B in data.csv and the standard deviations per parameter
    in results.
    """

    data, parameters = preprocess_dataframe(
        metrics=["ocean_mean_trend_step_5"],
        filter_wanted_values=DEFAULT_FILTER_WANTED_VALUES,
        filter_unwanted_values=DEFAULT_FILTER_UNWANTED_VALUES,
    )
    results, sorted_parameters = preprocess_variabilities(
        data=data, parameters=parameters, metrics=["ocean_mean_trend_step_5"]
    )
    sorted_parameters.reverse()
    replace_and_save_to_csv(df=data, filepath=figures_path.joinpath("data_step_5.csv"), index=False)
    save_base_model(
        obj={"results": results, "sorted_parameters": sorted_parameters},
        name="figure_4",
        path=figures_path,
    )
