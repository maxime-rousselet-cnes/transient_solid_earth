"""
Formats data for figures of the supplementary materials.
"""

import numpy
from scipy import interpolate

from transient_solid_earth import (
    DEFAULT_SOLID_EARTH_MODEL_OPTION_PARAMETERS,
    BoundaryCondition,
    Direction,
    ElasticLoadModel,
    SolidEarthVariableParameters,
    adaptative_step_parallel_computing_loop,
    anelastic_load_models_path,
    create_all_model_variations,
    get_ocean_mean_trend,
    get_periods,
    harmonic_geoid_deformation_trends_path,
    harmonic_residual_trends_path,
    harmonic_vertical_displacement_trends_path,
    intermediate_result_subpaths,
    load_base_model,
    load_complex_array,
    load_elastic_load_model,
    load_parameters,
    make_harmonics,
    make_unstacked_harmonics,
    save_base_model,
    stack_harmonics,
)

from .figures_data_formater_utils import (
    ANELASTIC_REFERENCE_LOAD_MODEL_ID,
    DEFAULT_FILTER_UNWANTED_VALUES,
    DEFAULT_FILTER_WANTED_VALUES,
    ELASTIC_LOAD_MODEL_WITH_LIA_ID,
    ELASTIC_LOAD_MODEL_WITHOUT_LIA_ID,
    ELASTIC_REFERENCE_LOAD_MODEL_ID,
    MODEL_ID_PER_SOLUTION,
    REFERENCE_ELASTIC_LOAD_MODEL_ID,
    figures_path,
    preprocess_dataframe,
    preprocess_grid,
    process_residuals,
    replace_and_save_to_csv,
)


def preprocess_figure_sup_1() -> None:
    """
    Barystatic history model with and without LIA.
    """

    elastic_load_model_without_lia = load_elastic_load_model(
        model_id=ELASTIC_LOAD_MODEL_WITHOUT_LIA_ID
    )
    elastic_load_model_with_lia = load_elastic_load_model(model_id=ELASTIC_LOAD_MODEL_WITH_LIA_ID)
    y_with_lia = elastic_load_model_with_lia.base_products.time_dependent_component[
        numpy.where(elastic_load_model_with_lia.base_products.time_dependent_component != 0)[0][
            0
        ] : elastic_load_model_with_lia.side_products.recent_trend_indices[-1]
    ]
    y_without_lia = elastic_load_model_without_lia.base_products.time_dependent_component[
        numpy.where(elastic_load_model_with_lia.base_products.time_dependent_component != 0)[0][
            0
        ] : elastic_load_model_with_lia.side_products.recent_trend_indices[-1]
    ]
    dates = elastic_load_model_with_lia.base_products.temporal_products.full_load_model_dates[
        numpy.where(elastic_load_model_with_lia.base_products.time_dependent_component != 0)[0][
            0
        ] : elastic_load_model_with_lia.side_products.recent_trend_indices[-1]
    ]
    y_with_lia -= y_with_lia[len(y_with_lia) // 2]
    save_base_model(
        obj={"dates": dates, "y_with_lia": y_with_lia, "y_without_lia": y_without_lia},
        name="figure_sup_1",
        path=figures_path,
    )


def preprocess_figure_sup_2() -> None:
    """
    CSR trends and diff CSR with 3 other solutions.
    """

    elastic_load_models: dict[str, ElasticLoadModel] = {}

    for solution, model_id in MODEL_ID_PER_SOLUTION.items():

        elastic_load_models[solution] = load_elastic_load_model(model_id=model_id)
        elastic_load_models[solution].elastic_load_model_spatial_products.ocean_land_mask = None

    obj = {}
    n_max = min(
        elastic_load_model.load_model_parameters.signature.n_max
        for elastic_load_model in elastic_load_models.values()
    )

    for solution in MODEL_ID_PER_SOLUTION:

        if solution != "CSR":

            elastic_load_models[solution].base_products.load_model_harmonic_component[
                : n_max + 1, : n_max + 1
            ] -= elastic_load_models["CSR"].base_products.load_model_harmonic_component[
                : n_max + 1, : n_max + 1
            ]

    for solution in MODEL_ID_PER_SOLUTION:

        elastic_load_models[solution].load_model_parameters.signature.n_max = n_max
        latitudes, longitudes, mask, grid, _ = preprocess_grid(
            load_model=elastic_load_models[solution]
        )
        obj |= {
            "latitudes": latitudes,
            "longitudes": longitudes,
            "mask": mask,
            solution: grid,
        }

    save_base_model(
        obj=obj,
        name="figure_sup_2",
        path=figures_path,
    )


def preprocess_figure_sup_3() -> None:
    """
    k2 and k2' over their elastic values for reference and extremal models.
    """

    # Loads parameters and rheological models.
    parameters = load_parameters()
    parameters.solid_earth_variabilities = SolidEarthVariableParameters(
        model_names={
            "elasticity": ["PREM"],
            "long_term_anelasticity": ["VM7"],
            "short_term_anelasticity": ["Benjamin_Q_Resovsky"],
        },
        rheological_parameters={
            "long_term_anelasticity": {"eta_m": {"ASTHENOSPHERE": [[3e19]]}},
            "short_term_anelasticity": {
                "alpha": {"MANTLE": [[0.223], [0.26], [0.297]]},
                "asymptotic_mu_ratio": {"MANTLE": [[0.1], [0.15], [0.2]]},
            },
        },
    )
    elastic_model, anelastic_models = create_all_model_variations(
        variable_parameters=parameters.solid_earth_variabilities,
        solid_earth_model_option_list=[DEFAULT_SOLID_EARTH_MODEL_OPTION_PARAMETERS],
    )[0]
    adaptative_step_parallel_computing_loop(
        rheologies=[elastic_model] + anelastic_models,
        degree_list=[2],
        parameters=parameters,
    )

    love_numbers = load_complex_array(
        path=intermediate_result_subpaths["love_numbers"].joinpath("PREM_____unused_____unused"),
        name="2.0",
    )
    elastic_potential_love_number = love_numbers[
        0, BoundaryCondition.POTENTIAL.value, Direction.POTENTIAL.value
    ]
    elastic_load_love_number = love_numbers[
        0, BoundaryCondition.LOAD.value, Direction.POTENTIAL.value
    ]
    new_periods = numpy.logspace(start=-1, stop=2, base=10, num=100)
    obj = {"periods": new_periods, "reference": {}, "highest": {}, "lowest": {}}
    exponentiation_base = parameters.discretization["love_numbers"].exponentiation_base

    for case, name in zip(
        ["reference", "highest", "lowest"],
        [
            "PREM_____VM7____eta_m__ASTHENOSPHERE__3e+19_____Benjamin_Q_Resovsky",
            "PREM_____VM7____eta_m__ASTHENOSPHERE__3e+19_____Benjamin_Q_Resovsky"
            + "____alpha__MANTLE__0.297___asymptotic_mu_ratio__MANTLE__0.1",
            "PREM_____VM7____eta_m__ASTHENOSPHERE__3e+19_____Benjamin_Q_Resovsky"
            + "____alpha__MANTLE__0.223___asymptotic_mu_ratio__MANTLE__0.2",
        ],
    ):

        path = intermediate_result_subpaths["love_numbers"].joinpath(name).joinpath("2.0")
        love_numbers = load_complex_array(path=path)
        potential_love_numbers: numpy.ndarray = (
            love_numbers[:, BoundaryCondition.POTENTIAL.value, Direction.POTENTIAL.value]
            / elastic_potential_love_number
        )
        load_love_numbers: numpy.ndarray = (
            love_numbers[:, BoundaryCondition.LOAD.value, Direction.POTENTIAL.value]
            / elastic_load_love_number
        )
        periods = get_periods(path=path)
        potential_love_numbers = interpolate.interp1d(
            x=numpy.log(periods) / exponentiation_base,
            y=potential_love_numbers.real,
            axis=0,
        )(x=numpy.log(new_periods) / exponentiation_base) + 1.0j * interpolate.interp1d(
            x=numpy.log(periods) / exponentiation_base,
            y=potential_love_numbers.imag,
            axis=0,
        )(
            x=numpy.log(new_periods) / exponentiation_base
        )
        load_love_numbers = interpolate.interp1d(
            x=numpy.log(periods) / exponentiation_base,
            y=load_love_numbers.real,
            axis=0,
        )(x=numpy.log(new_periods) / exponentiation_base) + 1.0j * interpolate.interp1d(
            x=numpy.log(periods) / exponentiation_base,
            y=load_love_numbers.imag,
            axis=0,
        )(
            x=numpy.log(new_periods) / exponentiation_base
        )
        obj[case] = {
            "potential": {
                "real": potential_love_numbers.real,
                "imag": potential_love_numbers.imag,
            },
            "load": {
                "real": load_love_numbers.real,
                "imag": load_love_numbers.imag,
            },
        }

    save_base_model(
        obj=obj,
        name="figure_sup_3",
        path=figures_path,
    )


def preprocess_figure_sup_4() -> None:
    """
    Degree one comparison.
    """

    elastic_load_model = load_elastic_load_model(model_id=REFERENCE_ELASTIC_LOAD_MODEL_ID)
    elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask = (
        elastic_load_model.elastic_load_model_spatial_products.ocean_land_buffered_mask
    )

    grid = numpy.array(
        object=load_base_model(
            name=ELASTIC_REFERENCE_LOAD_MODEL_ID, path=anelastic_load_models_path.joinpath("step_3")
        )
    )
    pre_inversion_harmonics = make_unstacked_harmonics(
        grid=grid, n_max=elastic_load_model.load_model_parameters.signature.n_max
    )
    pre_inversion_harmonics[:, 2:, :] = 0
    pre_inversion_harmonics[0, 0, 0] = 0
    elastic_load_model.base_products.load_model_harmonic_component = stack_harmonics(
        harmonics=pre_inversion_harmonics
    )
    latitudes, longitudes, mask, pre_inversion_degree_one_grid, pre_inversion_degree_one_mean = (
        preprocess_grid(load_model=elastic_load_model)
    )

    grid = numpy.array(
        object=load_base_model(
            name=ELASTIC_REFERENCE_LOAD_MODEL_ID, path=anelastic_load_models_path.joinpath("step_4")
        )
    )
    elastic_post_inversion_harmonics = make_unstacked_harmonics(
        grid=grid, n_max=elastic_load_model.load_model_parameters.signature.n_max
    )
    elastic_post_inversion_harmonics[:, 2:, :] = 0
    elastic_post_inversion_harmonics[0, 0, 0] = 0
    elastic_load_model.base_products.load_model_harmonic_component = stack_harmonics(
        harmonics=elastic_post_inversion_harmonics
    )
    _, _, _, elastic_post_inversion_degree_one_grid, elastic_post_inversion_degree_one_mean = (
        preprocess_grid(load_model=elastic_load_model)
    )

    grid = numpy.array(
        object=load_base_model(
            name=ANELASTIC_REFERENCE_LOAD_MODEL_ID,
            path=anelastic_load_models_path.joinpath("step_4"),
        )
    )
    anelastic_post_inversion_harmonics = make_unstacked_harmonics(
        grid=grid,
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    anelastic_post_inversion_harmonics[:, 2:, :] = 0
    anelastic_post_inversion_harmonics[0, 0, 0] = 0
    elastic_load_model.base_products.load_model_harmonic_component = stack_harmonics(
        harmonics=anelastic_post_inversion_harmonics
    )
    _, _, _, anelastic_post_inversion_degree_one_grid, anelastic_post_inversion_degree_one_mean = (
        preprocess_grid(load_model=elastic_load_model)
    )

    save_base_model(
        obj={
            "latitudes": latitudes,
            "longitudes": longitudes,
            "mask": mask,
            "pre_inversion_degree_one_grid": pre_inversion_degree_one_grid,
            "pre_inversion_degree_one_mean": pre_inversion_degree_one_mean,
            "elastic_post_inversion_degree_one_grid": elastic_post_inversion_degree_one_grid,
            "elastic_post_inversion_degree_one_mean": elastic_post_inversion_degree_one_mean,
            "anelastic_post_inversion_degree_one_grid": anelastic_post_inversion_degree_one_grid,
            "anelastic_post_inversion_degree_one_mean": anelastic_post_inversion_degree_one_mean,
        },
        name="figure_sup_4",
        path=figures_path,
    )


def preprocess_figure_sup_5() -> None:
    """
    Fingerprint components.
    """

    elastic_load_model = load_elastic_load_model(model_id=REFERENCE_ELASTIC_LOAD_MODEL_ID)
    elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask = (
        elastic_load_model.elastic_load_model_spatial_products.ocean_land_buffered_mask
    )

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ELASTIC_REFERENCE_LOAD_MODEL_ID, path=harmonic_geoid_deformation_trends_path
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    latitudes, longitudes, mask, elastic_geoid_deformation_grid, elastic_geoid_deformation_mean = (
        preprocess_grid(load_model=elastic_load_model)
    )

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ANELASTIC_REFERENCE_LOAD_MODEL_ID, path=harmonic_geoid_deformation_trends_path
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    (
        latitudes,
        longitudes,
        mask,
        anelastic_geoid_deformation_grid,
        anelastic_geoid_deformation_mean,
    ) = preprocess_grid(load_model=elastic_load_model)

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ELASTIC_REFERENCE_LOAD_MODEL_ID, path=harmonic_vertical_displacement_trends_path
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    (
        _,
        _,
        _,
        elastic_vertical_displacement_grid,
        elastic_vertical_displacement_mean,
    ) = preprocess_grid(load_model=elastic_load_model)

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ANELASTIC_REFERENCE_LOAD_MODEL_ID, path=harmonic_vertical_displacement_trends_path
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    (
        latitudes,
        longitudes,
        mask,
        anelastic_vertical_displacement_grid,
        anelastic_vertical_displacement_mean,
    ) = preprocess_grid(load_model=elastic_load_model)

    save_base_model(
        obj={
            "latitudes": latitudes,
            "longitudes": longitudes,
            "mask": mask,
            "elastic_geoid_deformation_grid": elastic_geoid_deformation_grid,
            "elastic_geoid_deformation_mean": elastic_geoid_deformation_mean,
            "anelastic_geoid_deformation_grid": anelastic_geoid_deformation_grid,
            "anelastic_geoid_deformation_mean": anelastic_geoid_deformation_mean,
            "elastic_vertical_displacement_grid": elastic_vertical_displacement_grid,
            "elastic_vertical_displacement_mean": elastic_vertical_displacement_mean,
            "anelastic_vertical_displacement_grid": anelastic_vertical_displacement_grid,
            "anelastic_vertical_displacement_mean": anelastic_vertical_displacement_mean,
        },
        name="figure_sup_5",
        path=figures_path,
    )


def preprocess_figure_sup_6(use_backup: bool = True) -> None:
    """
    Inversion residuals.
    """

    if use_backup:

        backup = load_base_model(name="residuals", path=figures_path)

    elastic_load_model = load_elastic_load_model(model_id=REFERENCE_ELASTIC_LOAD_MODEL_ID)
    elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask = (
        elastic_load_model.elastic_load_model_spatial_products.ocean_land_buffered_mask
    )

    elastic_grid = load_base_model(
        name=ELASTIC_REFERENCE_LOAD_MODEL_ID, path=harmonic_residual_trends_path
    )

    if use_backup:

        harmonics = numpy.array(object=backup["elastic"])[
            :,
            : elastic_load_model.load_model_parameters.signature.n_max + 1,
            : elastic_load_model.load_model_parameters.signature.n_max + 1,
        ]

    else:

        harmonics = make_unstacked_harmonics(
            grid=elastic_grid,
            n_max=elastic_load_model.load_model_parameters.signature.n_max,
        )

    elastic_load_model.base_products.load_model_harmonic_component = stack_harmonics(harmonics)
    latitudes, longitudes, mask, residuals_grid, residuals_mean = preprocess_grid(
        load_model=elastic_load_model
    )
    obj = {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "mask": mask,
        "elastic_residuals_grid": residuals_grid.copy(),
        "elastic_residuals_mean": residuals_mean,
    }

    residuals_grid, residuals_mean = process_residuals(
        elastic_load_model=elastic_load_model,
        name=ELASTIC_REFERENCE_LOAD_MODEL_ID,
        use_backup=use_backup,
        apply_filter=True,
    )
    obj |= {
        "elastic_filtered_residuals_grid": residuals_grid.copy(),
        "elastic_filtered_residuals_mean": residuals_mean,
    }

    residuals_grid, residuals_mean = process_residuals(
        elastic_load_model=elastic_load_model,
        name=ELASTIC_REFERENCE_LOAD_MODEL_ID,
        use_backup=use_backup,
        remove_21=True,
        apply_filter=True,
    )
    obj |= {
        "elastic_residuals_grid_without_2_1": residuals_grid.copy(),
        "elastic_residuals_mean_without_2_1": residuals_mean,
    }

    if use_backup:

        elastic_load_model.base_products.load_model_harmonic_component = stack_harmonics(
            harmonics=numpy.array(object=backup["anelastic"])[
                :,
                : elastic_load_model.load_model_parameters.signature.n_max + 1,
                : elastic_load_model.load_model_parameters.signature.n_max + 1,
            ]
        )

    residuals_grid, residuals_mean = process_residuals(
        elastic_load_model=elastic_load_model,
        name=ANELASTIC_REFERENCE_LOAD_MODEL_ID,
        use_backup=use_backup,
    )
    obj |= {
        "anelastic_residuals_grid": residuals_grid.copy(),
        "anelastic_residuals_mean": residuals_mean,
    }

    residuals_grid, residuals_mean = process_residuals(
        elastic_load_model=elastic_load_model,
        name=ANELASTIC_REFERENCE_LOAD_MODEL_ID,
        use_backup=use_backup,
        apply_filter=True,
    )
    obj |= {
        "anelastic_filtered_residuals_grid": residuals_grid.copy(),
        "anelastic_filtered_residuals_mean": residuals_mean,
    }

    residuals_grid, residuals_mean = process_residuals(
        elastic_load_model=elastic_load_model,
        name=ANELASTIC_REFERENCE_LOAD_MODEL_ID,
        use_backup=use_backup,
        apply_filter=True,
        remove_21=True,
    )
    obj |= {
        "anelastic_residuals_grid_without_2_1": residuals_grid,
        "anelastic_residuals_mean_without_2_1": residuals_mean,
    }

    save_base_model(
        obj=obj,
        name="figure_sup_6",
        path=figures_path,
    )


def preprocess_figure_sup_7() -> None:
    """
    before/after leakage correction.
    """

    elastic_load_model = load_elastic_load_model(model_id=REFERENCE_ELASTIC_LOAD_MODEL_ID)
    elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask = (
        elastic_load_model.elastic_load_model_spatial_products.ocean_land_buffered_mask
    )

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ELASTIC_REFERENCE_LOAD_MODEL_ID, path=anelastic_load_models_path.joinpath("step_4")
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    elastic_load_model.load_model_parameters.numerical_parameters.ewh_threshold = None
    latitudes, longitudes, mask, elastic_grid, elastic_mean = preprocess_grid(
        load_model=elastic_load_model
    )

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ANELASTIC_REFERENCE_LOAD_MODEL_ID,
            path=anelastic_load_models_path.joinpath("step_4"),
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    _, _, _, anelastic_grid, anelastic_mean = preprocess_grid(load_model=elastic_load_model)

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ELASTIC_REFERENCE_LOAD_MODEL_ID, path=anelastic_load_models_path.joinpath("step_5")
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    latitudes, longitudes, mask, elastic_corrected_grid, elastic_corrected_mean = preprocess_grid(
        load_model=elastic_load_model
    )

    elastic_load_model.base_products.load_model_harmonic_component = make_harmonics(
        grid=load_base_model(
            name=ANELASTIC_REFERENCE_LOAD_MODEL_ID,
            path=anelastic_load_models_path.joinpath("step_5"),
        ),
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )
    _, _, _, anelastic_corrected_grid, anelastic_corrected_mean = preprocess_grid(
        load_model=elastic_load_model
    )

    save_base_model(
        obj={
            "latitudes": latitudes,
            "longitudes": longitudes,
            "mask": mask,
            "elastic_grid": elastic_grid,
            "elastic_mean": elastic_mean,
            "anelastic_grid": anelastic_grid,
            "anelastic_mean": anelastic_mean,
            "elastic_corrected_grid": elastic_corrected_grid,
            "elastic_corrected_mean": elastic_corrected_mean,
            "anelastic_corrected_grid": anelastic_corrected_grid,
            "anelastic_corrected_mean": anelastic_corrected_mean,
        },
        name="figure_sup_7",
        path=figures_path,
    )


def preprocess_figure_sup_8() -> None:
    """
    Influence of all successive processing steps.
    """

    all_metrics = ["ocean_mean_trend_step_" + str(i + 1) for i in range(5)]
    data, parameters = preprocess_dataframe(
        metrics=all_metrics,
        filter_wanted_values=DEFAULT_FILTER_WANTED_VALUES,
        filter_unwanted_values=DEFAULT_FILTER_UNWANTED_VALUES,
    )
    replace_and_save_to_csv(
        df=data, filepath=figures_path.joinpath("data_all_steps.csv"), index=False
    )


def preprocess_figure_sup_9() -> None:
    """
    Alpha variations.
    """

    data, _ = preprocess_dataframe(
        metrics=["ocean_mean_trend_step_5"],
        filter_wanted_values={
            k: value for k, value in DEFAULT_FILTER_WANTED_VALUES.items() if k != "alpha"
        },
        filter_unwanted_values=DEFAULT_FILTER_UNWANTED_VALUES,
    )
    replace_and_save_to_csv(df=data, filepath=figures_path.joinpath("data_alpha.csv"), index=False)


def preprocess_figure_sup_10() -> None:
    """
    Uniform continental load model variation.
    """

    data, _ = preprocess_dataframe(
        metrics=["ocean_mean_trend_step_5"],
        filter_wanted_values=DEFAULT_FILTER_WANTED_VALUES
        | {"Uniform\ncontinental load model": True},
        filter_unwanted_values=DEFAULT_FILTER_UNWANTED_VALUES,
    )
    replace_and_save_to_csv(
        df=data, filepath=figures_path.joinpath("data_uniform.csv"), index=False
    )


def get_ocean_time_series(
    solution: numpy.ndarray,
    elastic_load_model: ElasticLoadModel,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Plots recent time-span.
    """

    series = numpy.array(
        object=[
            get_ocean_mean_trend(
                harmonic_load_model_trend=harmonic_slice, elastic_load_model=elastic_load_model
            )
            for harmonic_slice in solution[elastic_load_model.side_products.recent_trend_indices]
        ]
    )

    return (
        elastic_load_model.base_products.temporal_products.full_load_model_dates[
            elastic_load_model.side_products.recent_trend_indices
        ],
        series - series[0],
    )


def preprocess_figure_sup_11() -> None:
    """
    Linear rate between reference cases.
    """

    elastic_load_model = load_elastic_load_model(model_id="968c9070db")
    e = numpy.array(
        object=load_base_model(
            name="95b25974d5", path=anelastic_load_models_path.joinpath("time_dependent")
        )
    )
    a = numpy.array(
        object=load_base_model(
            name="e60b5bdeee", path=anelastic_load_models_path.joinpath("time_dependent")
        )
    )
    dates, series_e = get_ocean_time_series(solution=e, elastic_load_model=elastic_load_model)
    dates, series_a = get_ocean_time_series(solution=a, elastic_load_model=elastic_load_model)
    d = series_a - series_e
    trend_dates = dates - dates[0]
    a_matrix = numpy.vstack(
        [
            trend_dates**2,
            trend_dates,
        ]
    ).T
    result: numpy.ndarray = numpy.linalg.pinv(a_matrix).dot(d[:, None])
    quadratic, linear_from_quadratic = result.flatten()
    rms_from_quadratic = sum((a_matrix.dot(result) - d[:, None]) ** 2) ** 0.5
    a_matrix = numpy.vstack(
        [
            trend_dates,
        ]
    ).T
    result: numpy.ndarray = numpy.linalg.pinv(a_matrix).dot(d[:, None])
    linear = result.flatten()
    rms_from_linear = sum((a_matrix.dot(result) - d[:, None]) ** 2) ** 0.5

    save_base_model(
        obj={
            "dates": dates,
            "trend_dates": trend_dates,
            "d": d,
            "series_e": series_e,
            "series_a": series_a,
            "quadratic": quadratic,
            "linear_from_quadratic": linear_from_quadratic,
            "rms_from_quadratic": rms_from_quadratic,
            "linear": linear,
            "rms_from_linear": rms_from_linear,
        },
        name="figure_sup_11",
        path=figures_path,
    )
