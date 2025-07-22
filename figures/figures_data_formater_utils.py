"""
Generic functions for figure data formaters.
"""

from pathlib import Path

import numpy
from pandas import DataFrame, read_csv

from transient_solid_earth import (
    LAYERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    VALUES_SEPARATOR,
    ElasticLoadModel,
    _pool_apply_DDK_filter,
    collection_sh_data_from_grid,
    grid_from_collection_sh_data,
    harmonic_residual_trends_path,
    load_base_model,
    make_grid,
    make_unstacked_harmonics,
    mean_on_mask,
    stack_harmonics,
    tables_path,
)

MODEL_ID_PER_SOLUTION = {
    "MSSA": "a53db61f31",
    "CSR": "2eb1f8a27a",
    "GFZ": "c3de56dc19",
    "JPL": "bb8d00c0e7",
}

REFERENCE_ELASTIC_LOAD_MODEL_ID = MODEL_ID_PER_SOLUTION["MSSA"]
ELASTIC_LOAD_MODEL_WITH_LIA_ID = "7f6c6be21c"
ELASTIC_LOAD_MODEL_WITHOUT_LIA_ID = MODEL_ID_PER_SOLUTION["MSSA"]
ELASTIC_REFERENCE_LOAD_MODEL_ID = "345d62c531"
ANELASTIC_REFERENCE_LOAD_MODEL_ID = "accde02769"

REFERENCE_MODEL_PARAMETERS: dict[str, str | bool | float] = {
    "Q profile": "Resovsky",
    "relaxed shear\nmodulus ratio": "0.15",
    "alpha": "0.26",
    "viscosity\nprofile": "VM7",
    "Asthenosphere\nviscosity": "3e+19",
    "LIA": "False",
    "pole\ntime series": "mean",
    "Uniform\ncontinental load model": "False",
    "elastic mean\nbarystatic sea\nlevel": "mean",
    "GRACE\nsolution": "MSSA",
}


DEFAULT_FILTER_WANTED_VALUES = {
    "Uniform\ncontinental load model": False,
    "pole\ntime series": "mean",
    "alpha": 0.26,
}

DEFAULT_FILTER_UNWANTED_VALUES = {
    "Asthenosphere\nviscosity": 3e18,
    "viscosity\nprofile": "Mao_Zhong",
}

LONG_TERM_MAP = {
    "VM7": "VM7",
    "VM5a": "VM5a",
    "Lau": "L2016",
    "Lambeck": "L2017",
    "Caron": "C2018",
    "Mao_Zhong": "M&Z2021",
}

SHORT_TERM_MAP = {
    "Benjamin_Q_Resovsky": "R2004",
    "Benjamin_Q_PAR3P": "PAR3P",
    "Benjamin_Q_PREM": "PREM",
    "Benjamin_Q_QL6": "QL6",
    "Benjamin_Q_QM1": "QM1",
}


ANELASTICITY_OPTIONS = ["elastic", "long-term", "short-term", "short-term\nand long-term"]

figures_path = Path(".").joinpath("figures")


def read_csv_and_replace(filepath: Path, **kwargs):
    """
    Reads a CSV file, replacing '_:_' with '\\n' in column names and string values.
    """

    df: DataFrame = read_csv(filepath, **kwargs)
    df.columns = [col.replace("_:_", "\n") for col in df.columns]

    for col in df.select_dtypes(include="object").columns:

        df[col] = df[col].apply(lambda x: x.replace("_:_", "\n") if isinstance(x, str) else x)

    return df


def replace_and_save_to_csv(df: DataFrame, filepath: Path, **kwargs):
    """
    Saves a DataFrame to CSV, replacing '\\n' with '_:_' in column names and string values.
    """

    df_copy = df.copy()
    df_copy.columns = [col.replace("\n", "_:_") for col in df_copy.columns]

    for col in df_copy.select_dtypes(include="object").columns:

        df_copy[col] = df_copy[col].apply(
            lambda x: x.replace("\n", "_:_") if isinstance(x, str) else x
        )

    df_copy.to_csv(filepath, **kwargs)


def get_grid(
    harmonics: numpy.ndarray[float], n_max: int, decimals: int = 6
) -> numpy.ndarray[float]:
    """
    Projects spherical harmonics on a (latitude x longitude) grid.
    """

    return numpy.round(a=make_grid(harmonics=harmonics, n_max=n_max), decimals=decimals)


def preprocess_grid(
    load_model: ElasticLoadModel,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
    """
    Masks, shift longitudes and clips.
    """

    grid = get_grid(
        harmonics=load_model.base_products.load_model_harmonic_component,
        n_max=load_model.load_model_parameters.signature.n_max,
    )

    if (
        load_model.elastic_load_model_spatial_products.ocean_land_mask is not None
        and load_model.load_model_parameters.numerical_parameters.ewh_threshold is not None
    ):

        load_model.elastic_load_model_spatial_products.ocean_land_mask = (
            load_model.elastic_load_model_spatial_products.ocean_land_mask
            * (abs(grid) < load_model.load_model_parameters.numerical_parameters.ewh_threshold)
        )

    # Ensures longitudes are in the range [-180, 180].
    longitudes = numpy.where(
        load_model.longitudes() > 180,
        load_model.longitudes() - 360,
        load_model.longitudes(),
    )
    latitudes = load_model.latitudes()

    # Sorts longitudes and corresponding data.
    sort_idx = numpy.argsort(longitudes)
    longitudes = longitudes[sort_idx]
    grid = grid[:, sort_idx]
    mask = (
        None
        if load_model.elastic_load_model_spatial_products.ocean_land_mask is None
        else load_model.elastic_load_model_spatial_products.ocean_land_mask[:, sort_idx]
    )

    return (
        latitudes,
        longitudes,
        mask,
        grid,
        mean_on_mask(
            mask=mask,
            latitudes=latitudes,
            load_model_parameters=load_model.load_model_parameters,
            grid_or_harmonics=grid,
            ewh_threshold=numpy.inf,
        ),
    )


def preprocess_dataframe(
    metrics: list[str],
    filter_wanted_values: dict,
    filter_unwanted_values: dict,
    table_file: str = "backup",
) -> tuple[DataFrame, list[str]]:
    """
    Formats and select information from the results table.
    """

    # Loads the dataframe.
    load_signal_trends = read_csv_and_replace(filepath=tables_path.joinpath(table_file + ".csv"))

    # Formats the wanted attributes.
    selected_columns = DataFrame()
    selected_columns["LIA"] = load_signal_trends["history:lia:use"].values
    selected_columns["pole\ntime series"] = load_signal_trends["history:pole:case"].values
    selected_columns["Uniform\ncontinental load model"] = load_signal_trends[
        "signature:opposite_load_on_continents"
    ].values
    selected_columns["elastic mean\nbarystatic sea\nlevel"] = load_signal_trends[
        "history:case"
    ].values
    selected_columns["viscosity\nprofile"] = [
        (
            None
            if model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[1] == "unused"
            else model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[1].split(
                SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR
            )[0]
        )
        for model_name in load_signal_trends["rheological_model_id"].values
    ]
    selected_columns["Asthenosphere\nviscosity"] = [
        (
            None
            if model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[1] == "unused"
            else float(
                model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[1].split(
                    VALUES_SEPARATOR
                )[-1]
            )
        )
        for model_name in load_signal_trends["rheological_model_id"].values
    ]
    selected_columns["Q profile"] = [
        (
            None
            if model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[2] == "unused"
            else model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[2].split("_")[2]
        )
        for model_name in load_signal_trends["rheological_model_id"].values
    ]
    selected_columns["GRACE\nsolution"] = [
        file.split("/")[-1] for file in load_signal_trends["signature:file"].values
    ]
    selected_columns["GRACE\nsolution"] = [
        "MSSA" if "MSSA" in solution else solution
        for solution in selected_columns["GRACE\nsolution"].values
    ]
    selected_columns["relaxed shear\nmodulus ratio"] = [
        (
            None
            if model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[2] == "unused"
            else float(
                model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[2]
                .split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR)[1]
                .split(LAYERS_SEPARATOR)[0]
                .split(VALUES_SEPARATOR)[-1]
            )
        )
        for model_name in load_signal_trends["rheological_model_id"].values
    ]
    selected_columns["alpha"] = [
        (
            None
            if model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[2] == "unused"
            else float(
                model_name.split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR)[2]
                .split(SOLID_EARTH_NUMERICAL_MODEL_PART_NAME_FROM_PARAMETERS_SEPARATOR)[1]
                .split(LAYERS_SEPARATOR)[-1]
                .split(VALUES_SEPARATOR)[-1]
            )
        )
        for model_name in load_signal_trends["rheological_model_id"].values
    ]
    selected_columns["Anelasticity"] = [
        (
            "elastic"
            if (not viscosity_profile) and (not q_profile)
            else (
                "long-term"
                if not q_profile
                else ("short-term" if not viscosity_profile else "short-term\nand long-term")
            )
        )
        for viscosity_profile, q_profile in zip(
            selected_columns["viscosity\nprofile"].values, selected_columns["Q profile"].values
        )
    ]

    for metric in metrics:

        selected_columns[metric] = load_signal_trends[metric].values

    # Eventually filters by wanted values.
    for parameter, value in filter_wanted_values.items():

        selected_columns = selected_columns[
            (selected_columns[parameter] == value) | (selected_columns[parameter].isna())
        ]

    # Eventually filters by unwanted values.
    for parameter, value in filter_unwanted_values.items():

        selected_columns = selected_columns[
            (selected_columns[parameter] != value) | (selected_columns[parameter].isna())
        ]

    # Computes standard deviation for given variable parameters, the other being fixed.
    parameters = [
        parameter
        for parameter in selected_columns.columns
        if parameter not in filter_wanted_values.keys()
        and parameter not in metrics
        and parameter != "Anelasticity"
    ]

    return selected_columns, parameters


def preprocess_variabilities(
    data: DataFrame, parameters: list[str], metrics: list[str]
) -> tuple[DataFrame, list[str]]:
    """
    Gets variabilities per parameters.
    """

    # Computes standard deviation for given variable parameters, the other being fixed.
    results = {}

    for metric in metrics:

        results[metric] = {}

        for option in ANELASTICITY_OPTIONS:

            sub_dataframe = data[data["Anelasticity"] == option]

            # Remove columns with only None or NaN values
            cols_to_drop = [
                col
                for col in sub_dataframe.columns
                if sub_dataframe[col].isna().all() or sub_dataframe[col].eq(None).all()
            ]
            sub_dataframe = sub_dataframe.drop(columns=cols_to_drop)
            sub_parameters = [p for p in parameters if p not in cols_to_drop]

            results[metric][option] = {}

            for parameter in sub_parameters:

                results[metric][option][parameter] = []
                other_parameters = [
                    other_parameter
                    for other_parameter in sub_parameters
                    if other_parameter != parameter
                ]

                for combination in numpy.unique(
                    numpy.array(
                        sub_dataframe.loc[:, other_parameters],
                        dtype=str,
                    ),
                    axis=0,
                ):

                    sub_sub_dataframe = sub_dataframe.copy()

                    for parameter_value, other_parameter in zip(combination, other_parameters):

                        sub_sub_dataframe = sub_sub_dataframe[
                            numpy.array(object=sub_sub_dataframe[other_parameter].values, dtype=str)
                            == str(parameter_value)
                        ]

                    results[metric][option][parameter] += (
                        [numpy.std(sub_sub_dataframe[metric])]
                        if len(sub_sub_dataframe[metric].unique()) > 1
                        else []
                    )

    # Sorts the parameters by decreasing order of induced variability for the first metric.
    return results, [
        parameters[i]
        for i in numpy.argsort(
            [
                (
                    0.0
                    if len(results[metrics[0]]["short-term\nand long-term"][parameter]) == 0
                    else max(results[metrics[0]]["short-term\nand long-term"][parameter])
                )
                for parameter in parameters
            ]
        )
    ]


def process_residuals(
    elastic_load_model: ElasticLoadModel,
    name: str,
    use_backup: bool,
    remove_21: bool = False,
    apply_filter: bool = False,
) -> tuple[numpy.ndarray, float]:
    """
    Returns only grid and mean.
    """

    if use_backup:

        grid = make_grid(
            harmonics=elastic_load_model.base_products.load_model_harmonic_component,
            n_max=elastic_load_model.load_model_parameters.signature.n_max,
        )

    else:

        grid = load_base_model(name=name, path=harmonic_residual_trends_path)

    if apply_filter:

        sh_data = collection_sh_data_from_grid(
            grid=grid,
            n_max=elastic_load_model.load_model_parameters.signature.n_max,
        )
        grid = grid_from_collection_sh_data(
            collection_data=_pool_apply_DDK_filter(sh_data, ddk_filter_level=2),
            n_max=elastic_load_model.load_model_parameters.signature.n_max,
        )

    harmonics = make_unstacked_harmonics(
        grid=grid,
        n_max=elastic_load_model.load_model_parameters.signature.n_max,
    )

    if remove_21:

        harmonics[0, 2, 1] = 0
        harmonics[1, 2, 1] = 0

    elastic_load_model.base_products.load_model_harmonic_component = stack_harmonics(harmonics)
    return preprocess_grid(load_model=elastic_load_model)[-2:]
