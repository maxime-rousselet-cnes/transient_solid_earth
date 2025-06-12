"""
Opens and formats the data when this formating is dependent on parameters.
"""

from pathlib import Path
from typing import Optional

import numpy
import pandas
from geopandas import GeoDataFrame, read_file
from pyshtools import SHGrid
from pyshtools.shclasses import DHRealGrid, SHCoeffs, SHComplexCoeffs, SHRealCoeffs
from scipy import interpolate
from scipy.fft import fftfreq
from scipy.interpolate import CubicSpline
from shapely.geometry import Point

from .constants import (
    BARYSTATIC_LOAD_MODEL_FILE_COLUMNS,
    EARTH_EQUAL_PROJECTION,
    GRACE_DATA_UNIT_FACTOR,
    LAT_LON_PROJECTION,
    MASK_DECIMALS,
    MEAN_POLE_COEFFICIENTS,
    MEAN_POLE_T_0,
    MILLI_ARC_SECOND_TO_RADIANS,
    SKIPROWS,
)
from .functions import surface_ponderation, trend
from .parameters import LoadModelParameters
from .paths import gmsl_data_path, grace_data_path, masks_data_path, pole_data_path


def load_barystatic_load_model(
    load_model_parameters: LoadModelParameters,
    path: Path = gmsl_data_path,
    zero_at_origin: bool = True,
) -> tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Opens Frederikse et al.'s file and formats its data.
    Mean load in equivalent water height with respect to time.
    """

    # Gets raw data.
    df = pandas.read_csv(
        filepath_or_buffer=path.joinpath(load_model_parameters.history.file), sep=","
    )
    signal_dates = df["Unnamed: 0"].values

    # Formats. Barystatic = Sum - Steric.
    sum_lists: dict[str, list[str]] = {
        column: list(df["Sum of contributors [" + column + "]"].values)
        for column in BARYSTATIC_LOAD_MODEL_FILE_COLUMNS
    }
    steric_lists: dict[str, list[str]] = {
        column: list(df["Steric [" + column + "]"].values)
        for column in BARYSTATIC_LOAD_MODEL_FILE_COLUMNS
    }
    barystatic: dict[str, numpy.ndarray[float]] = {
        column: numpy.array(
            object=[
                float(item.split(",")[0] + "." + item.split(",")[1]) for item in sum_lists[column]
            ],
            dtype=float,
        )
        - numpy.array(
            object=[
                float(item.split(",")[0] + "." + item.split(",")[1])
                for item in steric_lists[column]
            ],
            dtype=float,
        )
        for column in BARYSTATIC_LOAD_MODEL_FILE_COLUMNS
    }

    # For worst/best case, build the maximum/minimum slope barystatic curb.
    if load_model_parameters.history.case in ["best", "worst"]:
        if load_model_parameters.history.case == "best":
            start, end = "upper", "lower"
        else:
            start, end = "lower", "upper"
        a = (barystatic[start][-1] - barystatic[end][0]) / (
            barystatic["mean"][-1] - barystatic["mean"][0]
        )
        b = barystatic[end][0] - a * barystatic["mean"][0]
        barystatic[load_model_parameters.history.case] = a * barystatic["mean"] + b

    return signal_dates, barystatic[load_model_parameters.history.case] - (
        barystatic[load_model_parameters.history.case][0] if zero_at_origin else 0
    )


def convolve(a: numpy.ndarray[float], v: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    For Chandler wobble filtering.
    """
    return numpy.array(
        [sum(a[i + j] * v[i] for i in range(len(v))) for j in range(len(a) - len(v) + 1)]
    )


def filter_wobble(
    m_1: numpy.ndarray[float],
    m_2: numpy.ndarray[float],
    polar_time_series_dates: numpy.ndarray[float],
    load_model_parameters: LoadModelParameters,
) -> tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Filtering the Annual and Chandler wobbles.
    """

    kernel_length = 2 * load_model_parameters.history.pole.wobble_filtering_kernel_length + 1
    kernel = numpy.ones(shape=[kernel_length]) / kernel_length
    m_1 = convolve(a=m_1, v=kernel)
    m_2 = convolve(a=m_2, v=kernel)
    polar_time_series_dates = numpy.linspace(
        start=polar_time_series_dates[0], stop=polar_time_series_dates[-1], num=len(m_1)
    )

    return m_1, m_2, polar_time_series_dates


def load_polar_motion_time_series(
    load_model_parameters: LoadModelParameters,
    path: Path = pole_data_path,
    i: int = 1,  # Column factor.
) -> tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Opens C04 time series file and formats its data (mas).
    """

    with open(
        file=path.joinpath(load_model_parameters.history.pole.file + ".txt"),
        mode="r",
        encoding="utf-8",
    ) as file:

        lines = file.readlines()

    dates, pole_motion = [], {"m_1": [], "m_2": []}

    for line in lines[1:]:

        items = [
            float(k) for k in [string for string in line.split(" ") if not (string in ["", "\n"])]
        ]
        dates += [items[0]]
        for component in pole_motion:
            sup_index = 0 if component == "m_1" else 2
            mean_pole = MEAN_POLE_COEFFICIENTS[
                load_model_parameters.history.pole.mean_pole_convention
            ][component][1] * (
                dates[-1] - MEAN_POLE_T_0[load_model_parameters.history.pole.mean_pole_convention]
            )
            pole_motion[component] += [
                (
                    mean_pole
                    if load_model_parameters.history.pole.ramp
                    else items[i + sup_index]
                    + (
                        0.0
                        if load_model_parameters.history.pole.case == "mean"
                        else (
                            -items[i + 1 + sup_index] / 2.0
                            if load_model_parameters.history.pole.case == "lower"
                            else items[i + 1 + sup_index] / 2.0
                        )
                    )
                    - (mean_pole if load_model_parameters.history.pole.remove_mean_pole else 0.0)
                )
            ]

    polar_time_series_dates = numpy.array(object=dates, dtype=float)
    m_1 = numpy.array(object=pole_motion["m_1"], dtype=float) - pole_motion["m_1"][0]
    m_2 = numpy.array(object=pole_motion["m_2"], dtype=float) - pole_motion["m_2"][0]

    if load_model_parameters.history.pole.filter_wobble:
        m_1, m_2, polar_time_series_dates = filter_wobble(
            m_1=m_1,
            m_2=m_2,
            polar_time_series_dates=polar_time_series_dates,
            load_model_parameters=load_model_parameters,
        )
    return (
        polar_time_series_dates,
        MILLI_ARC_SECOND_TO_RADIANS * m_1,
        MILLI_ARC_SECOND_TO_RADIANS * m_2,
    )


def redefine_n_max(n_max: int, grid_or_harmonics: numpy.ndarray[float]) -> int:
    """
    Gets maximal number of degees, limited by grid or harmonics size.
    """

    if len(grid_or_harmonics.shape) == 3:
        return min(n_max, len(grid_or_harmonics[0]) - 1)
    return min(n_max, (len(grid_or_harmonics) - 1) // 2 - 1)


def make_grid(harmonics: numpy.ndarray[float], n_max: int) -> numpy.ndarray[float]:
    """
    From C/S spherical harmonics (2, n_max + 1, n_max + 1)
    to grid (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1).
    """

    result: DHRealGrid = SHCoeffs.from_array(harmonics, lmax=n_max).expand(extend=True, lmax=n_max)
    return result.data


def harmonics_sampling_to_grid(harmonics: numpy.ndarray[float], n_max: int) -> numpy.ndarray[float]:
    """
    Redefines a grid (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1) from potentially higher definition
    C/S spherical harmonics.
    """

    return make_grid(harmonics=harmonics[:, : n_max + 1, : n_max + 1], n_max=n_max)


def make_harmonics(grid: numpy.ndarray[float], n_max: int) -> numpy.ndarray[float]:
    """
    Defines C/S harmonics (2, n_max + 1, n_max + 1) from a grid.
    """

    result: SHRealCoeffs | SHComplexCoeffs = SHGrid.from_array(array=grid).expand(lmax_calc=n_max)
    return result.coeffs


def grid_sampling(grid: numpy.ndarray[float], n_max: int) -> numpy.ndarray[float]:
    """
    Redefines a grid (2 * (n_max + 1) + 1, 4 * (n_max + 1) + 1) from potentially higher definition
    grid.
    """

    return harmonics_sampling_to_grid(harmonics=make_harmonics(grid=grid, n_max=n_max), n_max=n_max)


def load_grace_file(file: Path) -> numpy.ndarray[float]:
    """
    Opens a GRACE (.xyz) or (.csv) datafile.
    """

    # Gets raw data.
    df = pandas.read_csv(
        filepath_or_buffer=file,
        skiprows=(SKIPROWS[None] if file.name not in SKIPROWS else SKIPROWS[file.name]),
        sep=";",
    )

    # Converts to array.
    grid = GRACE_DATA_UNIT_FACTOR * numpy.flip(
        m=[df["EWH"][df["lat"] == lat] for lat in numpy.unique(ar=df["lat"])],
        axis=0,
    )

    longitudes = numpy.unique(df["lon"])

    # Eventually rearanges.
    if (
        "TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv" not in file.name
        and "GRACE_MSSA_2003_2022.xyz" not in file.name
    ):

        indices = numpy.arange(len(longitudes))
        grid = numpy.concatenate(
            (grid[:, indices[longitudes >= 0]], grid[:, indices[longitudes < 0]]), axis=1
        )

    return grid


def format_grace_name_to_date(name: str) -> float:
    """
    Transforms GRACE level 3 solution filename into date
    """

    year_and_mounth = "_".join(name.split("_")[2:]).replace(".xyz", "").split("_")
    year = float(year_and_mounth[0])
    mounth = float(year_and_mounth[1])
    return year + (mounth - 1.0) / 12


def load_grace_solutions(path: Path) -> tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Returns the time-dependent monthly solutions.
    """

    files = list(path.glob("*xyz"))
    times = [format_grace_name_to_date(name=file.name) for file in files]
    solutions = [load_grace_file(file=file) for file in files]
    indices = numpy.argsort(a=times)
    return numpy.array(object=[times[index] for index in indices]), numpy.array(
        object=[solutions[index] for index in indices]
    )


def get_trend_dates(
    dates: numpy.ndarray[float] | list[float],
    load_model_parameters: LoadModelParameters,
    recent_trend: bool = True,
) -> tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Returns trend indices and trend dates. Works for 1900 - 2003 trend or 2003 - 2022 depending on
    the boolean value.
    """

    trend_indices = numpy.where(
        (
            dates
            < (
                load_model_parameters.numerical_parameters.last_year_for_recent_trend
                if recent_trend
                else load_model_parameters.numerical_parameters.last_year_for_past_trend
            )
        )
        * (
            dates
            >= (
                load_model_parameters.numerical_parameters.first_year_for_recent_trend
                if recent_trend
                else load_model_parameters.numerical_parameters.first_year_for_past_trend
            )
        )
    )[0]
    return trend_indices, dates[trend_indices]


def compute_grace_trends(
    path: Path, load_model_parameters: LoadModelParameters
) -> numpy.ndarray[float]:
    """
    Computes the trends from time-dependent solutions.
    """

    times, solutions = load_grace_solutions(path=path)
    trend_indices, trend_dates = get_trend_dates(
        dates=times, load_model_parameters=load_model_parameters
    )
    trends = numpy.zeros(shape=solutions.shape[1:])

    for i_latitude in range(len(solutions[0])):

        for i_longitude in range(len(solutions[0, 0])):

            trends[i_latitude, i_longitude], _ = trend(
                trend_dates=trend_dates, signal=solutions[trend_indices, i_latitude, i_longitude]
            )

    return trends


def midpoints(tab: numpy.ndarray) -> numpy.ndarray:
    """
    Computes midpoints. Returns an array shorter by one element than the input.
    """

    return (tab[1:] + tab[:-1]) / 2.0


def load_load_model_harmonic_component(
    load_model_parameters: LoadModelParameters, path: Path = grace_data_path
) -> tuple[
    numpy.ndarray[float],
    numpy.ndarray[float],
    numpy.ndarray[float],
    numpy.ndarray[float],
    numpy.ndarray[float],
]:
    """
    Opens and formats GRACE (.xyz) or (.csv) datafile.
    """

    path = path.joinpath(load_model_parameters.signature.file)

    # Straightforward case: trends file.
    if (
        "xyz" in load_model_parameters.signature.file
        or "csv" in load_model_parameters.signature.file
    ):

        grid = load_grace_file(file=path)

    # Not straightforward case: deduce trends from monthly solutions.
    else:

        grid = compute_grace_trends(path=path, load_model_parameters=load_model_parameters)

    load_model_parameters.signature.n_max = redefine_n_max(
        n_max=load_model_parameters.signature.n_max, grid_or_harmonics=grid
    )
    grid = grid_sampling(grid=grid, n_max=load_model_parameters.signature.n_max)
    latitudes = midpoints(
        tab=numpy.linspace(90, -90, 2 * (load_model_parameters.signature.n_max + 1) + 2)
    )
    longitudes = midpoints(
        tab=numpy.linspace(0, 360, 4 * (load_model_parameters.signature.n_max + 1) + 2)
    )

    # Gets masks.
    ocean_land_mask, ocean_land_buffered_mask = load_masks(
        load_model_parameters=load_model_parameters, latitudes=latitudes, longitudes=longitudes
    )

    # Loads the continents with opposite value, such that global mean is null.
    if load_model_parameters.signature.opposite_load_on_continents:
        grid = grid * ocean_land_mask - (1 - ocean_land_mask) * (
            mean_on_mask(
                mask=ocean_land_mask,
                latitudes=latitudes,
                load_model_parameters=load_model_parameters.signature.n_max,
                grid_or_harmonics=grid,
                signal_threshold=numpy.inf,
            )
            * sum(surface_ponderation(mask=ocean_land_mask, latitudes=latitudes).flatten())
            / sum(surface_ponderation(mask=(1 - ocean_land_mask), latitudes=latitudes).flatten())
        )

    return (
        make_harmonics(grid=grid, n_max=load_model_parameters.signature.n_max),
        latitudes,
        longitudes,
        ocean_land_mask,
        ocean_land_buffered_mask,
    )


def get_continents(name: str) -> GeoDataFrame:
    """
    Gets the Polygonal information of continents.
    """

    continents: GeoDataFrame = read_file(str(masks_data_path.joinpath(name)))
    continents.crs = LAT_LON_PROJECTION
    return continents


def load_masks(
    load_model_parameters: LoadModelParameters,
    latitudes: numpy.ndarray[float],
    longitudes: numpy.ndarray[float],
) -> tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Loads the ocean/land mask and its computes its buffered version.
    """

    ocean_land_geopandas = get_continents(
        name=load_model_parameters.numerical_parameters.continents
    ).to_crs(epsg=EARTH_EQUAL_PROJECTION)
    lon_grid, lat_grid = numpy.meshgrid(longitudes, latitudes)
    gdf = GeoDataFrame(geometry=[Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())])

    # Bufferizes in a convenient projection.
    gdf.set_crs(epsg=LAT_LON_PROJECTION, inplace=True)
    gdf = gdf.to_crs(epsg=EARTH_EQUAL_PROJECTION)
    ocean_land_geopandas_buffered_reprojected: GeoDataFrame = ocean_land_geopandas.buffer(
        load_model_parameters.numerical_parameters.buffer_distance * 1e3
    )
    oceanic_mask = ~gdf.intersects(ocean_land_geopandas.unary_union)
    oceanic_mask_buffered = ~gdf.intersects(ocean_land_geopandas_buffered_reprojected.unary_union)
    ocean_land_mask = oceanic_mask.to_numpy().reshape(lon_grid.shape).astype(int)
    ocean_land_buffered_mask = oceanic_mask_buffered.to_numpy().reshape(lon_grid.shape).astype(int)

    # Ensures that the mask is binary (land = 0, ocean = 1).
    return (
        grid_sampling(grid=ocean_land_mask, n_max=load_model_parameters.signature.n_max) > 0.5,
        grid_sampling(grid=ocean_land_buffered_mask, n_max=load_model_parameters.signature.n_max)
        > 0.5,
    )


def mean_on_mask(
    mask: numpy.ndarray[float],
    latitudes: numpy.ndarray[float],
    load_model_parameters: LoadModelParameters,
    grid_or_harmonics: numpy.ndarray[float],
    signal_threshold: Optional[float] = None,
) -> float:
    """
    Computes mean value over a given surface. Uses a given mask.
    """

    grid = (
        grid_or_harmonics
        if len(grid_or_harmonics.shape) == 2
        else make_grid(harmonics=grid_or_harmonics, n_max=load_model_parameters.signature.n_max)
    )
    surface = surface_ponderation(
        mask=mask
        * (
            numpy.abs(grid)
            < (
                signal_threshold
                if signal_threshold
                else load_model_parameters.numerical_parameters.signal_threshold
            )
        ),
        latitudes=latitudes,
    )
    return numpy.round(
        a=sum(numpy.array(object=grid * surface).flatten()) / sum(surface.flatten()),
        decimals=MASK_DECIMALS,
    )


def generate_full_dates(
    load_model_parameters: LoadModelParameters, dt: float
) -> numpy.ndarray[float]:
    """
    Generates indices for past trend, recent trend, perios and full model dates.
    """

    return numpy.arange(
        start=load_model_parameters.numerical_parameters.initial_plateau_date,
        stop=load_model_parameters.numerical_parameters.initial_plateau_date
        + 2.0
        * (
            load_model_parameters.numerical_parameters.last_year_for_recent_trend
            + 1.0
            - load_model_parameters.numerical_parameters.initial_plateau_date
            + load_model_parameters.numerical_parameters.spline_time_years
        ),
        step=dt,
    )


def generate_full_signal(
    load_model_parameters: LoadModelParameters,
    full_dates: numpy.ndarray[float],
    anti_symmetric_signal: numpy.ndarray[float],
    target_full_dates: Optional[numpy.ndarray[float]] = None,
) -> tuple[
    numpy.ndarray[int],
    numpy.ndarray[int],
    numpy.ndarray[float],
    numpy.ndarray[float],
    numpy.ndarray[float],
]:
    """
    Generates indices for past trend, recent trend, perios, full model dates and the interpolated
    signal.
    """

    full_load_model_dates = numpy.linspace(
        start=full_dates[0],
        stop=full_dates[-1],
        num=(
            int(
                2.0
                ** (
                    load_model_parameters.numerical_parameters.anti_Gibbs_effect_factor
                    + round(numpy.ceil(numpy.log2(len(full_dates))))
                )
            )
            if target_full_dates is None
            else len(target_full_dates)
        ),
    )
    past_trend_indices, _ = get_trend_dates(
        dates=full_load_model_dates,
        load_model_parameters=load_model_parameters,
        recent_trend=False,
    )
    recent_trend_indices, _ = get_trend_dates(
        dates=full_load_model_dates, load_model_parameters=load_model_parameters
    )

    with numpy.errstate(divide="ignore"):

        periods = 1.0 / fftfreq(
            n=len(full_load_model_dates), d=full_load_model_dates[1] - full_load_model_dates[0]
        )

    return (
        past_trend_indices,
        recent_trend_indices,
        periods,
        full_load_model_dates,
        interpolate.splev(
            x=full_load_model_dates,
            tck=interpolate.splrep(x=full_dates, y=anti_symmetric_signal, k=1),
        ),
    )


def generate_anti_symmetric_signal_model(
    load_model_parameters: LoadModelParameters,
    dates: numpy.ndarray[float],
    signal: numpy.ndarray[float],
) -> tuple[numpy.ndarray[float], numpy.ndarray[float], float, float]:
    """
    Generates an anti-symmetric signal model with initial zero-value plateau and central cubic
    interpolation spline for contintuity.
    """

    dt = dates[1] - dates[0]
    full_dates = generate_full_dates(
        load_model_parameters=load_model_parameters,
        dt=dt,
    )

    # Gets the trends.
    past_trend_indices, _ = get_trend_dates(
        dates=dates, load_model_parameters=load_model_parameters, recent_trend=False
    )
    recent_trend_indices, _ = get_trend_dates(
        dates=dates, load_model_parameters=load_model_parameters
    )
    past_trend, _ = trend(trend_dates=dates[past_trend_indices], signal=signal[past_trend_indices])
    recent_trend, _ = trend(
        trend_dates=dates[recent_trend_indices], signal=signal[recent_trend_indices]
    )

    # Initializes.
    anti_symmetric_signal = numpy.zeros(shape=full_dates.shape)

    # Gets the main signal.
    signal_start_index = numpy.where(full_dates >= load_model_parameters.history.start_date)[0][0]
    signal_stop_index = signal_start_index + len(signal)
    anti_symmetric_signal[signal_start_index:signal_stop_index] = signal

    # LIA.
    if load_model_parameters.history.lia.use:
        start_index = numpy.where(
            full_dates
            >= load_model_parameters.history.lia.end_date
            - load_model_parameters.history.lia.time_years
        )[0][0]
        stop_index = numpy.where(full_dates >= load_model_parameters.history.lia.end_date)[0][0]
        lia_amplitude = (
            past_trend
            * load_model_parameters.history.lia.amplitude_effect
            * load_model_parameters.history.lia.time_years
        )
        anti_symmetric_signal[stop_index:] -= lia_amplitude
        anti_symmetric_signal[start_index:stop_index] = numpy.linspace(
            start=0.0,
            stop=-lia_amplitude,
            num=stop_index - start_index,
        )

    # Eventually extends.
    stop_index = numpy.where(
        full_dates >= load_model_parameters.numerical_parameters.last_year_for_recent_trend + 1.0
    )[0][0]
    anti_symmetric_signal[signal_stop_index:stop_index] = numpy.linspace(
        start=anti_symmetric_signal[signal_stop_index - 1],
        stop=anti_symmetric_signal[signal_stop_index - 1]
        + recent_trend * (stop_index - signal_stop_index) * dt,
        num=stop_index - signal_stop_index,
    )

    # Anti-symmetries.
    anti_symmetric_signal[-stop_index:] = -numpy.flip(anti_symmetric_signal[:stop_index])

    # Interpolates with a cubic spline for continuity.
    anti_symmetric_signal[stop_index - 1 : -stop_index + 1] = CubicSpline(
        x=numpy.array(object=[full_dates[stop_index - 1], full_dates[-stop_index + 1]]),
        y=numpy.array(
            object=[anti_symmetric_signal[stop_index - 1], -anti_symmetric_signal[stop_index - 1]]
        ),
        bc_type=((1, 0.0), (1, 0.0)),
    )(x=full_dates[stop_index - 1 : -stop_index + 1])

    return full_dates, anti_symmetric_signal, past_trend, recent_trend
