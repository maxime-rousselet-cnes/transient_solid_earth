"""
Special case to handle time-dependent present-day GRACE/-FO solution.
"""

import numpy
from scipy import interpolate
from scipy.fft import fft, ifft

from .elastic_load_models import ElasticLoadModel
from .formating import load_grace_solutions, make_grid, make_harmonics
from .paths import grace_data_path
from .pole_tide import pole_motion_correction
from .trends import get_ocean_mean_trend, get_trend_from_signal, trend


def generate_time_dependent_elastic_load_model(
    elastic_load_model: ElasticLoadModel,
    elastic_love_numbers: numpy.ndarray,
) -> numpy.ndarray[complex]:
    """
    Uses GRACE/-FO full solution for the recent timespan.
    """

    times, solutions = load_grace_solutions(
        path=grace_data_path.joinpath(elastic_load_model.load_model_parameters.signature.file)
    )
    harmonics = [
        make_harmonics(grid=grid, n_max=elastic_load_model.load_model_parameters.signature.n_max)
        for grid in solutions
    ]
    recent_dates = elastic_load_model.base_products.temporal_products.full_load_model_dates[
        elastic_load_model.side_products.recent_trend_indices
    ]
    recent_harmonics = interpolate.interp1d(x=times, y=harmonics, axis=0)(x=recent_dates)
    c_2_1_elastic_pole_tide, s_2_1_elastic_pole_tide = pole_motion_correction(
        m_1=elastic_load_model.side_products.time_dependent_m_1,
        m_2=elastic_load_model.side_products.time_dependent_m_2,
        love_numbers=elastic_love_numbers,
    )
    recent_harmonics[:, 2, 1] += numpy.real(ifft(c_2_1_elastic_pole_tide))[
        elastic_load_model.side_products.recent_trend_indices
    ]
    recent_harmonics[:, -3, -2] += numpy.real(ifft(s_2_1_elastic_pole_tide))[
        elastic_load_model.side_products.recent_trend_indices
    ]
    time_dependent_load_model = numpy.tensordot(
        # (yr) := (mm) / (mm/yr).
        a=elastic_load_model.base_products.time_dependent_component
        / get_trend_from_signal(
            signal=elastic_load_model.base_products.time_dependent_component,
            elastic_load_model=elastic_load_model,
        ),
        # (mm/yr).
        b=[
            [
                trend(
                    trend_dates=recent_dates,
                    signal=recent_harmonics[:, i, j],
                )[0]
                for j in range(elastic_load_model.load_model_parameters.signature.n_max + 1)
            ]
            for i in range(elastic_load_model.load_model_parameters.signature.n_max + 1)
        ],
        axes=0,
    )
    mean = get_ocean_mean_trend(
        harmonic_load_model_trend=time_dependent_load_model[
            elastic_load_model.side_products.recent_trend_indices
        ][0],
        elastic_load_model=elastic_load_model,
    )
    a, b = trend(
        trend_dates=recent_dates,
        signal=numpy.array(
            object=[
                get_ocean_mean_trend(
                    harmonic_load_model_trend=harmonic_slice,
                    elastic_load_model=elastic_load_model,
                )
                for harmonic_slice in recent_harmonics
            ]
        ),
    )
    recent_harmonics = [
        make_harmonics(
            grid=make_grid(
                harmonics=harmonic_slice,
                n_max=elastic_load_model.load_model_parameters.signature.n_max,
            )
            - (a * recent_dates[0] + b - mean),
            n_max=elastic_load_model.load_model_parameters.signature.n_max,
        )
        for harmonic_slice in recent_harmonics
    ]
    time_dependent_load_model[elastic_load_model.side_products.recent_trend_indices] = (
        recent_harmonics
    )
    time_dependent_load_model[-elastic_load_model.side_products.recent_trend_indices] = (
        -time_dependent_load_model[elastic_load_model.side_products.recent_trend_indices]
    )
    time_dependent_load_model = fft(time_dependent_load_model, axis=0)

    return time_dependent_load_model
