"""
Functions for signal trend computing.
"""

import numpy
from scipy.fft import ifft

from .elastic_load_models import ElasticLoadModel
from .formating import mean_on_mask
from .functions import trend


def get_ocean_mean_trend(
    harmonic_load_model_trend: numpy.ndarray,
    elastic_load_model: ElasticLoadModel,
    buffered: bool = True,
    recent_trend: bool = True,
) -> float:
    """
    Gets mean value over ocean surface.
    """

    numerical_parameters = elastic_load_model.load_model_parameters.numerical_parameters
    return mean_on_mask(
        mask=(
            elastic_load_model.elastic_load_model_spatial_products.ocean_land_buffered_mask
            if buffered
            else elastic_load_model.elastic_load_model_spatial_products.ocean_land_mask
        ),
        latitudes=elastic_load_model.elastic_load_model_spatial_products.latitudes,
        load_model_parameters=elastic_load_model.load_model_parameters,
        grid_or_harmonics=harmonic_load_model_trend,
        ewh_threshold=(  # Eventually overwrites with "ewh_threshold".
            (
                numerical_parameters.mean_ewh_threshold
                if numerical_parameters.mean_ewh_threshold
                else numerical_parameters.ewh_threshold
            )
            if recent_trend
            else (
                numerical_parameters.mean_ewh_threshold_past
                if numerical_parameters.mean_ewh_threshold_past
                else numerical_parameters.ewh_threshold_past
            )
        ),
    )


def get_trend_from_period_dependent_harmonic_model(
    period_dependent_harmonic_model: numpy.ndarray,
    elastic_load_model: ElasticLoadModel,
    recent_trend: bool = True,
) -> numpy.ndarray[float]:
    """
    Gets the trend of a signal expressed in the spherical harmonics domain.
    """

    n_lines = period_dependent_harmonic_model.shape[1]
    return numpy.array(
        object=[
            [
                get_trend_from_signal(
                    signal=numpy.real(val=ifft(period_dependent_harmonic_model[:, i, j])),
                    elastic_load_model=elastic_load_model,
                    recent_trend=recent_trend,
                )
                for j in range(n_lines)
            ]
            for i in range(n_lines)
        ]
    )


def get_trend_from_complex_signal(
    signal: numpy.ndarray, elastic_load_model: ElasticLoadModel, recent_trend: bool = True
) -> float:
    """
    Gets the trend from a Fourier transformed signal.
    """

    return get_trend_from_signal(
        signal=numpy.real(val=ifft(signal)),
        elastic_load_model=elastic_load_model,
        recent_trend=recent_trend,
    )


def get_trend_from_signal(
    signal: numpy.ndarray, elastic_load_model: ElasticLoadModel, recent_trend: bool = True
) -> float:
    """
    Gets the trend from real signal on the wanted timespan.
    """

    trend_indices = (
        elastic_load_model.side_products.recent_trend_indices
        if recent_trend
        else elastic_load_model.side_products.past_trend_indices
    )
    result, _ = trend(
        trend_dates=elastic_load_model.base_products.temporal_products.full_load_model_dates[
            trend_indices
        ],
        signal=signal[trend_indices],
    )
    return result
