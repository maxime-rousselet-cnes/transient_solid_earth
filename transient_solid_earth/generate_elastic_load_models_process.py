"""
Worker to interpolate Love numbers on the same periods.
"""

import numpy

from .database import add_result_to_table, extract_terminal_attributes, is_in_table, load_base_model
from .elastic_load_models import (
    BaseProducts,
    ElasticLoadModel,
    ElasticLoadModelSpatialProducts,
    SideProducts,
    TemporalProducts,
)
from .formating import (
    generate_anti_symmetric_signal_model,
    generate_full_signal,
    load_barystatic_load_model,
    load_load_model_harmonic_component,
    load_polar_motion_time_series,
)
from .parameters import LoadModelParameters
from .paths import elastic_load_model_parameters_subpath
from .worker_parser import WorkerInformation


def get_time_dependent_components(
    load_model_parameters: LoadModelParameters,
) -> tuple[
    numpy.ndarray[int],
    numpy.ndarray[int],
    numpy.ndarray[float],
    numpy.ndarray[float],
    numpy.ndarray[float],
    float,
]:
    """
    Merges two function calls in a single one.
    """

    barystatic_load_model_dates, barystatic_load_model = load_barystatic_load_model(
        load_model_parameters=load_model_parameters
    )
    full_dates, anti_symmetric_load_model, past_trend, recent_trend = (
        generate_anti_symmetric_signal_model(
            load_model_parameters=load_model_parameters,
            dates=barystatic_load_model_dates,
            signal=barystatic_load_model,
        )
    )
    (
        past_trend_indices,
        recent_trend_indices,
        periods,
        full_load_model_dates,
        time_dependent_component,
    ) = generate_full_signal(
        load_model_parameters=load_model_parameters,
        full_dates=full_dates,
        anti_symmetric_signal=anti_symmetric_load_model,
    )
    return (
        periods,
        full_load_model_dates,
        time_dependent_component / recent_trend,  # (yr) := (mm) / (mm/yr).
        past_trend,
        past_trend_indices,
        recent_trend_indices,
    )


def get_time_dependent_m(
    load_model_parameters: LoadModelParameters,
    polar_motion_dates: numpy.ndarray[float],
    m: numpy.ndarray[float],
    target_full_dates: numpy.ndarray[float],
) -> numpy.ndarray[float]:
    """
    Merges two function calls in a single one.
    """

    full_dates, anti_symmetric_m, _, _ = generate_anti_symmetric_signal_model(
        load_model_parameters=load_model_parameters,
        dates=polar_motion_dates,
        signal=m,
    )
    (
        _,
        _,
        _,
        _,
        time_dependent_m,
    ) = generate_full_signal(
        load_model_parameters=load_model_parameters,
        full_dates=full_dates,
        anti_symmetric_signal=anti_symmetric_m,
        target_full_dates=target_full_dates,
    )
    return time_dependent_m


def generate_time_dependent_products(
    load_model_parameters: LoadModelParameters,
    polar_motion_dates: numpy.ndarray[float],
    m_1: numpy.ndarray[float],
    m_2: numpy.ndarray[float],
) -> tuple[TemporalProducts, numpy.ndarray[float], SideProducts]:
    """
    Generates the period-dependent components of the elastic load model.
    """

    (
        periods,
        full_load_model_dates,
        time_dependent_component,
        past_trend,
        past_trend_indices,
        recent_trend_indices,
    ) = get_time_dependent_components(load_model_parameters=load_model_parameters)
    time_dependent_m_1 = get_time_dependent_m(
        load_model_parameters=load_model_parameters,
        polar_motion_dates=polar_motion_dates,
        m=m_1,
        target_full_dates=full_load_model_dates,
    )
    time_dependent_m_2 = get_time_dependent_m(
        load_model_parameters=load_model_parameters,
        polar_motion_dates=polar_motion_dates,
        m=m_2,
        target_full_dates=full_load_model_dates,
    )

    return (
        TemporalProducts(
            full_load_model_dates=full_load_model_dates,
            target_past_trend=past_trend,
            periods=periods,
        ),
        time_dependent_component,
        SideProducts(
            past_trend_indices=past_trend_indices,
            recent_trend_indices=recent_trend_indices,
            time_dependent_m_1=time_dependent_m_1,
            time_dependent_m_2=time_dependent_m_2,
        ),
    )


def worker_generate_elastic_load_models(worker_information: WorkerInformation) -> None:
    """
    Generates an elastic load model from its parameters.
    """

    load_model_parameters: LoadModelParameters = load_base_model(
        name=worker_information.model_id,
        path=elastic_load_model_parameters_subpath,
        base_model_type=LoadModelParameters,
    )

    # Spatial products and harmonic component
    (
        load_model_harmonic_component,
        latitudes,
        longitudes,
        ocean_land_mask,
        ocean_land_buffered_mask,
    ) = load_load_model_harmonic_component(load_model_parameters=load_model_parameters)

    # Polar motion
    polar_motion_dates, m_1, m_2 = load_polar_motion_time_series(
        load_model_parameters=load_model_parameters
    )

    # Period-dependent components
    (
        temporal_products,
        time_dependent_component,
        side_products,
    ) = generate_time_dependent_products(
        load_model_parameters=load_model_parameters,
        polar_motion_dates=polar_motion_dates,
        m_1=m_1,
        m_2=m_2,
    )

    load_model_line = extract_terminal_attributes(obj=load_model_parameters)
    load_model_line["ID"] = load_model_parameters.model_id()

    if not is_in_table(table_name="elastic_load_models", id_to_check=load_model_line["ID"]):

        add_result_to_table(table_name="elastic_load_models", dictionary=load_model_line)

        # Creates the elastic load model.
        ElasticLoadModel(
            elastic_load_model_spatial_products=ElasticLoadModelSpatialProducts(
                latitudes=latitudes,
                longitudes=longitudes,
                ocean_land_mask=ocean_land_mask,
                ocean_land_buffered_mask=ocean_land_buffered_mask,
            ),
            base_products=BaseProducts(
                temporal_products=temporal_products,
                load_model_harmonic_component=load_model_harmonic_component,
                time_dependent_component=time_dependent_component,
            ),
            side_products=side_products,
            load_model_parameters=load_model_parameters,
        ).save()

        # Renames input (.JSON) file.
        elastic_load_model_parameters_subpath.joinpath(
            worker_information.model_id + ".json"
        ).rename(elastic_load_model_parameters_subpath.joinpath(load_model_line["ID"] + ".json"))
