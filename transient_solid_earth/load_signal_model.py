"""
Needed classes to describe a load signal.
"""

import dataclasses
from pathlib import Path
from typing import Optional

import numpy
from pydantic import BaseModel, model_validator

from .database import load_base_model, save_base_model
from .parameters import DEFAULT_LOAD_MODEL_PARAMETERS, LoadModelParameters
from .paths import elastic_load_models_path


class ElasticLoadModelSpatialProducts(BaseModel):
    """
    Spatial products of an elastic load model, needed for anelastic re-estimation.
    """

    latitudes: numpy.ndarray | list = numpy.zeros(shape=())
    longitudes: Optional[numpy.ndarray | list] = None
    ocean_land_mask: numpy.ndarray | list = numpy.zeros(shape=())
    ocean_land_buffered_mask: numpy.ndarray | list = numpy.zeros(shape=())

    @dataclasses.dataclass
    class Config:
        """
        To authorize arrays.
        """

        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_arrays(self):
        """
        To authorize reccursion.
        """

        self.latitudes = numpy.array(self.latitudes)
        self.longitudes = None if self.longitudes is None else numpy.array(self.longitudes)
        self.ocean_land_mask = numpy.array(self.ocean_land_mask)
        self.ocean_land_buffered_mask = numpy.array(self.ocean_land_buffered_mask)
        return self


class ElasticLoadModelTemporalProducts(BaseModel):
    """
    Temporal products of an elastic load model, needed for anelastic re-estimation.
    """

    full_load_model_dates: numpy.ndarray | list = numpy.zeros(shape=())
    target_past_trend: float = 0.0
    periods: numpy.ndarray | list = numpy.zeros(shape=())

    @dataclasses.dataclass
    class Config:
        """
        To authorize arrays.
        """

        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_arrays(self):
        """
        To authorize reccursion.
        """

        self.full_load_model_dates = numpy.array(self.full_load_model_dates)
        self.periods = numpy.array(self.periods)
        return self


class ElasticLoadModelBaseProducts(BaseModel):
    """
    Base products of an elastic load model, needed for anelastic re-estimation.
    """

    elastic_load_model_temporal_products: ElasticLoadModelTemporalProducts = (
        ElasticLoadModelTemporalProducts()
    )
    load_model_harmonic_component: numpy.ndarray | list = numpy.zeros(shape=())
    # (yr) := (mm) / (mm/yr).
    time_dependent_component: numpy.ndarray | list = numpy.zeros(shape=())

    @dataclasses.dataclass
    class Config:
        """
        To authorize arrays.
        """

        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_arrays(self):
        """
        To authorize reccursion.
        """

        self.load_model_harmonic_component = numpy.array(self.load_model_harmonic_component)
        self.time_dependent_component = numpy.array(self.time_dependent_component)
        return self


class ElasticLoadModelSideProducts(BaseModel):
    """
    Side products of an elastic load model, needed for anelastic re-estimation.
    """

    past_trend_indices: numpy.ndarray | list = numpy.zeros(shape=())
    recent_trend_indices: numpy.ndarray | list = numpy.zeros(shape=())
    time_dependent_m_1: numpy.ndarray | list = numpy.zeros(shape=())
    time_dependent_m_2: numpy.ndarray | list = numpy.zeros(shape=())

    @dataclasses.dataclass
    class Config:
        """
        To authorize arrays.
        """

        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_arrays(self):
        """
        To authorize reccursion.
        """

        self.past_trend_indices = numpy.array(self.past_trend_indices)
        self.recent_trend_indices = numpy.array(self.recent_trend_indices)
        self.time_dependent_m_1 = numpy.array(self.time_dependent_m_1)
        self.time_dependent_m_2 = numpy.array(self.time_dependent_m_2)
        return self


class ElasticLoadModel(BaseModel):
    """
    Defines a preprocessed elastic load, ready for anelastic re-estimation.
    """

    elastic_load_model_spatial_products: ElasticLoadModelSpatialProducts = (
        ElasticLoadModelSpatialProducts()
    )
    elastic_load_model_base_products: ElasticLoadModelBaseProducts = ElasticLoadModelBaseProducts()
    elastic_load_model_side_products: ElasticLoadModelSideProducts = ElasticLoadModelSideProducts()
    load_model_parameters: LoadModelParameters = DEFAULT_LOAD_MODEL_PARAMETERS

    @dataclasses.dataclass
    class Config:
        """
        To authorize arrays.
        """

        arbitrary_types_allowed = True

    def save(self, path: Path = elastic_load_models_path) -> None:
        """
        Saves a preprocessed elastic load model in a (.JSON) file.
        """

        save_base_model(obj=self, name=self.load_model_parameters.model_id(), path=path)


def load_elastic_load_model(
    model_id: str, path: Path = elastic_load_models_path
) -> ElasticLoadModel:
    """
    Gets a preprocessed elastic load model from (.JSON) file.
    """

    return ElasticLoadModel.model_validate(load_base_model(name=model_id, path=path))
