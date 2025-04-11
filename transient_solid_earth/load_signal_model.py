"""
Needed classes to describe a load signal.
"""

from pathlib import Path

import numpy
from pydantic import BaseModel

from .database import load_base_model
from .parameters import LoadParameters, Parameters
from .paths import input_load_signal_models_path, output_load_signal_trends_path


class InputLoadSignalModel(BaseModel):
    """
    Defines a preprocessed load signal, ready for anelastic re-estimation.
    """

    dates: numpy.ndarray[float]
    frequencies: numpy.ndarray[float]
    signal: numpy.ndarray[float]
    load_parameters: LoadParameters

    def __init__(
        self,
        dates: list[float] | numpy.ndarray[float],
        frequencies: list[float] | numpy.ndarray[float],
        signal: list[list[list[float]]] | numpy.ndarray[float],
        load_parameters: LoadParameters,
    ) -> None:

        super().__init__()

        self.dates = numpy.array(object=dates)
        self.frequencies = numpy.array(object=frequencies)
        self.signal = numpy.array(object=signal)
        self.load_parameters = load_parameters


class OutputLoadSignalTrend(BaseModel):
    """
    Defines a re-estimated signal's trend and the parameters that where needed for its processing.
    """

    parameters: Parameters
    trend: list[list[float]] | numpy.ndarray[float]

    def __init__(
        self,
        parameters: Parameters,
        trend: list[list[float]] | numpy.ndarray[float],
    ):

        super().__init__()

        self.parameters = parameters
        self.trend = numpy.array(object=trend)


def load_input_load_signal_model(
    name: str, path: Path = input_load_signal_models_path
) -> InputLoadSignalModel:
    """
    Gets a preprocessed input load signal model from (.JSON) file.
    """

    return load_base_model(name=name, path=path, base_model_type=InputLoadSignalModel)


def load_output_load_signal_trend(
    name: str, path: Path = output_load_signal_trends_path
) -> OutputLoadSignalTrend:
    """
    Gets a processed output load signal trend from (.JSON) file.
    """

    return load_base_model(name=name, path=path, base_model_type=InputLoadSignalModel)
