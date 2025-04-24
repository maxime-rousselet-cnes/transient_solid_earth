"""
Test.
"""

import math

import numpy
from pydantic import BaseModel

from .paths import data_path


def sigmoid(x):
    """
    Test.
    """

    return 1 / (1 + math.exp(-x))


test_models_path = data_path.joinpath("test_models")


class Model(BaseModel):
    """
    Test.
    """

    name: str
    alpha: float
    beta: float
    gamma: float
    x_0: float

    def process(self, t: float) -> numpy.ndarray:
        """
        Test.
        """

        return numpy.array(
            object=[
                self.alpha + self.beta * sigmoid(self.gamma * (2.0**t - self.x_0)),
                self.alpha + 2.0 * self.beta * sigmoid(self.gamma * (2.0**t - self.x_0)),
                2.0 * self.alpha + self.beta * sigmoid(self.gamma * (2.0**t - self.x_0)),
                2.0 * self.alpha + 2.0 * self.beta * sigmoid(self.gamma * (2.0**t - self.x_0)),
            ]
        )
