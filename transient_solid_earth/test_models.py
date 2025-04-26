"""
Test.
"""

from typing import Optional

import numpy
from pydantic import BaseModel

from .database import save_base_model
from .parameters import SolidEarthParameters
from .paths import logs_subpaths


def sigmoid(x):
    """
    Test.
    """

    with numpy.errstate(over="ignore"):
        return numpy.nan_to_num(x=1 / (1 + numpy.exp(-x)), nan=0.0)


class TestModelsRheology(BaseModel):
    """
    Test.
    """

    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    x_0: float = 0.0

    def model_id(self):
        """
        Test.
        """
        return "_".join([str(self.alpha), str(self.beta), str(self.gamma), str(self.x_0)])


DEFAULT_TEST_MODELS_RHEOLOGY = TestModelsRheology()


class TestModels(BaseModel):
    """
    Test.
    """

    model_id: str = ""
    rheology: TestModelsRheology = DEFAULT_TEST_MODELS_RHEOLOGY
    real_crust: Optional[bool] = None

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __init__(
        self,
        solid_earth_parameters: Optional[SolidEarthParameters] = None,
        rheology: Optional[TestModelsRheology | dict] = None,
        model_id: Optional[str] = None,
        real_crust: Optional[bool] = None,
    ) -> None:

        super().__init__()

        if solid_earth_parameters:
            self.rheology = TestModelsRheology()
            self.rheology.alpha = rheology["alpha"]
            self.rheology.beta = rheology["beta"]
            self.rheology.gamma = rheology["gamma"]
            self.rheology.x_0 = rheology["x_0"]
            self.model_id = self.rheology.model_id()
            self.real_crust = solid_earth_parameters.model.real_crust
            save_base_model(
                obj=self, name=self.model_id, path=logs_subpaths["test_models"].joinpath("models")
            )

        else:
            self.model_id = model_id
            self.rheology = TestModelsRheology(**rheology)
            self.real_crust = real_crust

    def process(self, variable_parameter: float) -> numpy.ndarray:
        """
        Test.
        """
        s = sigmoid(self.rheology.gamma * (variable_parameter - self.rheology.x_0))
        return numpy.array(
            object=[
                [
                    self.rheology.alpha + self.rheology.beta * s,
                    self.rheology.alpha + 2.0 * self.rheology.beta * s,
                ],
                [
                    2.0 * self.rheology.alpha + self.rheology.beta * s,
                    2.0 * self.rheology.alpha + 2.0 * self.rheology.beta * s,
                ],
            ]
        )
