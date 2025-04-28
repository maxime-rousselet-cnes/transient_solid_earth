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


class TestModelRheology(BaseModel):
    """
    Test.
    """

    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    def model_id(self):
        """
        Test.
        """
        return "_".join([str(self.alpha), str(self.beta), str(self.gamma)])


DEFAULT_TEST_MODELS_RHEOLOGY = TestModelRheology()


class TestModel(BaseModel):
    """
    Test.
    """

    model_id: str = ""
    rheology: TestModelRheology = DEFAULT_TEST_MODELS_RHEOLOGY
    real_crust: Optional[bool] = None

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __init__(
        self,
        solid_earth_parameters: Optional[SolidEarthParameters] = None,
        rheology: Optional[TestModelRheology | dict] = None,
        model_id: Optional[str] = None,
        real_crust: Optional[bool] = None,
    ) -> None:

        super().__init__()

        if solid_earth_parameters:
            self.rheology = TestModelRheology()
            self.rheology.alpha = rheology["alpha"]
            self.rheology.beta = rheology["beta"]
            self.rheology.gamma = rheology["gamma"]
            self.model_id = self.rheology.model_id()
            self.real_crust = solid_earth_parameters.model.real_crust
            save_base_model(
                obj=self, name=self.model_id, path=logs_subpaths["test_models"].joinpath("models")
            )

        else:
            self.model_id = model_id
            self.rheology = TestModelRheology(**rheology)
            self.real_crust = real_crust

    def process(self, fixed_parameter: float, variable_parameter: float) -> numpy.ndarray:
        """
        Test.
        """
        s = sigmoid(self.rheology.gamma * (variable_parameter - fixed_parameter))
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
