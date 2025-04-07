"""
The class describes the radial quantities of a solid earth layer.
"""

from typing import Optional

import numpy
from pydantic import BaseModel
from scipy import interpolate


class DescriptionLayer(BaseModel):
    """
    Defines a layer of a description.
    """

    name: Optional[str]
    x_inf: float
    x_sup: float
    splines: dict[str, tuple[numpy.ndarray | float, numpy.ndarray | float, int]]

    def evaluate(
        self, x: numpy.ndarray | float, variable: str, derivative_order: int = 0
    ) -> numpy.ndarray | float:
        """
        Evaluates some quantity polynomial spline over an array x.
        """

        if not isinstance(self.splines[variable][0], numpy.ndarray):  # Handles constant cases.
            return (
                numpy.inf if self.splines[variable][0] == numpy.inf else self.splines[variable][0]
            ) * numpy.ones(  # Handles infinite cases.
                shape=(numpy.shape(x))
            )

        return interpolate.splev(x=x, tck=self.splines[variable], der=derivative_order)

    def x_profile(self, spline_number: int) -> numpy.ndarray:
        """
        Builds an array of x values in the layer.
        """

        return numpy.linspace(start=self.x_inf, stop=self.x_sup, num=spline_number)
