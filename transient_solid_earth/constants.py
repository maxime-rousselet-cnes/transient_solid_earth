"""
Numerical constants.
"""

from enum import Enum
from typing import Optional

import numpy
from pydantic import BaseModel

from .paths import SolidEarthModelPart


class Direction(Enum):
    """
    Love numbers / Green functions directions.
    """

    RADIAL = 0
    TANGENTIAL = 1
    POTENTIAL = 2


class BoundaryCondition(Enum):
    """
    Love numbers / Green functions boundary conditions.
    """

    LOAD = 0
    SHEAR = 1
    POETENTIAL = 2


LAT_LON_PROJECTION = 4326
EARTH_EQUAL_PROJECTION = 3857
MASK_DECIMALS = 6

ARC_SECOND_TO_RADIANS = 2 * numpy.pi / (60 * 60 * 360)
MILLI_ARC_SECOND_TO_RADIANS = ARC_SECOND_TO_RADIANS / 1000

# Earth mean radius (m).
EARTH_RADIUS = 6.371e6

EARTH_MASS = 5.972e24  # (kg).

# Ratio betwenn surface water density and mean Earth density.
DENSITY_RATIO = 997.0 / 5513.0

OMEGA = 2.0 * numpy.pi / 86164.0  # (Rad.s^-1)
MEAN_G = 9.8  # (m.s^-2).
PHI_CONSTANT = OMEGA**2 * EARTH_RADIUS / MEAN_G / 15**0.5  # (Unitless).
MEAN_POLE_COEFFICIENTS = {
    "IERS_2018_update": {"m_1": [55.0, 1.677], "m_2": [320.5, 3.460]},  # (mas/yr^index).
}
MEAN_POLE_T_0 = {
    "IERS_2018_update": 2000.0,  # (yr).
}
METERS_TO_MILLIMETERS = 1000
STOKES_TO_EWH_CONSTANT = 5.0 / 3.0 / DENSITY_RATIO * EARTH_RADIUS * METERS_TO_MILLIMETERS

M_1_GIA_TREND = 0.62 * MILLI_ARC_SECOND_TO_RADIANS  # (mas/yr) -> rad.
M_2_GIA_TREND = 3.48 * MILLI_ARC_SECOND_TO_RADIANS  # (mas/yr) -> rad.
K_2_BASE = 0.3077 + 0.0036j
K_2_PRIME_BASE = 0.30523

M_1_TREND = MEAN_POLE_COEFFICIENTS["IERS_2018_update"]["m_1"][1]
M_2_TREND = -MEAN_POLE_COEFFICIENTS["IERS_2018_update"]["m_2"][1]

# Universal Gravitationnal constant (m^3.kg^-1.s^-2).
G = 6.67430e-11

# s.y^-1
SECONDS_PER_YEAR = 365.25 * 86400


def years_to_seconds(period: float) -> float:
    """
    Time unit conversion.
    """

    return SECONDS_PER_YEAR * period


# For integration.
INITIAL_Y_VECTOR = numpy.array(
    object=[
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)

# Default hyper parameters.
DEFAULT_MODELS: dict[Optional[SolidEarthModelPart], Optional[str]] = {
    SolidEarthModelPart.ELASTICITY: "PREM",
    SolidEarthModelPart.LONG_TERM_ANELASTICITY: "uniform",
    SolidEarthModelPart.SHORT_TERM_ANELASTICITY: "uniform",
    None: None,
}
DEFAULT_SPLINE_NUMBER = 10

# Other low level parameters.
ASYMPTOTIC_MU_RATIO_DECIMALS = 5
LAYER_DECIMALS = 5

# (cm/yr) -> (mm/yr).
GRACE_DATA_UNIT_FACTOR = 10


# Lakes and islands to remove from ocean label.
class Rectangle(BaseModel):
    """
    Rectangle area to erase islands or lakes.
    """

    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float


RECTANGLES: dict[str, Rectangle] = {
    "Lake Superior": Rectangle(
        min_latitude=41.375963,
        max_latitude=50.582521,
        min_longitude=-93.748270,
        max_longitude=-75.225322,
    ),
    "Lake Victoria": Rectangle(
        min_latitude=-2.809322,
        max_latitude=0.836983,
        min_longitude=31.207700,
        max_longitude=34.530942,
    ),
    "Caspian Sea": Rectangle(
        min_latitude=35.569650,
        max_latitude=47.844035,
        min_longitude=44.303403,
        max_longitude=60.937192,
    ),
    "Svalbard": Rectangle(
        min_latitude=76.515316,
        max_latitude=80.801823,
        min_longitude=5.435317,
        max_longitude=35.933364,
    ),
    "Southern Georgia": Rectangle(
        min_latitude=-57.496798,
        max_latitude=-52.844035,
        min_longitude=-40.681460,
        max_longitude=-32.937192,
    ),
    "Kerguelen": Rectangle(
        min_latitude=-53.262545,
        max_latitude=-47.572174,
        min_longitude=64.599279,
        max_longitude=72.572817,
    ),
    "Arkhanglesk": Rectangle(
        min_latitude=79.447297,
        max_latitude=82.165567,
        min_longitude=43.404067,
        max_longitude=70.925551,
    ),
    "Krasnoiarsk": Rectangle(
        min_latitude=77.162848,
        max_latitude=81.586588,
        min_longitude=85.326874,
        max_longitude=112.397186,
    ),
}
