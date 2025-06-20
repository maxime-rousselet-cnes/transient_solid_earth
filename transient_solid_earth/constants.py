"""
Numerical constants.
"""

from enum import Enum
from typing import Optional

import numpy

from .paths import SolidEarthModelPart

HASH_LENGTH = 10


class Direction(Enum):
    """
    Love numbers directions.
    """

    VERTICAL = 0
    TANGENTIAL = 1
    POTENTIAL = 2


class BoundaryCondition(Enum):
    """
    Love numbers boundary conditions.
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


BARYSTATIC_LOAD_MODEL_FILE_COLUMNS = ["lower", "mean", "upper"]

SKIPROWS = {
    "GRACE_MSSA_2003_2022.xyz": 0,
    "TREND_GRACE(-FO)_MSSA_2003_2022_NoGIA_PELTIER_ICE6G-D.csv": 11,
    None: 10,
}

MEAN_POLE_COEFFICIENTS = {
    "IERS_2018_update": {"m_1": [55.0, 1.677], "m_2": [320.5, 3.460]},  # (mas/yr^index).
}
MEAN_POLE_T_0 = {
    "IERS_2018_update": 2000.0,  # (yr).
}
METERS_TO_MILLIMETERS = 1000
STOKES_TO_EWH_CONSTANT = 5.0 / 3.0 / DENSITY_RATIO * EARTH_RADIUS * METERS_TO_MILLIMETERS

K_2_BASE = 0.3077 + 0.0036j
K_2_PRIME_BASE = 0.30523


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

EMPIRICAL_INTERPOLATION_TIEMOUT_FACTOR = 1e4
