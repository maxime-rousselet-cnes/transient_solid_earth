"""
Defines the polar tide correction.
"""

import numpy
from scipy.fft import fft, ifft

from .constants import PHI_CONSTANT, STOKES_TO_EWH_CONSTANT, BoundaryCondition, Direction
from .elastic_load_models import ElasticLoadModel
from .trends import get_trend_from_complex_signal


def polar_motion_correction(
    m_1: numpy.ndarray[float],
    m_2: numpy.ndarray[float],
    love_numbers: numpy.ndarray[float],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Computes polar tide correction time series coherent with the given rheological model.
    """

    frequencial_m1: numpy.ndarray[complex] = fft(m_1)
    frequencial_m2: numpy.ndarray[complex] = fft(m_2)

    # Gets element in position 1 for degree 2. Solid Earth (SE) Polar Tide (PT).
    phi_se_pt_complex: numpy.ndarray[complex] = (
        -PHI_CONSTANT
        * love_numbers[:, 1, BoundaryCondition.POETENTIAL.value, Direction.POTENTIAL.value]
        * (frequencial_m1 - 1.0j * frequencial_m2)
    )

    # C_PT_SE_2_1, S_PT_SE_2_1.
    stokes_to_ewh_factor = STOKES_TO_EWH_CONSTANT / (
        1.0 + love_numbers[:, 1, BoundaryCondition.LOAD.value, Direction.POTENTIAL.value]
    )  # Divides by 1 + k'.

    coherent_polar_motion = numpy.array(object=ifft(phi_se_pt_complex), dtype=numpy.complex64)

    return (
        -stokes_to_ewh_factor * fft(coherent_polar_motion.real),  # C_2_1 frequencial correction.
        stokes_to_ewh_factor * fft(coherent_polar_motion.imag),  # S_2_1 frequencial correction.
    )


def elastic_polar_tide_correction_back(
    elastic_load_model: ElasticLoadModel, elastic_love_numbers: numpy.ndarray
) -> None:
    """
    Performs back the IERS elastic polar tide correction for future self-coherent time-dependent
    correction.
    """

    c_2_1_elastic_polar_tide, s_2_1_elastic_polar_tide = polar_motion_correction(
        m_1=elastic_load_model.side_products.time_dependent_m_1,
        m_2=elastic_load_model.side_products.time_dependent_m_2,
        love_numbers=elastic_love_numbers,
    )
    base_products = elastic_load_model.base_products
    base_products.load_model_harmonic_component[2, 1] += get_trend_from_complex_signal(
        signal=c_2_1_elastic_polar_tide,
        elastic_load_model=elastic_load_model,
    )
    base_products.load_model_harmonic_component[-3, -2] += get_trend_from_complex_signal(
        signal=s_2_1_elastic_polar_tide,
        elastic_load_model=elastic_load_model,
    )
