"""
Formulas involving rheological constants.
"""

import numpy
from scipy import integrate

from .constants import ASYMPTOTIC_MU_RATIO_DECIMALS


def mu_0_computing(rho_0: numpy.ndarray, v_s: numpy.ndarray) -> numpy.ndarray:
    """
    Computes real elastic modulus mu given density rho_0 and S wave speed v_s.
    """

    return rho_0 * v_s**2


def lambda_0_computing(
    rho_0: numpy.ndarray, v_p: numpy.ndarray, mu_0: numpy.ndarray
) -> numpy.ndarray:
    """
    Computes real elastic modulus lambda given density rho_0, P wave speed v_p and real elastic
    modulus mu.
    """

    return rho_0 * v_p**2 - 2 * mu_0


def g_0_computing(
    x: numpy.ndarray, rho_0: numpy.ndarray, g_0_inf: float, spline_number: int
) -> numpy.ndarray:
    """
    Integrates the internal mass GM to get gravitational acceleration g.
    """

    # Trapezoidal rule integral method for GM = integral(rho_0 G dV).
    g_dv_spherical = 4.0 / 3.0 * numpy.diff(x**3)
    mean_rho = numpy.convolve(a=rho_0, v=[0.5, 0.5])[1:-1]
    dgm_0 = numpy.zeros(shape=spline_number)
    dgm_0[0] = g_0_inf * x[0] ** 2
    dgm_0[1:] = mean_rho * g_dv_spherical
    gm_0 = numpy.cumsum(dgm_0)
    g_0 = numpy.zeros(shape=spline_number)
    g_0[0] = g_0_inf
    g_0[1:] = gm_0[1:] / x[1:] ** 2  # To avoid division by 0 for first point.
    return g_0


def p_0_computing(
    x: numpy.ndarray, rho_0: numpy.ndarray, g_0: numpy.ndarray, p_0_inf: float, spline_number: int
) -> numpy.ndarray:
    """
    Integrates the static equation to get P(x).
    """

    # Trapezoidal rule integral method for P = C^te - integral(rho_0 g dz).
    dz = numpy.diff(x)
    mean_rho_g = numpy.convolve(a=rho_0 * g_0, v=[0.5, 0.5])[1:-1]
    dp_0 = numpy.zeros(shape=spline_number)
    dp_0[0] = p_0_inf
    dp_0[1:] = -mean_rho_g * dz
    return numpy.cumsum(dp_0)


def mu_k_computing(mu_k1: numpy.ndarray, c: numpy.ndarray, mu_0: numpy.ndarray) -> numpy.ndarray:
    """
    Computes Kelvin's equivalent elastic modulus given the parameters mu_k1, c, and real elastic
    modulus mu_0.
    """

    return mu_k1 + c * mu_0


def omega_cut_computing(mu: numpy.ndarray, eta: numpy.ndarray) -> numpy.ndarray[complex]:
    """
    Computes cut frequency value given the real elastic modulus mu and viscosity eta. Handles
    infinite viscosities.
    """

    with numpy.errstate(divide="ignore"):
        return numpy.nan_to_num(x=mu / eta, nan=0.0)


def build_cutting_omegas(
    variables: dict[str, numpy.ndarray[complex]],
) -> dict[str, numpy.ndarray[complex]]:
    """
    Builds a dictionary containing cut frequencies.
    """

    return {
        "omega_cut_m": omega_cut_computing(
            mu=variables["mu"],
            eta=variables["eta_m"],
        ),
        "omega_cut_k": omega_cut_computing(
            mu=variables["mu_k"],
            eta=variables["eta_k"],
        ),
        "omega_cut_b": omega_cut_computing(
            mu=variables["mu"],
            eta=variables["eta_k"],
        ),
    }


def m_prime_computing(
    omega_cut_m: numpy.ndarray[complex], omega_j: complex
) -> numpy.ndarray[complex]:
    """
    Computes m_prime transfert function value given the Maxwell's cut pulsation omega_cut_m, and
    pulsation value omega.
    """

    return omega_cut_m / (omega_cut_m + omega_j)


def b_computing(
    omega_cut_m: numpy.ndarray,
    omega_cut_k: numpy.ndarray,
    omega_cut_b: numpy.ndarray,
    omega_j: complex,
) -> numpy.ndarray[complex]:
    """
    Computes b transfert function value given the Maxwell's, Kelvin's and Burgers cut frequencies
    omega_cut_m, omega_cut_k and omega_cut_b and pulsation value omega.
    """

    return (omega_j * omega_cut_b) / ((omega_j + omega_cut_k) * (omega_j + omega_cut_m))


def mu_computing(
    mu_complex: numpy.ndarray[complex],
    m_prime: numpy.ndarray[complex],
    b: numpy.ndarray[complex],
) -> numpy.ndarray[complex]:
    """
    Computes complex analog mu values, given the complex elastic modulus mu and m_prime and b
    transfert function values at pulsation value omega.
    """

    return mu_complex * (1 - m_prime) / (1 + b)


def tau(tau_log: float) -> float:
    """
    Tau from its log value.
    """

    return numpy.exp(tau_log)


def y_attenuation_spectrum(tau_log: float, alpha: float) -> float:
    """
    Power law attenuation spectrum.
    """

    return tau(tau_log=tau_log) ** alpha


def denom(tau_log: float, omega: float) -> float:
    """
    Relaxation spectrum denominator.
    """

    return 1.0 + (omega * tau(tau_log=tau_log) * 1.0j)


def integrand(tau_log: float, omega: float, alpha: float) -> float:
    """
    Computes the whole integrand for attenuation spectrum formulation.
    """

    return y_attenuation_spectrum(tau_log=tau_log, alpha=alpha) / denom(
        tau_log=tau_log, omega=omega
    )


def f_attenuation_computing(
    variables: dict[str, numpy.ndarray],
    omega: float,
    frequency: float,
    frequency_unit: float,
    use_bounded_attenuation_functions: bool,
) -> numpy.ndarray[complex]:
    """
    Computes the attenuation function f using parameters omega_m and alpha.
    'omega_m' is a unitless frequency.
    'frequency' is a unitless frequency.
    'omega' is a unitless pulsation.
    """

    if use_bounded_attenuation_functions:
        with numpy.errstate(invalid="ignore", divide="ignore"):
            return numpy.array(
                object=[
                    (
                        0.0
                        if omega_m <= 0.0 or tau_m <= 0.0
                        else -integrate.quad(
                            func=integrand,
                            a=numpy.log(1.0 / omega_m),
                            b=numpy.log(tau_m),
                            args=(omega, alpha),
                            complex_func=True,
                        )[0]
                    )
                    for alpha, omega_m, tau_m in zip(
                        variables["alpha"], variables["omega_m"], variables["tau_m"]
                    )
                ]
            )
    else:
        high_frequency_domain: numpy.ndarray[bool] = frequency >= variables["omega_m"]
        omega_0 = 1.0 / frequency_unit  # (Unitless frequency).
        with numpy.errstate(invalid="ignore", divide="ignore"):
            return numpy.nan_to_num(  # Alpha or omega_m may be equal to 0.0, meaning no attenuation
                # should be taken into account.
                x=((2.0 / numpy.pi) * numpy.log(frequency / omega_0) + 1.0j) * high_frequency_domain
                + (
                    (2.0 / numpy.pi)
                    * (
                        numpy.log(variables["omega_m"] / omega_0)
                        + (1 / variables["alpha"])
                        * (1 - (variables["omega_m"] / frequency) ** variables["alpha"])
                    )
                    + (variables["omega_m"] / frequency) ** variables["alpha"] * 1.0j
                )
                * (1 - high_frequency_domain),
                nan=0.0,
            )


def delta_mu_computing(
    mu_0: numpy.ndarray, q_mu: numpy.ndarray, f: numpy.ndarray[complex]
) -> numpy.ndarray[complex]:
    """
    Computes the first order frequency dependent variation from elasticity delta_mu at frequency
    value frequency, given the real elastic modulus mu_0, the elasticicty's quality factor q_mu
    and attenuation function f.
    """

    with numpy.errstate(invalid="ignore", divide="ignore"):
        return numpy.nan_to_num(  # q_mu may be infinite, meaning no attenuation should be taken
            # into account.
            x=(mu_0 / q_mu) * f,
            nan=0.0,
        )


def find_tau_m(omega_m: float, alpha: float, asymptotic_mu_ratio: float, q_mu: float) -> float:
    """
    Uses asymptotic equation to find tau_m such as
    """

    with numpy.errstate(invalid="ignore"):
        return (
            0.0
            if round(number=asymptotic_mu_ratio, ndigits=ASYMPTOTIC_MU_RATIO_DECIMALS) == 1.0
            or alpha == 0.0
            else (alpha * (1.0 - asymptotic_mu_ratio) * q_mu + omega_m ** (-alpha)) ** (1.0 / alpha)
        )
