"""
Formulas involving rheological constants.
"""

from numpy import array, convolve, cumsum, diff, errstate, exp, log, nan_to_num, ndarray, pi, zeros
from scipy import integrate

from .constants import ASYMPTOTIC_MU_RATIO_DECIMALS, SECONDS_PER_YEAR


def frequencies_to_periods(
    frequencies: ndarray | list[float],
) -> ndarray:
    """
    Converts tab from (Hz) to (yr). Also works from (yr) to (Hz).
    """

    return (1.0 / SECONDS_PER_YEAR) / array(object=frequencies)


def mu_0_computing(rho_0: ndarray, v_s: ndarray) -> ndarray:
    """
    Computes real elastic modulus mu given density rho_0 and S wave speed v_s.
    """

    return rho_0 * v_s**2


def lambda_0_computing(rho_0: ndarray, v_p: ndarray, mu_0: ndarray) -> ndarray:
    """
    Computes real elastic modulus lambda given density rho_0, P wave speed v_p and real elastic
    modulus mu.
    """

    return rho_0 * v_p**2 - 2 * mu_0


def g_0_computing(x: ndarray, rho_0: ndarray, g_0_inf: float, spline_number: int) -> ndarray:
    """
    Integrates the internal mass GM to get gravitational acceleration g.
    """

    # Trapezoidal rule integral method for GM = integral(rho_0 G dV).
    g_dv_spherical = 4.0 / 3.0 * diff(x**3)
    mean_rho = convolve(a=rho_0, v=[0.5, 0.5])[1:-1]
    dgm_0 = zeros(shape=spline_number)
    dgm_0[0] = g_0_inf * x[0] ** 2
    dgm_0[1:] = mean_rho * g_dv_spherical
    gm_0 = cumsum(dgm_0)
    g_0 = zeros(shape=spline_number)
    g_0[0] = g_0_inf
    g_0[1:] = gm_0[1:] / x[1:] ** 2  # To avoid division by 0 for first point.
    return g_0


def mu_k_computing(mu_k1: ndarray, c: ndarray, mu_0: ndarray) -> ndarray:
    """
    Computes Kelvin's equivalent elastic modulus given the parameters mu_k1, c, and real elastic
    modulus mu_0.
    """

    return mu_k1 + c * mu_0


def omega_cut_computing(mu: ndarray, eta: ndarray) -> ndarray[complex]:
    """
    Computes cut frequency value given the real elastic modulus mu and viscosity eta. Handles
    infinite viscosities.
    """

    with errstate(divide="ignore"):
        return nan_to_num(x=mu / eta, nan=0.0)


def build_cutting_omegas(variables: dict[str, ndarray[complex]]) -> dict[str, ndarray[complex]]:
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


def m_prime_computing(omega_cut_m: ndarray[complex], omega_j: complex) -> ndarray[complex]:
    """
    Computes m_prime transfert function value given the Maxwell's cut pulsation omega_cut_m, and
    pulsation value omega.
    """

    return omega_cut_m / (omega_cut_m + omega_j)


def b_computing(
    omega_cut_m: ndarray, omega_cut_k: ndarray, omega_cut_b: ndarray, omega_j: complex
) -> ndarray[complex]:
    """
    Computes b transfert function value given the Maxwell's, Kelvin's and Burgers cut frequencies
    omega_cut_m, omega_cut_k and omega_cut_b and pulsation value omega.
    """

    return (omega_j * omega_cut_b) / ((omega_j + omega_cut_k) * (omega_j + omega_cut_m))


def mu_computing(
    mu_complex: ndarray[complex],
    m_prime: ndarray[complex],
    b: ndarray[complex],
) -> ndarray[complex]:
    """
    Computes complex analog mu values, given the complex elastic modulus mu and m_prime and b
    transfert function values at pulsation value omega.
    """

    return mu_complex * (1 - m_prime) / (1 + b)


def tau(tau_log: float) -> float:
    """
    Tau from its log value.
    """

    return exp(tau_log)


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
    variables: dict[str, ndarray],
    omega: float,
    frequency: float,
    frequency_unit: float,
    use_bounded_attenuation_functions: bool,
) -> ndarray[complex]:
    """
    Computes the attenuation function f using parameters omega_m and alpha.
    'omega_m' is a unitless frequency.
    'frequency' is a unitless frequency.
    'omega' is a unitless pulsation.
    """

    if use_bounded_attenuation_functions:
        with errstate(invalid="ignore", divide="ignore"):
            return array(
                object=[
                    (
                        0.0
                        if omega_m <= 0.0 or tau_m <= 0.0
                        else -integrate.quad(
                            func=integrand,
                            a=log(1.0 / omega_m),
                            b=log(tau_m),
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
        high_frequency_domain: ndarray[bool] = frequency >= variables["omega_m"]
        omega_0 = 1.0 / frequency_unit  # (Unitless frequency).
        with errstate(invalid="ignore", divide="ignore"):
            return nan_to_num(  # Alpha or omega_m may be equal to 0.0, meaning no attenuation
                # should be taken into account.
                x=((2.0 / pi) * log(frequency / omega_0) + 1.0j) * high_frequency_domain
                + (
                    (2.0 / pi)
                    * (
                        log(variables["omega_m"] / omega_0)
                        + (1 / variables["alpha"])
                        * (1 - (variables["omega_m"] / frequency) ** variables["alpha"])
                    )
                    + (variables["omega_m"] / frequency) ** variables["alpha"] * 1.0j
                )
                * (1 - high_frequency_domain),
                nan=0.0,
            )


def delta_mu_computing(mu_0: ndarray, q_mu: ndarray, f: ndarray[complex]) -> ndarray[complex]:
    """
    Computes the first order frequency dependent variation from elasticity delta_mu at frequency
    value frequency, given the real elastic modulus mu_0, the elasticicty's quality factor q_mu
    and attenuation function f.
    """

    with errstate(invalid="ignore", divide="ignore"):
        return nan_to_num(  # q_mu may be infinite, meaning no attenuation should be taken into
            # account.
            x=(mu_0 / q_mu) * f,
            nan=0.0,
        )


def find_tau_m(omega_m: float, alpha: float, asymptotic_mu_ratio: float, q_mu: float) -> float:
    """
    Uses asymptotic equation to find tau_m such as
    """

    with errstate(invalid="ignore"):
        return (
            0.0
            if round(number=asymptotic_mu_ratio, ndigits=ASYMPTOTIC_MU_RATIO_DECIMALS) == 1.0
            or alpha == 0.0
            else (alpha * (1.0 - asymptotic_mu_ratio) * q_mu + omega_m ** (-alpha)) ** (1.0 / alpha)
        )
