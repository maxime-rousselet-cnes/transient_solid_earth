"""
Custom RK45 numerical integration scheme thread-safe and multiprocess-safe.
"""

from typing import Callable

import numpy

from .parameters import SolidEarthIntegrationNumericalParameters

# Dormand-Prince coefficients for RK45.
DOPRI_C = numpy.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
DOPRI_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
DOPRI_B = numpy.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
DOPRI_B_ALT = numpy.array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)


def compute_stages(
    fun: Callable[[float, numpy.ndarray], numpy.ndarray],
    time: float,
    step: float,
    y: numpy.ndarray,
) -> list[numpy.ndarray]:
    """
    Compute intermediate RK45 stages using Dormand-Prince coefficients.
    """

    stages = []
    for i in range(7):
        t_i = time + DOPRI_C[i] * step
        y_i = y.copy()
        for j, stage in enumerate(stages[:i]):
            y_i += step * DOPRI_A[i][j] * stage
        stage = fun(t_i, y_i)
        if not numpy.all(numpy.isfinite(stage)):
            raise ValueError(f"Stage {i} produced non-finite values: {stage}")
        stages.append(stage)
    return stages


def estimate_solution(
    y: numpy.ndarray, stages: list[numpy.ndarray], step: float, b_coeffs: numpy.ndarray
) -> numpy.ndarray:
    """
    Estimate solution using given Butcher tableau weights.
    """

    y_estimate = y.copy()
    for i, stage in enumerate(stages):
        y_estimate += step * b_coeffs[i] * stage
    return y_estimate


def compute_error_ratio(
    y: numpy.ndarray,
    y_high: numpy.ndarray,
    y_low: numpy.ndarray,
    rtol: float,
    atol: float,
) -> float:
    """
    Compute RK45 error ratio for adaptive step size control.
    """

    scale = atol + rtol * numpy.maximum(numpy.abs(y), numpy.abs(y_high))
    return numpy.max(numpy.abs(y_high - y_low) / scale)


def runge_kutta_45(
    fun: Callable[[float, numpy.ndarray], numpy.ndarray],
    t_0: float,
    t_end: float,
    y_0: numpy.ndarray,
    integration_parameters: SolidEarthIntegrationNumericalParameters,
) -> numpy.ndarray:
    """
    Adaptive Runge-Kutta-Fehlberg (RK45) ODE solver with overflow and instability protection.
    """

    time = t_0
    step = (t_end - t_0) / 100
    max_step = (t_end - t_0) / 2

    y = y_0.astype(numpy.complex128 if numpy.iscomplexobj(y_0) else numpy.float64)

    while time < t_end:

        if time + step > t_end:
            step = t_end - time

        stages = compute_stages(fun, time, step, y)

        y_high = estimate_solution(y, stages, step, DOPRI_B)
        y_low = estimate_solution(y, stages, step, DOPRI_B_ALT)

        if not numpy.all(numpy.isfinite(y_high)):
            raise OverflowError(f"y_high overflowed at time={time}, step={step}, y={y}")

        error_ratio = compute_error_ratio(
            y, y_high, y_low, integration_parameters.rtol, integration_parameters.atol
        )

        if numpy.isnan(error_ratio) or numpy.isinf(error_ratio):
            raise RuntimeError(f"Unstable integration: error_ratio={error_ratio}")

        if error_ratio <= 1.0:
            time += step
            y = y_high

        step_factor = (
            0.9 * error_ratio ** (-0.25) if error_ratio > integration_parameters.atol else 2.0
        )
        step = min(step * min(4.0, max(0.1, step_factor)), max_step)

    return y
