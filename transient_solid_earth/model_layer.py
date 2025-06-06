"""
The class describes the radial quantities of a solid earth layer.
"""

import dataclasses
from typing import Callable, Optional

import numpy
from pydantic import BaseModel
from scipy import interpolate
from scipy.interpolate import make_lsq_spline

from .parameters import SolidEarthNumericalParameters
from .runge_kutta_scheme import runge_kutta_45


def high_degree_approximation(
    x: float, n: int, numerical_parameters: SolidEarthNumericalParameters
) -> bool:
    """
    Checks whether to use high degrees approximation or not.
    """

    return x**n < numerical_parameters.integration_parameters.high_degrees_radius_sensibility


def splprep_highdim(x: numpy.ndarray, u: numpy.ndarray, k: int = 3):
    """
    Simplified splprep for idim > 11.
    Assumes s=0, and both x and u are specified.

    Parameters:
        x : array_like, shape (idim, N)
            Input data points to fit.
        u : array_like, shape (N,)
            Parameter values (must be strictly increasing).
        k : int
            Degree of the spline. Default is 3.

    Returns:
        tck : tuple
            (t, c_list, k) — shared knots, list of coefficient arrays, spline degree.
    """

    i_dim, spline_number = x.shape

    # Interior knots — excluding start and end.
    num_internal_knots = max(spline_number - (k + 1), 0)
    if num_internal_knots <= 0:
        raise ValueError("Not enough data points for the requested spline degree.")
    t_internal = numpy.linspace(u[0], u[-1], num_internal_knots + 2)[1:-1]

    # Full knot vector with boundary padding.
    t = numpy.concatenate(([u[0]] * (k + 1), t_internal, [u[-1]] * (k + 1)))

    c_list = [make_lsq_spline(u, x[dim], t, k=k).c for dim in range(i_dim)]
    return (t, c_list, k)


class ModelLayer(BaseModel):
    """
    Defines a numerical model's layer.
    """

    name: Optional[str]
    x_inf: float
    x_sup: float
    splines: dict[
        str, tuple[numpy.ndarray | list[float] | float, numpy.ndarray | list[float] | float, int]
    ] = {}
    y_i_system: Optional[Callable[[float, numpy.ndarray], numpy.ndarray]] = None

    @dataclasses.dataclass
    class Config:
        """
        To authorize arrays.
        """

        arbitrary_types_allowed = True

    def evaluate(
        self,
        x: numpy.ndarray | float,
        variable: str,
        derivative_order: int = 0,
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

        return numpy.array(
            object=interpolate.splev(x=x, tck=self.splines[variable], der=derivative_order)
        )

    def x_profile(self, spline_number: int) -> numpy.ndarray:
        """
        Builds an array of x values in the layer.
        """

        return numpy.linspace(start=self.x_inf, stop=self.x_sup, num=spline_number)

    def update_matrix_splines(
        self,
        a_matrix: numpy.ndarray[complex],
        spline_number: int,
        x: numpy.ndarray[float],
        is_fluid: bool = False,
    ) -> None:
        """
        Creates splines that interpolates the integration system's matrix.
        """

        a_spline_real = splprep_highdim(x=a_matrix.real.reshape(-1, spline_number), u=x)
        a_spline_imag = splprep_highdim(x=a_matrix.imag.reshape(-1, spline_number), u=x)
        self.splines.update({"A_real": a_spline_real, "A_imag": a_spline_imag})
        self.y_i_system = lambda t, y: numpy.dot(  # t as curvilign abscissa.
            a=numpy.array(
                object=self.evaluate(x=t, variable="A_real")
                + self.evaluate(x=t, variable="A_imag") * 1.0j,
                dtype=complex,
            ).reshape((2, 2) if is_fluid else (6, 6)),
            b=y,
        )

    def update_solid_system_matrix(
        self,
        n: int,
        omega: float,
        spline_number: int,
        variables: dict[str, numpy.ndarray[float]],
    ) -> None:
        """
        Defines the solid integration system matrix.
        """

        lambda_complex = variables["lambda"]
        mu_complex = variables["mu"]
        x = variables["x"]
        zeros = numpy.zeros(shape=x.shape)

        # Interpolate variables.
        rho_0 = self.evaluate(x=x, variable="rho_0")
        g_0 = self.evaluate(x=x, variable="g_0")

        #  Smylie (2013).
        dyn_term = -rho_0 * omega**2.0
        n_1 = n * (n + 1.0)
        b = 1.0 / (lambda_complex + 2.0 * mu_complex)
        c = 2.0 * mu_complex * (3.0 * lambda_complex + 2.0 * mu_complex) * b

        # Updates the matrix splines.
        self.update_matrix_splines(
            a_matrix=numpy.array(
                [
                    [
                        -2.0 * lambda_complex * b / x,
                        b,
                        n_1 * lambda_complex * b / x,
                        zeros,
                        zeros,
                        zeros,
                    ],
                    [
                        (-4.0 * g_0 * rho_0 / x) + (2.0 * c / (x**2)) + dyn_term,
                        -4.0 * mu_complex * b / x,
                        n_1 * (rho_0 * g_0 / x - c / (x**2)),
                        n_1 / x,
                        zeros,
                        -rho_0,
                    ],
                    [-1.0 / x, zeros, 1.0 / x, 1.0 / mu_complex, zeros, zeros],
                    [
                        rho_0 * g_0 / x - c / (x**2),
                        -lambda_complex * b / x,
                        (
                            4.0 * n_1 * mu_complex * (lambda_complex + mu_complex) * b
                            - 2.0 * mu_complex
                        )
                        / (x**2)
                        + dyn_term,
                        -3.0 / x,
                        -rho_0 / x,
                        zeros,
                    ],
                    [4.0 * rho_0, zeros, zeros, zeros, zeros, numpy.ones(shape=x.shape)],
                    [zeros, zeros, -4.0 * rho_0 * n_1 / x, zeros, n_1 / (x**2), -2.0 / x],
                ],
                dtype=complex,
            ),
            spline_number=spline_number,
            x=x,
        )

    def update_fluid_system_matrix(
        self, n: int, spline_number: int, x: numpy.ndarray[float]
    ) -> None:
        """
        Defines the fluid integration system matrix.
        """

        # Interpolate variables.
        rho_0 = self.evaluate(x=x, variable="rho_0")
        g_0 = self.evaluate(x=x, variable="g_0")

        # Smylie (2013) Eq.9.42 & 9.43.
        c_1_1 = 4.0 * rho_0 / g_0

        # Updates the matrix splines.
        self.update_matrix_splines(
            a_matrix=numpy.array(
                [
                    [c_1_1, numpy.ones(shape=rho_0.shape)],
                    [(n * (n + 1.0) / x**2) - 16.0 * rho_0 / (g_0 * x), (-2.0 / x) - c_1_1],
                ],
                dtype=complex,
            ),
            spline_number=spline_number,
            x=x,
            is_fluid=True,
        )

    def integrate_y_i_system(
        self, y_i: numpy.ndarray, n: int, numerical_parameters: SolidEarthNumericalParameters
    ) -> numpy.ndarray:
        """
        Integrates the Y_i system from the bottom to the top of the layer.
        """

        # Numerical approximation for high degrees.
        if high_degree_approximation(x=self.x_sup, n=n, numerical_parameters=numerical_parameters):
            return y_i

        return runge_kutta_45(
            fun=self.y_i_system,
            t_0=self.x_inf,
            t_end=self.x_sup,
            y_0=y_i,
            integration_parameters=numerical_parameters.integration_parameters,
        )

    def solid_to_fluid(
        self,
        y_1: numpy.ndarray,
        y_2: numpy.ndarray,
        y_3: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Converts the Y_i system solution at a fluid/solid interface.
        To call for the first fluid layer.
        """

        # Interpolates parameters at current radius.
        rho_0 = self.evaluate(x=self.x_inf, variable="rho_0")
        g_0 = self.evaluate(x=self.x_inf, variable="g_0")

        k_1_3 = y_1[3] / y_3[3]
        k_2_3 = y_2[3] / y_3[3]
        k_numerator = (
            g_0 * (y_1[0] + y_3[0] * k_1_3)
            - (y_1[4] + y_3[4] * k_1_3)
            + (1.0 / rho_0) * (y_1[1] + y_3[1] * k_1_3)
        )
        k_denominator = (
            g_0 * (y_2[0] + y_3[0] * k_2_3)
            - (y_2[4] + y_3[4] * k_2_3)
            + (1.0 / rho_0) * (y_2[1] + y_3[1] * k_2_3)
        )
        k_k = k_numerator / k_denominator

        sol_2 = y_1[1] + k_k * y_2[1] + (k_1_3 + k_k * k_2_3) * y_3[1]
        sol_5 = y_1[4] + k_k * y_2[4] + (k_1_3 + k_k * k_2_3) * y_3[4]
        sol_6 = y_1[5] + k_k * y_2[5] + (k_1_3 + k_k * k_2_3) * y_3[5]

        return numpy.array([sol_5, sol_6 + (4.0 / g_0) * sol_2], dtype=complex)

    def fluid_to_solid(
        self, yf_1: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Converts the Y_i system solution at a fluid/solid interface.
        To call for the last fluid layer.
        """

        # Interpolates parameters at current radius.
        rho_0 = self.evaluate(x=self.x_sup, variable="rho_0")
        g_0 = self.evaluate(x=self.x_sup, variable="g_0")

        return (
            numpy.array([1.0, g_0 * rho_0, 0.0, 0.0, 0.0, -4.0 * rho_0], dtype=complex),
            numpy.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex),
            numpy.array([yf_1[0] / g_0, 0.0, 0.0, 0.0, yf_1[0], yf_1[1]], dtype=complex),
        )

    def surface_solution(
        self,
        n: int,
        y_1_s: numpy.ndarray,
        y_2_s: numpy.ndarray,
        y_3_s: numpy.ndarray,
    ) -> numpy.ndarray[float]:
        """
        Returns load Love numbers from the Y_i system solution at Earth surface.
        To call for the very last layer.
        """

        g_0 = self.evaluate(x=self.x_sup, variable="g_0")
        surface_factor_1 = (2.0 * n + 1.0) * g_0
        surface_factor_2 = surface_factor_1 * g_0 / 4.0

        # Forms the outer surface vectors. See Okubo & Saito (1983), Saito (1978).
        d_marix_load = numpy.array([[-surface_factor_2, 0.0, surface_factor_1]])
        d_matrix_shear = numpy.array([[0.0, surface_factor_2 / (n * (n + 1)), 0.0]])
        d_matrix_potential = numpy.array([[0.0, 0.0, surface_factor_1]])

        # Forms the G matrix from integrated solutions.
        g_matrix = numpy.array(
            [
                [y_1_s[1], y_2_s[1], y_3_s[1]],
                [y_1_s[3], y_2_s[3], y_3_s[3]],
                [
                    (y_1_s[5] + (n + 1.0) * y_1_s[4]),
                    (y_2_s[5] + (n + 1.0) * y_2_s[4]),
                    (y_3_s[5] + (n + 1.0) * y_3_s[4]),
                ],
            ]
        )

        # Love numbers.
        love = []

        # Iterates on boundary conditions.
        for d_matrix in [d_marix_load] + ([] if n == 1 else [d_matrix_shear, d_matrix_potential]):

            # Solves the system.
            m_vector = numpy.linalg.solve(g_matrix, d_matrix.T).flatten()

            # Computes solutions.
            love += [
                numpy.dot(numpy.array([y_1_s[0], y_2_s[0], y_3_s[0]]), m_vector),
                numpy.dot(numpy.array([y_1_s[2], y_2_s[2], y_3_s[2]]), m_vector),
                numpy.dot(numpy.array([y_1_s[4], y_2_s[4], y_3_s[4]]), m_vector) / g_0
                - 1.0,  # Because k + 1 is solution.
            ]

        love = numpy.array(love)

        # Transforms to the isomorphic frame (Blewitt, 2003) for which the potential field outside
        # the Earth vanishes (e.g. Merriam 1985).
        if n == 1:
            return numpy.array(  # Substracts k_load from h_load and l_load.
                [[love[0] - love[2], love[1] - love[2], 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )

        love[5] += 1.0  # No shear component on the unperturbed potential.

        return love.reshape((3, 3))
