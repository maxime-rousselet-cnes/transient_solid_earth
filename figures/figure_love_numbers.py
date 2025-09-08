"""
Plots Love numbers for different rheological models.
"""

import numpy
from matplotlib.axes import Axes
from matplotlib.pyplot import show, subplots

from transient_solid_earth import (
    BoundaryCondition,
    Direction,
    interpolated_love_numbers_path,
    load_base_model,
    load_complex_array,
)

from .figures_generation_utils import LONG_TERM_COLORS


def love_numbers_plot(
    labels: dict[str, str],
    periods_id: str,
    degree_index: int = 1,
    direction: Direction = Direction.POTENTIAL,
    boundary_condition: BoundaryCondition = BoundaryCondition.POTENTIAL,
) -> None:
    """
    Shows Love numbers with respect to frequency for a given degree and given models.
    """

    ax1: Axes
    ax2: Axes
    _, (ax1, ax2) = subplots(1, 2, figsize=(10, 3))
    ax1.set_title("Real")
    ax2.set_title("Imag")

    for (model_id, label), color in zip(labels.items(), LONG_TERM_COLORS):

        path = interpolated_love_numbers_path(periods_id=periods_id, rheological_model_id=model_id)
        love_numbers = load_complex_array(path=path)
        periods = numpy.array(object=load_base_model(name="periods", path=path.parent))
        to_plot: numpy.ndarray = love_numbers[
            :, degree_index, boundary_condition.value, direction.value
        ]
        ax1.semilogx(
            periods[(periods > 0) * (periods < 2000)][:-1],
            to_plot.real[(periods > 0) * (periods < 2000)][:-1],
            label=label,
            color=color,
        )
        ax2.semilogx(
            periods[(periods > 0) * (periods < 2000)][:-1],
            to_plot.imag[(periods > 0) * (periods < 2000)][:-1],
            color=color,
        )

    ax1.legend(frameon=False)
    ax1.grid()
    ax2.grid()
    show()
