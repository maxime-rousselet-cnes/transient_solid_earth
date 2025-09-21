"""
2025 paper's supplementary materials.
"""

import numpy
from cartopy.crs import Robinson
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure, figure, setp, subplots, text

from transient_solid_earth import load_base_model

from .figures_data_formater_utils import figures_path, read_csv_and_replace
from .figures_generation_utils import (
    ANELASTICITY_OPTIONS,
    FONTSIZE,
    FONTSIZE_AXE_LABELS,
    FONTSIZE_PANEL_TITLES,
    FONTSIZE_TICKLABELS,
    LABELSIZE,
    LINEWIDTH,
    OPTION_COLORS,
    REFERENCE_MODEL_PARAMETERS,
    REFERENCE_RED,
    SIZE,
    draw_violin_and_boxplot,
    get_reference_values,
    natural_projection,
)


def generate_figure_sup_1(figsize: tuple[float, float] = (6, 4)) -> None:
    """
    2025's article.
    """

    data: dict[str] = load_base_model(name="figure_sup_1", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: Axes = fig.add_axes([0.12, 0.12, 0.8, 0.8])  # [left, bottom, width, height].
    ax1.plot(
        data["dates"],
        data["y_with_lia"],
        color=REFERENCE_RED,
        linestyle="--",
        linewidth=LINEWIDTH,
        label="Simplified LIA Model",
    )
    ax1.plot(data["dates"], data["y_without_lia"], color=REFERENCE_RED, linewidth=LINEWIDTH)
    ax1.yaxis.set_ticks_position("both")
    ax1.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax1.set_xlabel(xlabel="(yr)", fontsize=FONTSIZE_AXE_LABELS)
    ax1.set_ylabel(ylabel="(mm)", fontsize=FONTSIZE_AXE_LABELS)
    ax1.set_yticks(range(0, 101, 25), minor=True)
    ax1.set_xticks(range(1250, 2021, 50), minor=True)
    ax1.set_yticks(range(0, 101, 50))
    ax1.set_xticks(range(1250, 2021, 150))
    ax1.tick_params(axis="both", which="major", length=10, width=1.5)
    ax1.tick_params(axis="both", which="minor", length=5, width=1)
    ax1.text(
        0.2,
        1.02,
        "Mean barystatic sea level model",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )
    ax1.legend(frameon=False, fontsize=LABELSIZE)

    fig.savefig(figures_path.joinpath("figure_sup_1.svg"), format="svg")


def generate_figure_sup_2(
    figsize: tuple[float, float] = (12, 5.6), correct_for_latitudes: bool = False
) -> None:
    """
    2025's article.
    """

    # dates, lower_bound, mean_curb, upper_bound, latitudes, longitudes, mask, grid.
    data = load_base_model(name="figure_sup_2", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: GeoAxes = fig.add_axes([0.05, 0.55, 0.45, 0.45], projection=Robinson(central_longitude=0))
    ax2: GeoAxes = fig.add_axes([0.05, 0.05, 0.45, 0.45], projection=Robinson(central_longitude=0))
    ax3: GeoAxes = fig.add_axes([0.55, 0.55, 0.45, 0.45], projection=Robinson(central_longitude=0))
    ax4: GeoAxes = fig.add_axes([0.55, 0.05, 0.45, 0.45], projection=Robinson(central_longitude=0))

    for solution, ax, letter, saturation_threshold in zip(
        ["CSR", "JPL", "GFZ", "MSSA"],
        [ax1, ax2, ax3, ax4],
        ["A.", "B.", "C.", "D."],
        [50, 5, 5, 20],
    ):

        contour = natural_projection(
            ax=ax,
            data=data,
            grid=data[solution],
            saturation_threshold=saturation_threshold,
        )
        cbar = fig.colorbar(
            contour,
            ax=ax,
            orientation="vertical",
            shrink=0.7,
            ticks=numpy.array(object=[-40, -20, 0, 20, 40]) * saturation_threshold / 50,
            extend="both",
        )
        cbar.ax.tick_params(labelsize=LABELSIZE)
        cbar.set_label(
            label=solution + ("" if solution == "CSR" else " - CSR") + " (mm/yr)",
            fontsize=FONTSIZE,
        )
        ax.text(
            0.11,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_TITLES,
            fontweight="bold",
        )

        if correct_for_latitudes:

            ax.text(
                0.0, 0.85, "60°N", transform=ax.transAxes, fontsize=FONTSIZE, fontweight="regular"
            )
            ax.text(
                0.0, 0.11, "60°S", transform=ax.transAxes, fontsize=FONTSIZE, fontweight="regular"
            )

    fig.savefig(figures_path.joinpath("figure_sup_2.svg"), format="svg", dpi=300)


def generate_figure_sup_3(figsize: tuple[float, float] = (6, 2.5)) -> None:
    """
    2025's article.
    """

    # dates, lower_bound, mean_curb, upper_bound, latitudes, longitudes, mask, grid.
    data = load_base_model(name="figure_sup_3", path=figures_path)
    fig = figure(figsize=figsize)
    width = 0.42
    height = 0.7
    ax1: Axes = fig.add_axes([0.06, 0.1, width, height])
    ax2: Axes = fig.add_axes([0.56, 0.1, width, height])

    ax1.plot(
        data["periods"],
        data["reference"]["potential"]["real"],
        label=r"$k_2 / k_2^{elastic}$",
        color="blue",
    )
    ax1.fill_between(
        data["periods"],
        data["lowest"]["potential"]["real"],
        data["highest"]["potential"]["real"],
        color="blue",
        alpha=0.3,
    )
    ax1.plot(
        data["periods"],
        data["reference"]["load"]["real"],
        label=r"$k'_2 / k'_2^{elastic}$",
        color="orange",
    )
    ax1.fill_between(
        data["periods"],
        data["lowest"]["load"]["real"],
        data["highest"]["load"]["real"],
        color="orange",
        alpha=0.3,
    )
    ax1.text(
        -0.1,
        1.1,
        "A.",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )
    ax1.set_xscale("log")
    ax1.legend(frameon=False, fontsize=LABELSIZE)
    ax1.set_title(
        "Real part",
        fontsize=FONTSIZE_PANEL_TITLES,
    )

    ax2.plot(
        data["periods"],
        data["reference"]["potential"]["imag"],
        label=r"$k_2 / k_2^{elastic}$",
        color="blue",
    )
    ax2.fill_between(
        data["periods"],
        data["lowest"]["potential"]["imag"],
        data["highest"]["potential"]["imag"],
        color="blue",
        alpha=0.3,
    )
    ax2.plot(
        data["periods"],
        data["reference"]["load"]["imag"],
        label=r"$k'_2 / k'_2^{elastic}$",
        color="orange",
    )
    ax2.fill_between(
        data["periods"],
        data["lowest"]["load"]["imag"],
        data["highest"]["load"]["imag"],
        color="orange",
        alpha=0.3,
    )
    ax2.text(
        -0.1,
        1.1,
        "B.",
        transform=ax2.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )
    ax2.set_xscale("log")
    ax2.set_title(
        "Imaginary part",
        fontsize=FONTSIZE_PANEL_TITLES,
    )

    fig.savefig(figures_path.joinpath("figure_sup_3.svg"), format="svg")


def sub_function_figure_sup_4(
    axes: list[GeoAxes], data: dict, ax_left: Axes, ax_right: Axes, fig: Figure
) -> None:
    """
    Sub-function to minimize local variables.
    """

    for solution, title, ax, letter in zip(
        [
            "pre_inversion_degree_one_grid",
            "elastic_post_inversion_degree_one_grid",
            "anelastic_post_inversion_degree_one_grid",
            "d_grid",
            "e_grid",
        ],
        [
            "TN13",
            "Elastic inversion",
            "Anelastic inversion",
            "Elastic - TN13",
            "Anelastic - Elastic",
        ],
        axes,
        ["A.", "B.", "C.", "D. = B. - A.", "E. = C. - B."],
    ):

        contour = natural_projection(
            ax=ax,
            data=data,
            grid=data[solution],
            saturation_threshold=1 if "=" in letter else 5,
        )
        ax.text(
            0.11,
            1.1,
            letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_TITLES,
            fontweight="bold",
        )
        ax.set_title(
            title
            + " :"
            + str(numpy.round(data[solution.replace("grid", "mean")], decimals=3))
            + " (mm/yr)",
            fontsize=FONTSIZE_PANEL_TITLES,
        )

        if letter == "C." or "E" in letter:

            cbar = fig.colorbar(
                contour,
                ax=ax_left if letter == "C." else ax_right,
                orientation="vertical",
                shrink=0.4,
                ticks=numpy.array(object=[-40, -20, 0, 20, 40]) * (1 if "=" in letter else 5) / 50,
                extend="both",
            )
            cbar.ax.tick_params(labelsize=LABELSIZE)
            cbar.set_label(
                label=" (mm/yr)",
                fontsize=FONTSIZE,
            )


def generate_figure_sup_4(figsize: tuple[float, float] = (12, 10)) -> None:
    """
    2025's article.
    """

    # dates, lower_bound, mean_curb, upper_bound, latitudes, longitudes, mask, grid.
    data = load_base_model(name="figure_sup_4", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: GeoAxes = fig.add_axes([0.04, 0.65, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax2: GeoAxes = fig.add_axes([0.04, 0.35, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax3: GeoAxes = fig.add_axes([0.04, 0.05, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax4: GeoAxes = fig.add_axes([0.54, 0.55, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax5: GeoAxes = fig.add_axes([0.54, 0.25, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax_left: Axes = fig.add_axes([0.0, 0.05, 0.5, 0.9], frameon=False)
    ax_left.get_xaxis().set_ticks([])
    ax_left.get_yaxis().set_ticks([])
    ax_right: Axes = fig.add_axes([0.5, 0.05, 0.5, 0.9], frameon=False)
    ax_right.get_xaxis().set_ticks([])
    ax_right.get_yaxis().set_ticks([])

    data["d_grid"] = numpy.array(
        object=data["elastic_post_inversion_degree_one_grid"]
    ) - numpy.array(object=data["pre_inversion_degree_one_grid"])
    data["e_grid"] = numpy.array(
        object=data["anelastic_post_inversion_degree_one_grid"]
    ) - numpy.array(object=data["elastic_post_inversion_degree_one_grid"])
    data["d_mean"] = (
        data["elastic_post_inversion_degree_one_mean"] - data["pre_inversion_degree_one_mean"]
    )
    data["e_mean"] = (
        data["anelastic_post_inversion_degree_one_mean"]
        - data["elastic_post_inversion_degree_one_mean"]
    )

    sub_function_figure_sup_4(
        axes=[ax1, ax2, ax3, ax4, ax5], data=data, ax_left=ax_left, ax_right=ax_right, fig=fig
    )

    fig.savefig(figures_path.joinpath("figure_sup_4.svg"), format="svg", dpi=300)


def sub_function_figure_sup_5(axes: list[GeoAxes], data: dict, ax_bar: Axes, fig: Figure) -> None:
    """
    Sub-function to minimize local variables.
    """

    for solution, ax, letter in zip(
        [
            "elastic_geoid_deformation_grid",
            "elastic_vertical_displacement_grid",
            "anelastic_geoid_deformation_grid",
            "anelastic_vertical_displacement_grid",
            "e_grid",
            "f_grid",
        ],
        axes,
        ["A.", "B.", "C.", "D.", "E. = C. - A.", "F. = D. - B."],
    ):

        contour = natural_projection(
            ax=ax,
            data=data,
            grid=data[solution],
            saturation_threshold=1.5,
        )
        ax.text(
            0.1,
            1.1,
            letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_TITLES,
            fontweight="bold",
        )
        ax.set_title(
            str(numpy.round(data[solution.replace("grid", "mean")], decimals=3)) + " (mm/yr)",
            fontsize=FONTSIZE_PANEL_TITLES,
        )

        if "F" in letter:

            cbar = fig.colorbar(
                contour,
                ax=ax_bar,
                orientation="vertical",
                shrink=0.5,
                ticks=numpy.array(object=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5]),
                extend="both",
            )
            cbar.ax.tick_params(labelsize=LABELSIZE)
            cbar.set_label(
                label=" (mm/yr)",
                fontsize=FONTSIZE,
            )


def generate_figure_sup_5(figsize: tuple[float, float] = (12, 10)) -> None:
    """
    2025's article.
    """

    # dates, lower_bound, mean_curb, upper_bound, latitudes, longitudes, mask, grid.
    data = load_base_model(name="figure_sup_5", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: GeoAxes = fig.add_axes([0.1, 0.65, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax2: GeoAxes = fig.add_axes([0.6, 0.65, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax3: GeoAxes = fig.add_axes([0.1, 0.35, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax4: GeoAxes = fig.add_axes([0.6, 0.35, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax5: GeoAxes = fig.add_axes([0.1, 0.05, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax6: GeoAxes = fig.add_axes([0.6, 0.05, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax_bar: Axes = fig.add_axes([0.06, 0.05, 0.5, 0.9], frameon=False)
    ax_bar.get_xaxis().set_ticks([])
    ax_bar.get_yaxis().set_ticks([])

    data["e_grid"] = numpy.array(object=data["anelastic_geoid_deformation_grid"]) - numpy.array(
        object=data["elastic_geoid_deformation_grid"]
    )
    data["f_grid"] = numpy.array(object=data["anelastic_vertical_displacement_grid"]) - numpy.array(
        object=data["elastic_vertical_displacement_grid"]
    )
    data["e_mean"] = (
        data["anelastic_geoid_deformation_mean"] - data["elastic_geoid_deformation_mean"]
    )
    data["f_mean"] = (
        data["anelastic_vertical_displacement_mean"] - data["elastic_vertical_displacement_mean"]
    )

    sub_function_figure_sup_5(
        axes=[ax1, ax2, ax3, ax4, ax5, ax6], data=data, ax_bar=ax_bar, fig=fig
    )

    ax1.text(
        0.3,
        1.15,
        "Geoid deformation",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    ax2.text(
        0.3,
        1.15,
        "Vertical displacement",
        transform=ax2.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    ax1.text(
        -0.25,
        0.4,
        "  Elastic",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    ax3.text(
        -0.25,
        0.4,
        " Anelastic",
        transform=ax3.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    ax5.text(
        -0.25,
        0.4,
        "Difference",
        transform=ax5.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    fig.savefig(figures_path.joinpath("figure_sup_5.svg"), format="svg", dpi=300)


def generate_figure_sup_6(figsize: tuple[float, float] = (18, 7)) -> None:
    """
    2025's article.
    """

    # dates, lower_bound, mean_curb, upper_bound, latitudes, longitudes, mask, grid.
    data = load_base_model(name="figure_sup_6", path=figures_path)
    fig = figure(figsize=figsize)
    axes: list[GeoAxes] = [
        fig.add_axes([0.06, 0.55, 0.26, 0.4], projection=Robinson(central_longitude=0)),
        fig.add_axes([0.35, 0.55, 0.26, 0.4], projection=Robinson(central_longitude=0)),
        fig.add_axes([0.64, 0.55, 0.26, 0.4], projection=Robinson(central_longitude=0)),
        fig.add_axes([0.06, 0.05, 0.26, 0.4], projection=Robinson(central_longitude=0)),
        fig.add_axes([0.35, 0.05, 0.26, 0.4], projection=Robinson(central_longitude=0)),
        fig.add_axes([0.64, 0.05, 0.26, 0.4], projection=Robinson(central_longitude=0)),
    ]
    ax_top: Axes = fig.add_axes([0.5, 0.55, 0.5, 0.4], frameon=False)
    ax_top.get_xaxis().set_ticks([])
    ax_top.get_yaxis().set_ticks([])
    ax_bottom: Axes = fig.add_axes([0.5, 0.05, 0.5, 0.4], frameon=False)
    ax_bottom.get_xaxis().set_ticks([])
    ax_bottom.get_yaxis().set_ticks([])

    for solution, ax, letter in zip(
        [
            "elastic_residuals_grid",
            "elastic_filtered_residuals_grid",
            "elastic_residuals_grid_without_2_1",
            "anelastic_residuals_grid",
            "anelastic_filtered_residuals_grid",
            "anelastic_residuals_grid_without_2_1",
        ],
        axes,
        ["A.", "B.", "C.", "D.", "E.", "F."],
    ):

        contour = natural_projection(
            ax=ax,
            data=data,
            grid=numpy.array(object=data[solution]),
            saturation_threshold=0.01 if letter in ["A.", "B.", "C."] else 0.5,
        )
        ax.text(
            0.1,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_TITLES,
            fontweight="bold",
        )
        ax.text(
            -0.01, 0.11, "60°S", transform=ax.transAxes, fontsize=FONTSIZE, fontweight="regular"
        )

        if "F" in letter:

            cbar = fig.colorbar(
                contour,
                ax=ax_bottom,
                orientation="vertical",
                shrink=0.7,
                ticks=numpy.array(object=[-0.5, -0.25, 0, 0.25, 0.5]),
                extend="both",
            )
            cbar.ax.tick_params(labelsize=LABELSIZE)
            cbar.set_label(
                label=" (mm/yr)",
                fontsize=FONTSIZE,
            )

        if "C" in letter:

            cbar = fig.colorbar(
                contour,
                ax=ax_top,
                orientation="vertical",
                shrink=0.7,
                ticks=numpy.array(object=[-0.01, -0.005, 0, 0.005, 0.01]),
                extend="both",
            )
            cbar.ax.tick_params(labelsize=LABELSIZE)
            cbar.set_label(
                label=" (mm/yr)",
                fontsize=FONTSIZE,
            )

    axes[0].text(
        0.3,
        1.15,
        "     Raw residuals",
        transform=axes[0].transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    axes[1].text(
        0.3,
        1.15,
        "    Filtered residuals",
        transform=axes[1].transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    axes[2].text(
        0.2,
        1.15,
        r"Filtered residuals\nwithout $C_{21}/S_{21}$",
        transform=axes[2].transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    axes[0].text(
        -0.25,
        0.4,
        "  Elastic",
        transform=axes[0].transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    axes[3].text(
        -0.25,
        0.4,
        " Anelastic",
        transform=axes[3].transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    fig.savefig(figures_path.joinpath("figure_sup_6.svg"), format="svg", dpi=300)


def sub_function_figure_sup_7(
    axes: list[GeoAxes], data: dict, ax_top: Axes, ax_bottom: Axes, fig: Figure
) -> None:
    """
    Sub-function to minimize local variables.
    """

    for solution, ax, letter in zip(
        [
            "elastic_grid",
            "elastic_corrected_grid",
            "anelastic_grid",
            "anelastic_corrected_grid",
            "e_grid",
            "f_grid",
        ],
        axes,
        ["A.", "B.", "C.", "D.", "E. = C. - A.", "F. = D. - B."],
    ):

        contour = natural_projection(
            ax=ax,
            data=data,
            grid=data[solution],
            saturation_threshold=50 if len(letter) == 2 else 5,
        )
        ax.text(
            0.1,
            1.1,
            letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_TITLES,
            fontweight="bold",
        )

        if "B" in letter:

            cbar = fig.colorbar(
                contour,
                ax=ax_top if len(letter) == 2 else ax_bottom,
                orientation="vertical",
                shrink=0.5,
                ticks=numpy.array(
                    object=[-40, -20, 0, 20, 40] if len(letter) == 2 else [-4, -2, 0, 2, 4]
                ),
                extend="both",
            )
            cbar.ax.tick_params(labelsize=LABELSIZE)
            cbar.set_label(
                label=" (mm/yr)",
                fontsize=FONTSIZE,
            )


def generate_figure_sup_7(figsize: tuple[float, float] = (12, 10)) -> None:
    """
    2025's article.
    """

    # dates, lower_bound, mean_curb, upper_bound, latitudes, longitudes, mask, grid.
    data = load_base_model(name="figure_sup_7", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: GeoAxes = fig.add_axes([0.1, 0.65, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax2: GeoAxes = fig.add_axes([0.6, 0.65, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax3: GeoAxes = fig.add_axes([0.1, 0.35, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax4: GeoAxes = fig.add_axes([0.6, 0.35, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax5: GeoAxes = fig.add_axes([0.1, 0.05, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax6: GeoAxes = fig.add_axes([0.6, 0.05, 0.37, 0.28], projection=Robinson(central_longitude=0))
    ax_top: Axes = fig.add_axes([0.03, 0.35, 0.55, 0.6], frameon=False)
    ax_top.get_xaxis().set_ticks([])
    ax_top.get_yaxis().set_ticks([])
    ax_bottom: Axes = fig.add_axes([0.03, 0.05, 0.55, 0.3], frameon=False)
    ax_bottom.get_xaxis().set_ticks([])
    ax_bottom.get_yaxis().set_ticks([])

    data["e_grid"] = numpy.array(object=data["anelastic_grid"]) - numpy.array(
        object=data["elastic_grid"]
    )
    data["f_grid"] = numpy.array(object=data["anelastic_corrected_grid"]) - numpy.array(
        object=data["elastic_corrected_grid"]
    )

    sub_function_figure_sup_7(
        axes=[ax1, ax2, ax3, ax4, ax5, ax6], data=data, ax_top=ax_top, ax_bottom=ax_bottom, fig=fig
    )

    ax1.text(
        0.25,
        1.15,
        "Before Leakage Correction",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    ax2.text(
        0.25,
        1.15,
        "After Leakage Correction",
        transform=ax2.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    ax1.text(
        -0.25,
        0.4,
        "  Elastic",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    ax3.text(
        -0.25,
        0.4,
        " Anelastic",
        transform=ax3.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    ax5.text(
        -0.25,
        0.4,
        "Difference",
        transform=ax5.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        rotation=45,
    )
    fig.savefig(figures_path.joinpath("figure_sup_7.svg"), format="svg", dpi=300)


def generate_figure_sup_8(figsize: tuple[float, float] = (8, 16)) -> None:
    """
    2025's article.
    """

    df = read_csv_and_replace(filepath=figures_path.joinpath("data_all_steps.csv"))
    data: dict[str, dict[str, dict]] = load_base_model(name="figure_4", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: Axes = fig.add_axes([0.15, 0.74, 0.8, 0.21])
    ax2: Axes = fig.add_axes([0.15, 0.51, 0.8, 0.21], sharex=ax1)
    ax3: Axes = fig.add_axes([0.15, 0.28, 0.8, 0.21], sharex=ax1)
    ax4: Axes = fig.add_axes([0.15, 0.05, 0.8, 0.21], sharex=ax1)
    reference_values = get_reference_values(df=df, data=data, all_metrics=True)

    for ax, anelasticity_option, color, letter in zip(
        [ax1, ax2, ax3, ax4], ANELASTICITY_OPTIONS, OPTION_COLORS, ["A.", "B.", "C.", "D."]
    ):

        for i_step in range(5):

            cloud_data = df[df["Anelasticity"] == anelasticity_option][
                "ocean_mean_trend_step_" + str(i_step + 1)
            ].values
            draw_violin_and_boxplot(
                ax=ax, x_value=i_step, cloud_data=cloud_data, color=color, width=1
            )

        ax.set_ylabel(
            anelasticity_option.replace("\n", " ")
            + " model\n Mean Barystatic Sea Level\nTrend (2003 - 2022) (mm/yr)",
            fontsize=FONTSIZE_AXE_LABELS,
        )
        ax.grid()
        ax.tick_params(
            axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
        )
        ax.set_ylim(1.8, 2.7)
        ax.set_yticks([2.0, 2.2, 2.4, 2.6])
        ax.text(
            -0.1,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_TITLES,
            fontweight="bold",
        )
        ax.scatter(
            x=list(range(5)),
            y=[
                reference_values[anelasticity_option]["ocean_mean_trend_step_" + str(i + 1)]
                for i in range(5)
            ],
            label="reference model",
            s=SIZE,
            marker="*",
            color=REFERENCE_RED,
            zorder=100,
        )

    ax1.legend(frameon=False, fontsize=LABELSIZE)
    setp(ax1.get_xticklabels(), visible=False)
    setp(ax2.get_xticklabels(), visible=False)
    setp(ax3.get_xticklabels(), visible=False)
    ax4.set_xticks(
        ticks=range(5),
        labels=[
            "Initial\nsignal",
            "With pole tide\ncorrection",
            "Anelasticity\nre-estimation",
            "With degree\none inversion",
            "With leakage\ncorrection",
        ],
    )

    fig.savefig(figures_path.joinpath("figure_sup_8.svg"), format="svg")


def generate_figure_sup_9(figsize: tuple[float, float] = (6, 4)) -> None:
    """
    2025's article.
    """

    df = read_csv_and_replace(filepath=figures_path.joinpath("data_alpha.csv"))
    fig = figure(figsize=figsize)
    ax1: Axes = fig.add_axes([0.12, 0.12, 0.8, 0.8])  # [left, bottom, width, height].

    for x_position in range(2):

        for x_offset, alpha in zip([-0.25, 0.0, 0.25], [0.223, 0.26, 0.297]):

            cloud_data = df[df["alpha"] == alpha]["ocean_mean_trend_step_5"]
            draw_violin_and_boxplot(
                ax=ax1,
                x_value=x_position + x_offset,
                cloud_data=cloud_data,
                color=OPTION_COLORS[2 + x_position],
                width=0.25,
            )
            text(
                x_position + x_offset,
                max(cloud_data) + 0.01,
                str(alpha),
                ha="center",
                fontsize=FONTSIZE,
            )

    ax1.yaxis.set_ticks_position("both")
    ax1.grid()
    ax1.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax1.set_ylabel(ylabel="(mm/yr)", fontsize=FONTSIZE_AXE_LABELS)
    ax1.set_xticks(ticks=[0, 1], labels=ANELASTICITY_OPTIONS[2:4])
    ax1.set_title(
        r"Mean Barystatic Sea Level Trend (2003 - 2022) per $\alpha$ value",
        fontsize=FONTSIZE_PANEL_TITLES,
    )
    ax1.set_ylim(2.2, 3.1)

    fig.savefig(figures_path.joinpath("figure_sup_9.svg"), format="svg")


def generate_figure_sup_10(figsize: tuple[float, float] = (7, 4)) -> None:
    """
    2025's article.
    """

    df_modified = read_csv_and_replace(filepath=figures_path.joinpath("data_uniform.csv"))
    df_ref = read_csv_and_replace(filepath=figures_path.joinpath("data_step_5.csv"))
    data: dict[str, dict[str, dict]] = load_base_model(name="figure_4", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: Axes = fig.add_axes([0.12, 0.12, 0.8, 0.8])  # [left, bottom, width, height].

    for x_position, color in enumerate(OPTION_COLORS):

        cloud_data = df_modified[df_modified["Anelasticity"] == ANELASTICITY_OPTIONS[x_position]][
            "ocean_mean_trend_step_5"
        ].values
        draw_violin_and_boxplot(
            ax=ax1, x_value=x_position, cloud_data=cloud_data, color=color, width=1
        )

    for label in ["modified reference model", "reference model"]:

        reference_values = {}

        for option in ANELASTICITY_OPTIONS:

            selection = df_modified if "modified" in label else df_ref

            for parameter, reference_value in REFERENCE_MODEL_PARAMETERS.items():

                if parameter == "Uniform\ncontinental load model":

                    continue

                if parameter in data["results"]["ocean_mean_trend_step_5"][option]:

                    selection = selection[
                        numpy.array(object=selection[parameter], dtype=str) == reference_value
                    ]

            reference_values[option] = {
                "ocean_mean_trend_step_5": selection[selection["Anelasticity"] == option][
                    "ocean_mean_trend_step_5"
                ]
            }

        ax1.scatter(
            x=ANELASTICITY_OPTIONS,
            y=[
                reference_values[option]["ocean_mean_trend_step_5"]
                for option in ANELASTICITY_OPTIONS
            ],
            label=label,
            s=SIZE,
            marker="o" if "modified" in label else "*",
            color=REFERENCE_RED,
            zorder=100,
        )

    ax1.set_ylabel(
        "Mean Barystatic Sea Level\nTrend (2003 - 2022) (mm/yr)", fontsize=FONTSIZE_AXE_LABELS
    )
    ax1.grid()
    ax1.set_xticks(
        ticks=range(4),
        labels=ANELASTICITY_OPTIONS,
    )
    ax1.set_xlim(-0.5, 3.7)
    ax1.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax1.legend(frameon=False, fontsize=FONTSIZE)

    fig.savefig(figures_path.joinpath("figure_sup_10.svg"), format="svg")


def generate_figure_sup_11(figsize: tuple[float, float] = (12, 4)) -> None:
    """
    2025's article.
    """

    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = subplots(1, 2, figsize=figsize, sharex=True)
    data = load_base_model(name="figure_sup_11", path=figures_path)

    ax1.plot(
        data["dates"],
        data["series_e"],
        label="purely elastic model",
        color="blue",
    )
    ax1.plot(
        data["dates"],
        data["series_a"],
        label="reference model",
        color="red",
    )
    ax1.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax1.legend(frameon=False)
    ax1.set_ylabel("mm")
    ax1.set_xlabel("yr")
    ax1.text(
        -0.1,
        1.1,
        "A.",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )

    ax2.plot(data["dates"], data["d"], label="difference: anelastic - elastic", color="black")
    ax2.plot(
        data["dates"],
        numpy.array(object=data["quadratic"]) * numpy.array(object=data["trend_dates"]) ** 2
        + numpy.array(object=data["linear_from_quadratic"])
        * numpy.array(object=data["trend_dates"]),
        label="Quadratic fit\nRMS = " + str(data["rms_from_quadratic"])[1:5] + " mm",
        color="orange",
        linestyle="--",
    )
    ax2.plot(
        data["dates"],
        numpy.array(object=data["linear"]) * numpy.array(object=data["trend_dates"]),
        label="Linear fit\nRMS = " + str(data["rms_from_linear"])[1:5] + " mm",
        color="black",
        linestyle="--",
    )
    ax2.hlines(
        y=[4],
        xmin=2003,
        xmax=2022,
        linestyles="--",
        color="blue",
        label="observational uncertainty",
    )
    ax2.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax2.legend(frameon=False)
    ax2.set_ylabel("mm")
    ax2.set_xlabel("yr")
    ax2.set_xticks(ticks=range(2003, 2023, 3))
    ax2.text(
        -0.1,
        1.1,
        "B.",
        transform=ax2.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )

    fig.savefig(figures_path.joinpath("figure_sup_11.svg"), format="svg")
