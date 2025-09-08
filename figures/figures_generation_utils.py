"""
Constants and colors.
"""

from typing import Any

import matplotlib.ticker as mticker
import numpy
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from numpy.ma import MaskedArray
from pandas import DataFrame

from transient_solid_earth import data_path

from .figures_data_formater_utils import ANELASTICITY_OPTIONS, REFERENCE_MODEL_PARAMETERS

FONTSIZE_AXE_LABELS = 12
LABELSIZE = 12
FONTSIZE_TICKLABELS = 12
FONTSIZE_PANEL_TITLES = 12
LABELSIZE = 12
FONTSIZE = 12
LINEWIDTH = 3
SIZE = 30

# Generate interpolated colors
LAYER_COLORS = {
    " Lower Mantle": (248 / 255, 246 / 255, 246 / 255),
    " Upper Mantle": (255 / 255, 239 / 255, 217 / 255),
    "Asthenosphere": (255 / 255, 245 / 255, 232 / 255),
    "   Lithosphere": (243 / 255, 246 / 255, 246 / 255),
    "       Crust": (244 / 255, 241 / 255, 236 / 255),
}

ELASTIC_COLOR = numpy.array((34, 139, 34)) / 255.0
LONG_TERM_COLOR = numpy.array((31, 78, 121)) / 255.0
SHORT_TERM_COLOR = numpy.array((92, 75, 139)) / 255.0
REFERENCE_RED = numpy.array((199, 48, 43)) / 255.0

OPTION_COLORS = numpy.array(
    [
        ELASTIC_COLOR,
        LONG_TERM_COLOR,
        SHORT_TERM_COLOR,
        REFERENCE_RED,
    ]
)

SHORT_TERM_COLORS = (
    numpy.array(
        [
            255 * REFERENCE_RED,
            (0, 72, 172),
            (170, 181, 217),
            (32, 121, 190),
            (0, 72, 172),
        ]
    )
    / 255.0
)

LONG_TERM_COLORS = (
    numpy.array(
        [255 * REFERENCE_RED, (113, 74, 130), (136, 89, 155), (147, 100, 166), (91, 60, 104)]
    )
    / 255.0
)
REFERENCE_RED = numpy.array((199, 48.0, 43.0)) / 255.0

MAIN_LAYER_DEPTHS = {
    " Lower Mantle": (2891.0, 670.0),
    " Upper Mantle": (670.0, 300.0),
    "Asthenosphere": (300.0, 100.0),
    "   Lithosphere": (100.0, 25.0),
    "       Crust": (25.0, 0.0),
}

JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD = 2.49
JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD_UNCERTAINTY = 0.254

# Gets the grid colormap.
with open(data_path.joinpath("cmap.txt"), "r", encoding="utf-8") as file:
    lines = file.readlines()
    cmap_lines = []
    for line in lines:
        cmap_lines += [[int(element) for element in line.split(",")]]


CMAP = ListedColormap(numpy.array(cmap_lines) / 255.0)


def add_ticks(ax: Axes):
    """
    Adds only tick marks (no labels), placed just *outside* the visible map,
    on the left and bottom, for a Robinson projection.
    """

    for lon in [-90, 0, 90]:

        ax.plot(
            [lon],
            [-89.9],
            marker="|",
            color="black",
            transform=PlateCarree(),
            markersize=8,
            clip_on=False,
        )

    for lat in [-60, -30, 0, 30, 60]:

        ax.plot(
            [-179.9],
            [lat],
            marker="_",
            color="black",
            transform=PlateCarree(),
            markersize=8,
            clip_on=False,
        )


def natural_projection(
    ax: GeoAxes,
    data: dict,
    grid: numpy.ndarray,
    saturation_threshold: float = 50.0,
) -> Any:
    """
    Displays a projection of a given grid on the given matplotlib Axes.
    Ensures areas outside the mask are shaded in grey.
    """

    # For bar extends.
    grid[0] = [1e10 * saturation_threshold] * len(data["longitudes"])
    grid[-1] = [-1e10 * saturation_threshold] * len(data["longitudes"])

    if data["mask"]:

        grid = MaskedArray(grid, mask=1 - numpy.array(object=data["mask"]))

    # Plots.
    contour = ax.contourf(
        data["longitudes"],
        data["latitudes"],
        grid,
        levels=numpy.concatenate(
            (
                [-1e10 * saturation_threshold * saturation_threshold],
                numpy.linspace(-saturation_threshold, saturation_threshold, 100),
                [1e10 * saturation_threshold * saturation_threshold],
            )
        ),
        cmap=CMAP,
        norm=TwoSlopeNorm(vcenter=0, vmin=-saturation_threshold, vmax=saturation_threshold),
        transform=PlateCarree(central_longitude=0),
        extend="both",
        rasterized=True,
    )
    # Overlays gray where mask is False (masked area).
    if data["mask"] is not None:
        mask = numpy.array(data["mask"])
        masked_area = numpy.where(mask == 0, 1, numpy.nan)
        ax.pcolormesh(
            data["longitudes"],
            data["latitudes"],
            masked_area,
            cmap=ListedColormap(["grey"]),
            shading="auto",
            transform=PlateCarree(central_longitude=0),
            rasterized=True,
        )
    ax.coastlines()
    gl = ax.gridlines(
        crs=PlateCarree(central_longitude=0),
        alpha=0,
        ylocs=[-60, -30, 0, 30, 60],
    )
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.left_labels = [-60, -30, 0, 30, 60]
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.xlabel_style = {"size": LABELSIZE}
    gl.ylabel_style = {"size": LABELSIZE}
    gl.bottom_labels = True
    gl.top_labels = False
    gl.xlines = False
    gl.left_labels = True
    gl.right_labels = False
    add_ticks(ax=ax)

    return contour


def get_reference_values(
    df: DataFrame, data: dict, all_metrics: bool = False
) -> dict[str, dict[str, float]]:
    """
    Gets the reference values for the model parameters.
    """
    reference_values = {}

    for option in ANELASTICITY_OPTIONS:

        selection = df

        for parameter, reference_value in REFERENCE_MODEL_PARAMETERS.items():

            if parameter in data["results"]["ocean_mean_trend_step_5"][option]:

                selection = selection[
                    numpy.array(object=selection[parameter], dtype=str) == reference_value
                ]

        reference_values[option] = {}

        if all_metrics:

            for i in range(5):

                reference_values[option]["ocean_mean_trend_step_" + str(i + 1)] = selection[
                    selection["Anelasticity"] == option
                ]["ocean_mean_trend_step_" + str(i + 1)]

        else:

            reference_values[option]["ocean_mean_trend_step_5"] = selection[
                selection["Anelasticity"] == option
            ]["ocean_mean_trend_step_5"]
            reference_values[option]["vertical_deformation_ocean_mean_trend"] = selection[
                selection["Anelasticity"] == option
            ]["vertical_deformation_ocean_mean_trend"]

    return reference_values


def add_reference_values_and_altimetry(df: DataFrame, ax2: Axes, ax3: Axes, data: dict) -> None:
    """
    Sub-functions.
    """

    reference_values = get_reference_values(df=df, data=data)

    odb_anelastic_elastic_correction = (
        reference_values["short-term\nand long-term"][
            "vertical_deformation_ocean_mean_trend"
        ].values[0]
        - reference_values["elastic"]["vertical_deformation_ocean_mean_trend"]
    )
    ax2.scatter(
        x=ANELASTICITY_OPTIONS,
        y=[reference_values[option]["ocean_mean_trend_step_5"] for option in ANELASTICITY_OPTIONS],
        label="reference model",
        s=SIZE,
        marker="*",
        color=REFERENCE_RED,
        zorder=10,
    )
    ax2.legend(frameon=False, fontsize=LABELSIZE, loc="upper left")
    odb_anelastic_elastic_correction = list(odb_anelastic_elastic_correction)
    boxprops = {"color": (0, 0, 0), "linewidth": 2, "alpha": 0.5}
    medianprops = {"color": (0, 0, 0), "linewidth": 2}
    whiskerprops = {"color": (0, 0, 0), "linewidth": 2, "alpha": 0.5}
    capprops = {"color": (0, 0, 0), "linewidth": 2, "alpha": 0.5}
    boxplot = ax3.boxplot(
        [
            JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD
            - odb_anelastic_elastic_correction[0]
            - JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD_UNCERTAINTY,
            JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD
            - odb_anelastic_elastic_correction[0]
            + JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD_UNCERTAINTY,
        ],
        positions=[0],
        showfliers=False,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )
    boxplot["boxes"][0].set_facecolor((0, 0, 0))

    box_mean = JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD - odb_anelastic_elastic_correction[0]

    draw_arrows(df=df, box_mean=box_mean, ax2=ax2, ax3=ax3)


def draw_arrows(df: DataFrame, box_mean: float, ax2: Axes, ax3: Axes) -> None:
    """
    Draws main vertical lines on the altimetry subplot.
    """

    ref_y_elastic = numpy.median(
        df[df["Anelasticity"] == ANELASTICITY_OPTIONS[0]]["ocean_mean_trend_step_5"].values
    )

    ref_y = numpy.median(
        df[df["Anelasticity"] == ANELASTICITY_OPTIONS[-1]]["ocean_mean_trend_step_5"].values
    )

    ax2_pos = ax2.get_position()
    ax3_pos = ax3.get_position()
    ref_xy_fig = ax2.figure.transFigure.inverted().transform(ax2.transData.transform((3, ref_y)))
    ax3_left_fig = (ax3_pos.x0, ref_xy_fig[1])
    ax2.figure.lines.append(
        Line2D(
            [ref_xy_fig[0], ax3_left_fig[0]],
            [ref_xy_fig[1], ax3_left_fig[1]],
            linestyle=":",
            color=REFERENCE_RED,
            linewidth=2,
            zorder=100,
            transform=ax2.figure.transFigure,
        )
    )
    ref_xy_fig = ax2.figure.transFigure.inverted().transform(
        ax2.transData.transform((0, ref_y_elastic))
    )
    ax3_left_fig = (ax3_pos.x0, ref_xy_fig[1])
    ax2.figure.lines.append(
        Line2D(
            [ref_xy_fig[0], ax3_left_fig[0]],
            [ref_xy_fig[1], ax3_left_fig[1]],
            linestyle=":",
            color="gray",
            linewidth=2,
            zorder=100,
            transform=ax2.figure.transFigure,
        )
    )

    ref_xy_prime_fig = ax3.figure.transFigure.inverted().transform(
        ax3.transData.transform((0, box_mean))
    )
    ax2_right_fig = (ax2_pos.x1, ref_xy_prime_fig[1])
    ax3.figure.lines.append(
        Line2D(
            [ref_xy_prime_fig[0], ax2_right_fig[0]],
            [ref_xy_prime_fig[1], ax2_right_fig[1]],
            linestyle=":",
            color=REFERENCE_RED,
            linewidth=2,
            zorder=100,
            transform=ax3.figure.transFigure,
        )
    )

    ref_xy_fig = ax3.figure.transFigure.inverted().transform(ax2.transData.transform((3, ref_y)))
    ref_xy_elastic_fig = ax3.figure.transFigure.inverted().transform(
        ax2.transData.transform((0, ref_y_elastic))
    )
    box_xy_fig = ax3.figure.transFigure.inverted().transform(ax3.transData.transform((0, box_mean)))

    x_arrow = (ax2_pos.x1 + ax3_pos.x0) / 2

    draw_vertical_arrow(
        fig=ax3.figure, x=x_arrow - 0.03, y1=ref_xy_fig[1], y2=box_xy_fig[1], color=REFERENCE_RED
    )
    ax3.figure.text(
        x_arrow - 0.02,
        (ref_xy_fig[1] + box_xy_fig[1]) / 2,
        f"${box_mean - ref_y:.2f}$ mm/yr",
        fontsize=FONTSIZE_PANEL_TITLES - 2,
        color=REFERENCE_RED,
        va="center",
        ha="left",
        zorder=200,
    )

    draw_vertical_arrow(
        fig=ax3.figure, x=x_arrow - 0.05, y1=ref_xy_elastic_fig[1], y2=box_xy_fig[1], color="gray"
    )
    ax3.figure.text(
        x_arrow - 0.04,
        (ref_xy_elastic_fig[1] + box_xy_fig[1]) / 2,
        f"${box_mean - ref_y_elastic:.2f}$ mm/yr",
        fontsize=FONTSIZE_PANEL_TITLES - 2,
        color="black",
        va="center",
        ha="left",
        zorder=200,
    )

    ax3.figure.text(
        x_arrow,
        (ref_xy_fig[1] + box_xy_fig[1]) / 2 + 0.05,
        "Sea level\nMisclosure",
        fontsize=FONTSIZE_PANEL_TITLES,
        color="black",
        va="center",
        ha="center",
        zorder=200,
    )


def draw_vertical_arrow(fig, x, y1, y2, color):
    """
    Draws main vertical line.
    """

    fig.lines.append(
        Line2D(
            [x, x],
            [y1, y2],
            linestyle=":",
            color=color,
            linewidth=2,
            zorder=150,
            transform=fig.transFigure,
        )
    )


def draw_violin_and_boxplot(
    ax: Axes, x_value: float, cloud_data: list[float], color: str, width: float
) -> None:
    """
    Draws a single violin and boxplot on the given axes.
    """

    violins = ax.violinplot(
        dataset=cloud_data,
        positions=[x_value],
        showmeans=False,
        points=100,
        showextrema=False,
        side="high",
        widths=width,
    )

    for body in violins["bodies"]:

        body.set_color(color)

    boxprops = {"color": color, "linewidth": 2, "alpha": 0.8}
    medianprops = {"color": color, "linewidth": 2}
    whiskerprops = {"color": color, "linewidth": 2, "alpha": 0.8}
    capprops = {"color": color, "linewidth": 2, "alpha": 0.8}

    boxplot = ax.boxplot(
        cloud_data,
        positions=[x_value],
        showfliers=False,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )
    boxplot["boxes"][0].set_facecolor(color)
