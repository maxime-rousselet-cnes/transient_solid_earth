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

# Define start and end colors
grey_color = numpy.array((220, 220, 220)) / 255.0
brown_color = numpy.array((215, 179, 128)) / 255.0

# Generate interpolated colors
LAYER_COLORS = {
    " Lower Mantle": grey_color * 0.6 + brown_color * 0.4,
    " Upper Mantle": grey_color * 0.4 + brown_color * 0.6,
    "Asthenosphere": brown_color,
    "   Lithosphere": grey_color * 0.8 + brown_color * 0.2,
    "       Crust": grey_color,
}

# Blue: 30,144,255
# Green: 50,205,50
ELASTIC_COLOR = numpy.array((0, 150, 0)) / 255.0
LONG_TERM_COLOR = numpy.array((91, 60, 104)) / 255.0
SHORT_TERM_COLOR = numpy.array((0, 0, 255)) / 255.0
REFERENCE_RED = numpy.array((255, 0, 0)) / 255.0

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
            (55, 21, 233),
            (95, 191, 249),
            (88, 99, 248),
            255 * SHORT_TERM_COLOR,
        ]
    )
    / 255.0
)

LONG_TERM_COLORS = (
    numpy.array(
        [
            255 * REFERENCE_RED,
            (113, 74, 130),
            (136, 89, 155),
            (147, 100, 166),
            255 * LONG_TERM_COLOR,
        ]
    )
    / 255.0
)
REFERENCE_RED = numpy.array((255, 0, 0)) / 255.0
BACKGROUND_ALPHA = 0.4

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


def add_reference_values_and_altimetry(df: DataFrame, ax2: Axes, data: dict) -> None:
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
        x=ANELASTICITY_OPTIONS + ["Altimetry-ARGO"],
        y=[reference_values[option]["ocean_mean_trend_step_5"] for option in ANELASTICITY_OPTIONS]
        + [
            JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD - odb_anelastic_elastic_correction,
        ],
        label="reference model",
        s=SIZE,
        marker="*",
        color=REFERENCE_RED,
    )
    odb_anelastic_elastic_correction = list(odb_anelastic_elastic_correction)
    boxprops = {"color": REFERENCE_RED, "linewidth": 2, "alpha": 0.5}
    medianprops = {"color": REFERENCE_RED, "linewidth": 2}
    whiskerprops = {"color": REFERENCE_RED, "linewidth": 2, "alpha": 0.5}
    capprops = {"color": REFERENCE_RED, "linewidth": 2, "alpha": 0.5}
    boxplot = ax2.boxplot(
        [
            JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD
            - odb_anelastic_elastic_correction[0]
            - JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD_UNCERTAINTY,
            JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD
            - odb_anelastic_elastic_correction[0]
            + JASON3_DRIFT_CORRECTED_MINUS_ARGO_PLUS_ABS_OBD_UNCERTAINTY,
        ],
        positions=[4],
        showfliers=False,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )
    boxplot["boxes"][0].set_facecolor(REFERENCE_RED)


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

    boxprops = {"color": color, "linewidth": 2, "alpha": 0.5}
    medianprops = {"color": color, "linewidth": 2}
    whiskerprops = {"color": color, "linewidth": 2, "alpha": 0.5}
    capprops = {"color": color, "linewidth": 2, "alpha": 0.5}

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
