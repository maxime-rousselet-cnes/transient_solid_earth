"""
2025's main paper.
"""

import numpy
from cartopy.crs import Robinson
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pylab
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.pyplot import figure, setp, subplots

from transient_solid_earth import load_base_model

from .figures_data_formater_utils import (
    ANELASTICITY_OPTIONS,
    LONG_TERM_MAP,
    SHORT_TERM_MAP,
    figures_path,
    read_csv_and_replace,
)
from .figures_generation_utils import (
    FONTSIZE,
    FONTSIZE_AXE_LABELS,
    FONTSIZE_PANEL_TITLES,
    FONTSIZE_TICKLABELS,
    LABELSIZE,
    LAYER_COLORS,
    LINEWIDTH,
    LONG_TERM_COLORS,
    MAIN_LAYER_DEPTHS,
    OPTION_COLORS,
    REFERENCE_RED,
    SHORT_TERM_COLORS,
    add_reference_values_and_altimetry,
    draw_violin_and_boxplot,
    natural_projection,
)


def generate_figure_1(
    figsize: tuple[float, float] = (7, 8.5), correct_for_latitudes: bool = True
) -> None:
    """
    2025's article.
    """

    # dates, lower_bound, mean_curb, upper_bound, latitudes, longitudes, mask, grid.
    data = load_base_model(name="figure_1", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: Axes = fig.add_axes([0.11, 0.7, 0.78, 0.22])  # [left, bottom, width, height].
    ax2: GeoAxes = fig.add_axes([0.07, 0.075, 0.86, 0.54], projection=Robinson(central_longitude=0))
    ax3: Axes = fig.add_axes([0.1, 0.03, 0.8, 0.5], frameon=False)
    ax3.get_xaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])

    # Panel A.
    ax1.plot(data["dates"], data["mean_curb"], color=REFERENCE_RED, linewidth=LINEWIDTH)
    ax1.fill_between(
        data["dates"],
        data["lower_bound"],
        data["upper_bound"],
        color=(248 / 255, 230 / 255, 230 / 255),
    )
    ax1.yaxis.set_ticks_position("left")
    ax1.set_xlabel(xlabel="Time (yr)", fontsize=FONTSIZE_AXE_LABELS)
    ax1.set_ylabel(ylabel="Barystatic GMSL (mm)", fontsize=FONTSIZE_AXE_LABELS)
    ax1.set_yticks(range(0, 151, 25), minor=True)
    ax1.set_xticks(range(1900, 2021, 10), minor=True)
    ax1.set_ylim(bottom=-25, top=150)
    ax1.set_yticks(range(0, 151, 50))
    ax1.set_xticks(range(1900, 2021, 20))
    ax1.tick_params(
        axis="both",
        which="major",
        direction="inout",
        length=10,
        width=1.5,
        labelsize=FONTSIZE_TICKLABELS,
    )
    ax1.tick_params(
        axis="both",
        which="minor",
        direction="inout",
        length=5,
        width=1,
        labelsize=FONTSIZE_TICKLABELS,
    )
    ax1.text(
        -0.12,
        1.12,
        "A. Temporal evolution of ocean mass",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )

    # Panel B.
    contour = natural_projection(
        ax=ax2,
        data=data,
        grid=data["grid"],
    )
    cbar = fig.colorbar(
        contour,
        ax=ax3,
        orientation="horizontal",
        shrink=0.7,
        ticks=[-40, -20, 0, 20, 40],
        extend="both",
    )
    cbar.ax.tick_params(labelsize=LABELSIZE)
    cbar.set_label(label="GRACE/-FO 2003-2022 mass trends\n(mm of EWH/yr)", fontsize=FONTSIZE)
    ax2.text(
        -0.07,
        1.12,
        "B. Spatial distribution of surface loading",
        transform=ax2.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )

    if correct_for_latitudes:

        ax2.text(
            0.0, 0.85, "60°N", transform=ax2.transAxes, fontsize=FONTSIZE, fontweight="regular"
        )
        ax2.text(
            0.0, 0.11, "60°S", transform=ax2.transAxes, fontsize=FONTSIZE, fontweight="regular"
        )

    fig.savefig(figures_path.joinpath("figure_1.svg"), format="svg", dpi=300)


def generate_figure_2(figsize: tuple[float, float] = (9, 7)) -> None:
    """
    2025's article.
    """

    data: dict[str, dict[str]] = load_base_model(name="figure_2", path=figures_path)
    fig = figure(figsize=figsize)
    ax1: Axes = fig.add_axes([0.11, 0.1, 0.35, 0.75])  # [left, bottom, width, height].
    ax2: Axes = fig.add_axes([0.61, 0.1, 0.35, 0.75], sharey=ax1)  # [left, bottom, width, height].
    ax_mu = ax1.twiny()
    pylab.rcParams.update(
        {"axes.labelsize": FONTSIZE_AXE_LABELS, "axes.titlesize": FONTSIZE_AXE_LABELS}
    )

    # Patches for layers.
    for name, color in LAYER_COLORS.items():

        y = MAIN_LAYER_DEPTHS[name][1]
        y_mem = MAIN_LAYER_DEPTHS[name][0]
        con = ConnectionPatch(
            xyA=(620, y),
            xyB=(9e18, y),
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax2,
            linewidth=1,
            color=(0.5, 0.5, 0.5),
            linestyle="-." if "Asthenosphere" in name or "Lithosphere" in name else "--",
        )
        ax2.add_patch(
            Rectangle(
                (9e16, y),
                1e24 - 9e16,
                y_mem - y,
                color=color,
            )
        )
        ax1.add_patch(
            Rectangle(
                (-50, y),
                670,
                y_mem - y,
                color=color,
            )
        )
        ax1.text(
            x=630,
            y=y_mem
            - 30.0
            + (10.0 if "Lithosphere" in name else (-2000.0 if "Lower" in name else 0.0)),
            s=name,
            fontsize=FONTSIZE,
            fontstyle="italic",
        )

        if name != "       Crust":

            ax2.add_artist(con)

    # Panel A.
    ax_mu.plot(
        numpy.array(object=data["mu"]["value"]) / 1e9,
        data["mu"]["depth"],
        color=REFERENCE_RED,
        linewidth=2,
        linestyle="--",
        label=r"$\mu_0$",
        zorder=10,
    )

    for i_model, (name, variable) in enumerate(data["q_mu"].items()):

        ax1.plot(
            variable["value"],
            variable["depth"],
            color=SHORT_TERM_COLORS[i_model],
            linewidth=2,
            label=SHORT_TERM_MAP[name],
            zorder=10 - i_model,
        )

    ax1.tick_params(
        axis="both",
        which="both",
        direction="inout",
        labelsize=FONTSIZE_TICKLABELS,
        length=10,
        width=1,
    )
    ax_mu.tick_params(
        axis="both",
        which="both",
        direction="inout",
        labelsize=FONTSIZE_TICKLABELS,
        length=10,
        width=1,
    )
    ax1.set_xlim(left=-50, right=620)
    ax1.set_xlabel(r"$Q_{\mu}$", fontsize=FONTSIZE_AXE_LABELS)
    ax1.set_ylim(top=-100, bottom=2900)
    ax_mu.set_xscale("log")
    ax1.set_ylabel("Depth (km)", fontsize=FONTSIZE_AXE_LABELS)
    ax_mu.set_xticks([50, 100, 200])
    ax_mu.set_xticklabels(["50", "100", r"$\mu_0$ (GPa)"])
    ax_mu.legend(loc=(0.035, 0.1), frameon=False, fontsize=LABELSIZE)
    ax1.legend(loc=("center left"), frameon=False, fontsize=LABELSIZE)
    ax1.text(
        -0.15,
        1.11,
        r"A. Elastic shear modulus $\mu_0$ & attenuation $Q_\mu$",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )
    ax1.text(
        x=-35,
        y=950,
        s="Attenuation",
        fontsize=FONTSIZE,
        fontstyle="italic",
    )
    ax1.text(
        x=-35,
        y=2350,
        s="Elastic shear\nmodulus",
        fontsize=FONTSIZE,
        fontstyle="italic",
    )
    ax1.yaxis.set_ticks_position("left")

    # Panel B.
    values = numpy.array(object=data["eta_m"]["VM7"]["value"])
    depths = numpy.array(object=data["eta_m"]["VM7"]["depth"])
    ax2.plot(
        values[depths < MAIN_LAYER_DEPTHS["Asthenosphere"][1]],
        depths[depths < MAIN_LAYER_DEPTHS["Asthenosphere"][1]],
        color=REFERENCE_RED,
        linewidth=2,
        zorder=10,
    )
    ax2.plot(
        values[depths > MAIN_LAYER_DEPTHS["Asthenosphere"][0]],
        depths[depths > MAIN_LAYER_DEPTHS["Asthenosphere"][0]],
        color=REFERENCE_RED,
        linewidth=2,
        label="VM7",
        zorder=10,
    )
    for i_model, (name, variable) in enumerate(data["eta_m"].items()):

        if name != "VM7":

            ax2.plot(
                variable["value"],
                variable["depth"],
                color=LONG_TERM_COLORS[i_model],
                linewidth=2,
                label=LONG_TERM_MAP[name],
            )

    ax2.text(
        x=1e19,
        y=900,
        s="Viscosity profile",
        fontsize=FONTSIZE,
        fontstyle="italic",
    )
    ax2.plot(
        [data["asthenospheric_viscosity"]] * 2,
        MAIN_LAYER_DEPTHS["Asthenosphere"],
        color=REFERENCE_RED,
        linewidth=2,
        linestyle="--",
        label="Weak asth.",
    )
    ax2.set_xticks(ticks=[1e19, 1e20, 1e21, 1e22, 1e23])
    ax2.tick_params(
        axis="both",
        which="both",
        direction="inout",
        labelsize=FONTSIZE_TICKLABELS,
        length=10,
        width=1,
    )
    ax2.set_xlabel(r"$\eta_m$ (Pa.s)", fontsize=FONTSIZE_AXE_LABELS)
    ax2.set_xlim(left=9e18, right=1e23)
    ax2.set_xscale("log")
    ax2.legend(loc=("center left"), frameon=False, fontsize=LABELSIZE)
    ax2.text(
        0.15,
        1.11,
        r"B. Long-term viscosity $\eta_m$",
        transform=ax2.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )
    ax2.yaxis.set_ticks_position("left")
    setp(ax2.get_yticklabels(), visible=False)

    fig.savefig(figures_path.joinpath("figure_2.svg"), format="svg")


def sub_figure_3(
    period_data: dict,
    name: str,
    color: numpy.ndarray,
    ax_line: tuple[Axes],
    strings: tuple[str, str],
) -> None:
    """
    Sub-plots.
    """

    period, _ = strings
    period = str(int(float(period)))
    y = MAIN_LAYER_DEPTHS[name][1]
    y_mem = MAIN_LAYER_DEPTHS[name][0]
    con = ConnectionPatch(
        xyA=(7, y),
        xyB=(-4, y),
        coordsA="data",
        coordsB="data",
        axesA=ax_line[0],
        axesB=ax_line[1],
        linewidth=1,
        color=(0.5, 0.5, 0.5),
        linestyle="-." if "Asthenosphere" in name or "Lithosphere" in name else "--",
    )
    ax_line[0].add_patch(
        Rectangle(
            (0, y),
            7,
            y_mem - y,
            color=color,
        )
    )
    ax_line[1].add_patch(
        Rectangle(
            (-4, y),
            5,
            y_mem - y,
            color=color,
        )
    )

    if period == "10":

        if name != "       Crust":

            ax_line[1].add_artist(con)

        ax_line[1].text(
            x=-6.2,
            y=y_mem
            - 30.0
            + (10.0 if "Lithosphere" in name else (-2000.0 if "Lower" in name else 0.0)),
            s=name,
            fontsize=FONTSIZE - 2,
            fontstyle="italic",
        )

    setp(ax_line[1].get_yticklabels(), visible=False)

    for i_part, (ax, part) in enumerate(zip(ax_line, ["real", "imag"])):

        i_option = 0

        for short_term in ["false", "true"]:

            for long_term in ["false", "true"]:

                ax.plot(
                    period_data[long_term][short_term][part],
                    period_data[long_term][short_term]["depth"],
                    color=OPTION_COLORS[i_option],
                    linewidth=2,
                    label=capitalize_first(
                        s=ANELASTICITY_OPTIONS[i_option]
                        + ("" if ANELASTICITY_OPTIONS[i_option] == "elastic" else "\nanelasticity"),
                    ),
                )
                ax.text(
                    0.9,
                    0.05,
                    strings[1][i_part],  # Letter.
                    transform=ax.transAxes,
                    fontsize=FONTSIZE_PANEL_TITLES,
                    fontweight="bold",
                )
                i_option += 1

    ax_line[-1].text(
        -0.35,
        0.5,
        period + " yr",
        transform=ax_line[-1].transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )


def capitalize_first(s: str) -> str:
    """
    Helper function.
    """

    return s[:1].upper() + s[1:]


def generate_figure_3(figsize: tuple[float, float] = (8, 18)) -> None:
    """
    2025's article.
    """

    data: dict[str, dict[str, dict[str]]] = load_base_model(name="figure_3", path=figures_path)
    axes: tuple[tuple[Axes]]
    fig, axes = subplots(
        nrows=3,
        ncols=2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"wspace": 0.45},
    )

    # Patches for layers.
    for letter_line, ax_line, (period, period_data) in zip(
        [["A", "B"], ["C", "D"], ["E", "F"]], axes, data.items()
    ):

        for name, color in LAYER_COLORS.items():

            sub_figure_3(
                period_data=period_data,
                name=name,
                color=color,
                ax_line=ax_line,
                strings=(period, letter_line),
            )

    axes[-1][-1].yaxis.set_ticks_position("both")
    axes[1][0].set_ylabel("Depth (km)", fontsize=FONTSIZE_AXE_LABELS)
    axes[-1][-1].tick_params(
        axis="both",
        which="both",
        direction="inout",
        labelsize=FONTSIZE_TICKLABELS,
        length=10,
        width=1,
    )
    axes[-1][0].set_xlim(left=0, right=7)
    axes[-1][1].set_xlim(left=-4, right=1)
    axes[-1][0].set_xlabel(
        r"$Re(\mu_0/\mu)$",
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )
    axes[-1][1].set_xlabel(
        r"$Im(\mu_0/\mu)$",
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
    )
    axes[0][0].text(
        0.35,
        0.75,
        "Rheology",
        transform=axes[0][0].transAxes,
        fontsize=FONTSIZE,
        fontstyle="italic",
    )

    for column in range(2):

        ax: Axes

        for ax in axes[:-1, column]:

            ax.sharex(axes[-1, column])

    axes[-1][-1].set_ylim(top=-100, bottom=2900)
    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # labels as keys, handles as values
    axes[0][0].legend(
        by_label.values(), by_label.keys(), loc="center right", frameon=False, fontsize=LABELSIZE
    )

    fig.savefig(figures_path.joinpath("figure_3.svg"), format="svg", dpi=500)


def generate_figure_4(figsize: tuple[float, float] = (10, 12)) -> None:
    """
    2025's article.
    """

    data: dict[str, dict[str, dict]] = load_base_model(name="figure_4", path=figures_path)
    df = read_csv_and_replace(filepath=figures_path.joinpath("data_step_5.csv"))
    fig = figure(figsize=figsize)
    ax1: Axes = fig.add_axes([0.1, 0.66, 0.8, 0.3])  # [left, bottom, width, height].
    ax2: Axes = fig.add_axes([0.1, 0.13, 0.54, 0.32])  # [left, bottom, width, height].
    ax3: Axes = fig.add_axes([0.8, 0.13, 0.1, 0.32])  # [left, bottom, width, height].

    # Panel A.
    for x_position, variability_factor in enumerate(data["sorted_parameters"]):

        selected_options = [
            option
            for option, sub_data in data["results"]["ocean_mean_trend_step_5"].items()
            if variability_factor in sub_data
        ]

        for anelasticity_option, offset in zip(
            selected_options,
            numpy.linspace(
                start=-0.5,
                stop=0.5,
                num=len(selected_options) + 2,
            )[1:-1],
        ):

            color = OPTION_COLORS[ANELASTICITY_OPTIONS.index(anelasticity_option)]
            cloud_data = data["results"]["ocean_mean_trend_step_5"][anelasticity_option][
                variability_factor
            ]
            x_value = x_position + offset
            draw_violin_and_boxplot(
                ax=ax1, x_value=x_value, cloud_data=cloud_data, color=color, width=0.25
            )

    ax1.text(
        0.75,
        0.93,
        "Rheology",
        transform=ax1.transAxes,
        fontsize=FONTSIZE,
        fontstyle="italic",
    )

    for anelasticity_option, color in zip(ANELASTICITY_OPTIONS, OPTION_COLORS):

        ax1.add_line(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=4,
                label=capitalize_first(
                    s=anelasticity_option
                    + ("" if anelasticity_option == "elastic" else "\nanelasticity"),
                ),
            )
        )

    ax1.legend(loc=(0.7, 0.43), frameon=False, fontsize=LABELSIZE)
    ax1.set_xlim(-0.4, 6.6)
    ax1.grid()
    ax1.set_ylabel("Barystatic GMSL trend (mm/yr)", fontsize=FONTSIZE_AXE_LABELS)
    ax1.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax1.set_xticks(
        ticks=range(len(data["sorted_parameters"])),
        labels=[
            "Relaxed shear\nmodulus ratio",
            "GRACE/-FO\nsolution",
            "Attenuation\nprofile",
            "non-cratonic\nasthenospheric\nviscosity",
            "historical barystatic\nsea level",
            "long-term\nviscosity profile",
            "Extension to\nLittle Ice Age",
        ],
        rotation=45,
    )
    ax1.text(
        0.47,
        1.05,
        "A. Sensitivity of 2003-2022 GRACE-derived ocean mass trends to model parameters",
        transform=ax1.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
        ha="center",
    )
    ax1.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04])

    # Panel B.
    for x_position, (anelasticity_option, color) in enumerate(
        zip(ANELASTICITY_OPTIONS, OPTION_COLORS)
    ):

        draw_violin_and_boxplot(
            ax=ax2,
            x_value=x_position,
            cloud_data=df[df["Anelasticity"] == anelasticity_option][
                "ocean_mean_trend_step_5"
            ].values,
            color=color,
            width=1,
        )

    ax2.set_ylabel("Barystatic GMSL trend (mm/yr)", fontsize=FONTSIZE_AXE_LABELS)
    ax2.grid()
    ax2.text(
        0.5,
        1.05,
        "B. Impat on solid Earth rheology on (2003 - 2022)\nGRACE-derived ocean mass trend",
        transform=ax2.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
        ha="center",
    )
    ax2.set_xticks(
        ticks=range(4),
        labels=ANELASTICITY_OPTIONS,
        rotation=45,
    )
    ax2.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax2.set_ylim(2.0, 2.9)
    ax2.set_xlim(-0.3, 3.7)
    ax2.set_yticks([2.0, 2.2, 2.4, 2.6, 2.8])

    # Panel C.
    ax3.text(
        0.5,
        1.05,
        "C. 2003-2022 ocean mass trend\n derived from altimetry - Argo",
        transform=ax3.transAxes,
        fontsize=FONTSIZE_PANEL_TITLES,
        fontweight="bold",
        ha="center",
    )
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_yticks(ax2.get_yticks())
    ax3.set_ylabel("Barystatic GMSL trend (mm/yr)", fontsize=FONTSIZE_AXE_LABELS)
    ax3.tick_params(
        axis="both", which="both", length=6, direction="inout", labelsize=FONTSIZE_TICKLABELS
    )
    ax3.yaxis.set_ticks_position("right")
    ax3.yaxis.set_label_position("right")
    ax3.grid()

    fig.patches.append(
        Rectangle(
            (0.64, 0.13),  # left, bottom (normalized figure coordinates)
            0.16,  # width
            0.32,  # height
            color="lightgray",
            zorder=-1,
            transform=fig.transFigure,
            linewidth=0,
        )
    )
    add_reference_values_and_altimetry(df=df, ax2=ax2, ax3=ax3, data=data)
    ax3.set_xticks(
        ticks=[0],
        labels=["Altimetry-ARGO"],
        rotation=45,
    )

    fig.savefig(figures_path.joinpath("figure_4.svg"), format="svg")
